"""
CWRU preprocessing pipeline.

This script converts raw CWRU .mat files into windowed samples plus labels.
The code is split into small classes so we can extend label logic, channels,
windowing, or output formats later without rewriting the full pipeline.

Example:
    python3 preprocess_cwru.py \
        --data-root data \
        --output processed/cwru_fault_type_de_2048.npz \
        --channel DE \
        --label-mode fault_type \
        --window-size 2048 \
        --stride 1024
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.io import loadmat
from scipy.stats import kurtosis, skew


VALID_EXTENSIONS = {".mat"}
SAMPLE_RATE_BY_FOLDER = {
    "normal": 12000,
    "drive_end_12k": 12000,
    "fan_end_12k": 12000,
    "drive_end_48k": 48000,
}
CHANNEL_KEY_SUFFIX = {
    "DE": "_DE_time",
    "FE": "_FE_time",
    "BA": "_BA_time",
}
WINDOW_OUTCOME_MODES = {"keep_all", "filter_informative", "tag_only"}


@dataclass(frozen=True)
class PreprocessConfig:
    data_root: Path
    output_path: Path
    channel: str = "DE"
    label_mode: str = "fault_type"
    window_size: int = 2048
    stride: int = 1024
    normalization: str = "zscore"
    window_outcome_mode: str = "keep_all"
    kurtosis_threshold: float = 3.5
    skewness_threshold: float = 1.0
    min_windows_per_file: int = 1
    limit_files: int | None = None
    summary_only: bool = False


@dataclass(frozen=True)
class WindowFeatures:
    kurtosis: float
    skewness: float
    is_informative: bool


@dataclass(frozen=True)
class WindowRecord:
    window: np.ndarray
    features: WindowFeatures


@dataclass(frozen=True)
class FileMetadata:
    path: Path
    folder: str
    filename: str
    sample_rate: int
    load_hp: int
    fault_code: str
    fault_family: str
    fault_size_mils: int | None
    outer_race_position: str | None
    rpm: int | None = None


class LabelScheme(Protocol):
    name: str

    def make_label(self, metadata: FileMetadata) -> str:
        ...

    @property
    def ordered_labels(self) -> list[str] | None:
        ...


class FaultTypeScheme:
    name = "fault_type"
    ordered_labels = None

    def make_label(self, metadata: FileMetadata) -> str:
        return metadata.fault_family


class FaultTypeLoadScheme:
    name = "fault_type_load"
    ordered_labels = None

    def make_label(self, metadata: FileMetadata) -> str:
        return f"{metadata.fault_family}_load{metadata.load_hp}"


class CompositeScheme:
    name = "composite"
    ordered_labels = None

    def make_label(self, metadata: FileMetadata) -> str:
        size = "none" if metadata.fault_size_mils is None else str(metadata.fault_size_mils)
        position = metadata.outer_race_position or "none"
        return (
            f"{metadata.fault_family}"
            f"__size{size}"
            f"__load{metadata.load_hp}"
            f"__pos{position}"
            f"__sr{metadata.sample_rate}"
        )


class BinaryFaultScheme:
    name = "binary_fault"
    ordered_labels = ["normal", "fault"]

    def make_label(self, metadata: FileMetadata) -> str:
        if metadata.fault_family == "normal":
            return "normal"
        return "fault"


class LabelSchemeFactory:
    SCHEMES = {
        "binary_fault": BinaryFaultScheme,
        "fault_type": FaultTypeScheme,
        "fault_type_load": FaultTypeLoadScheme,
        "composite": CompositeScheme,
    }

    @classmethod
    def create(cls, name: str) -> LabelScheme:
        if name not in cls.SCHEMES:
            available = ", ".join(sorted(cls.SCHEMES))
            raise ValueError(f"Unsupported label mode '{name}'. Available: {available}")
        return cls.SCHEMES[name]()


class CWRUMetadataParser:
    FAULT_RE = re.compile(r"^(?P<kind>Normal|B|IR|OR)(?P<size>\d{3})?(?:@(?P<pos>\d+))?_(?P<load>\d+)$")

    FAMILY_MAP = {
        "Normal": "normal",
        "B": "ball",
        "IR": "inner_race",
        "OR": "outer_race",
    }

    def parse(self, path: Path) -> FileMetadata:
        stem = path.stem
        match = self.FAULT_RE.match(stem)
        if not match:
            raise ValueError(f"Could not parse CWRU filename: {path.name}")

        fault_code = match.group("kind")
        folder = path.parent.name
        sample_rate = SAMPLE_RATE_BY_FOLDER.get(folder)
        if sample_rate is None:
            raise ValueError(f"Unknown CWRU folder '{folder}' for {path}")

        size_str = match.group("size")
        pos_str = match.group("pos")
        load_str = match.group("load")

        return FileMetadata(
            path=path,
            folder=folder,
            filename=path.name,
            sample_rate=sample_rate,
            load_hp=int(load_str),
            fault_code=fault_code,
            fault_family=self.FAMILY_MAP[fault_code],
            fault_size_mils=int(size_str) if size_str else None,
            outer_race_position=pos_str,
        )


class MatSignalReader:
    def __init__(self, channel: str) -> None:
        channel = channel.upper()
        if channel not in CHANNEL_KEY_SUFFIX:
            available = ", ".join(CHANNEL_KEY_SUFFIX)
            raise ValueError(f"Unsupported channel '{channel}'. Available: {available}")
        self.channel = channel
        self.key_suffix = CHANNEL_KEY_SUFFIX[channel]

    def load_signal(self, metadata: FileMetadata) -> tuple[np.ndarray, FileMetadata]:
        raw = loadmat(metadata.path)
        signal_key = self._find_signal_key(raw)
        rpm_key = self._find_rpm_key(raw)

        signal = np.asarray(raw[signal_key]).squeeze().astype(np.float32)
        rpm = int(np.asarray(raw[rpm_key]).squeeze()) if rpm_key is not None else None

        return signal, FileMetadata(**{**asdict(metadata), "rpm": rpm})

    def _find_signal_key(self, mat: dict[str, object]) -> str:
        keys = sorted(k for k in mat.keys() if not k.startswith("__"))
        for key in keys:
            if key.endswith(self.key_suffix):
                return key
        available = ", ".join(keys)
        raise KeyError(f"Could not find channel {self.channel} in MAT file. Keys: {available}")

    @staticmethod
    def _find_rpm_key(mat: dict[str, object]) -> str | None:
        keys = sorted(k for k in mat.keys() if not k.startswith("__"))
        for key in keys:
            if key.endswith("RPM"):
                return key
        return None


class SignalWindowizer:
    def __init__(self, window_size: int, stride: int, normalization: str) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if normalization not in {"zscore", "minmax", "none"}:
            raise ValueError("normalization must be one of: zscore, minmax, none")
        self.window_size = window_size
        self.stride = stride
        self.normalization = normalization

    def transform(self, signal: np.ndarray) -> np.ndarray:
        if signal.ndim != 1:
            raise ValueError(f"Expected 1D signal, got shape {signal.shape}")
        if len(signal) < self.window_size:
            return np.empty((0, self.window_size), dtype=np.float32)

        windows = []
        for start in range(0, len(signal) - self.window_size + 1, self.stride):
            window = signal[start : start + self.window_size].copy()
            windows.append(self._normalize(window))

        return np.stack(windows).astype(np.float32)

    def _normalize(self, window: np.ndarray) -> np.ndarray:
        if self.normalization == "none":
            return window
        if self.normalization == "zscore":
            std = np.std(window)
            return (window - np.mean(window)) / (std + 1e-8)
        min_v = np.min(window)
        max_v = np.max(window)
        return (window - min_v) / (max_v - min_v + 1e-8)


class WindowFeatureExtractor:
    def __init__(self, kurtosis_threshold: float, skewness_threshold: float) -> None:
        self.kurtosis_threshold = kurtosis_threshold
        self.skewness_threshold = skewness_threshold

    def analyze(self, window: np.ndarray) -> WindowFeatures:
        kurtosis_value = float(kurtosis(window, fisher=False, bias=False))
        skewness_value = float(skew(window, bias=False))
        is_informative = (
            kurtosis_value >= self.kurtosis_threshold
            or abs(skewness_value) >= self.skewness_threshold
        )
        return WindowFeatures(
            kurtosis=kurtosis_value,
            skewness=skewness_value,
            is_informative=is_informative,
        )


class InformativeWindowSelector:
    def __init__(self, mode: str) -> None:
        if mode not in WINDOW_OUTCOME_MODES:
            available = ", ".join(sorted(WINDOW_OUTCOME_MODES))
            raise ValueError(f"Unsupported window outcome mode '{mode}'. Available: {available}")
        self.mode = mode

    def keep(self, features: WindowFeatures) -> bool:
        if self.mode == "filter_informative":
            return features.is_informative
        return True


class DatasetWriter:
    def save(
        self,
        output_path: Path,
        X: np.ndarray,
        y: np.ndarray,
        labels: list[str],
        metadata_rows: list[dict[str, object]],
        feature_array: np.ndarray,
        config: PreprocessConfig,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            X=X,
            y=y,
            label_names=np.array(labels, dtype=object),
            metadata=np.array(metadata_rows, dtype=object),
            stat_features=feature_array[:, :2],
            window_features=feature_array,
        )

        manifest_path = output_path.with_suffix(".json")
        manifest = {
            "output_path": str(output_path),
            "num_samples": int(len(y)),
            "window_size": config.window_size,
            "stride": config.stride,
            "channel": config.channel,
            "normalization": config.normalization,
            "label_mode": config.label_mode,
            "window_outcome_mode": config.window_outcome_mode,
            "kurtosis_threshold": config.kurtosis_threshold,
            "skewness_threshold": config.skewness_threshold,
            "labels": labels,
            "stat_feature_names": ["kurtosis", "skewness"],
            "window_feature_names": ["kurtosis", "skewness", "is_informative"],
            "files_seen": sorted({row["source_file"] for row in metadata_rows}),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


class CWRUPreprocessor:
    def __init__(
        self,
        config: PreprocessConfig,
        metadata_parser: CWRUMetadataParser,
        signal_reader: MatSignalReader,
        windowizer: SignalWindowizer,
        feature_extractor: WindowFeatureExtractor,
        selector: InformativeWindowSelector,
        label_scheme: LabelScheme,
        writer: DatasetWriter,
    ) -> None:
        self.config = config
        self.metadata_parser = metadata_parser
        self.signal_reader = signal_reader
        self.windowizer = windowizer
        self.feature_extractor = feature_extractor
        self.selector = selector
        self.label_scheme = label_scheme
        self.writer = writer

    def run(self) -> None:
        files = self._discover_files()
        print(f"[Discover] Found {len(files)} raw files under {self.config.data_root}")

        if self.config.summary_only:
            self._print_summary(files)
            return

        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        metadata_rows: list[dict[str, object]] = []
        feature_rows: list[list[float]] = []
        ordered_labels = getattr(self.label_scheme, "ordered_labels", None)
        label_to_index: dict[str, int] = (
            {label: index for index, label in enumerate(ordered_labels)}
            if ordered_labels is not None
            else {}
        )

        for path in files:
            file_meta = self.metadata_parser.parse(path)
            try:
                signal, file_meta = self.signal_reader.load_signal(file_meta)
            except Exception as exc:
                print(f"[Skip] {path.name}: failed to read raw file ({exc})")
                continue
            window_records = self._build_window_records(signal)

            if len(window_records) < self.config.min_windows_per_file:
                print(
                    f"[Skip] {path.name}: only {len(window_records)} windows, "
                    f"minimum is {self.config.min_windows_per_file}"
                )
                continue

            label_name = self.label_scheme.make_label(file_meta)
            if label_name not in label_to_index:
                label_to_index[label_name] = len(label_to_index)
            label_index = label_to_index[label_name]

            windows = np.stack([record.window for record in window_records]).astype(np.float32)
            X_parts.append(windows)
            y_parts.append(np.full(len(window_records), label_index, dtype=np.int64))

            informative_count = 0
            for window_index, record in enumerate(window_records):
                if record.features.is_informative:
                    informative_count += 1
                metadata_rows.append(
                    {
                        "source_file": file_meta.filename,
                        "folder": file_meta.folder,
                        "label_name": label_name,
                        "load_hp": file_meta.load_hp,
                        "fault_family": file_meta.fault_family,
                        "fault_size_mils": file_meta.fault_size_mils,
                        "outer_race_position": file_meta.outer_race_position,
                        "sample_rate": file_meta.sample_rate,
                        "rpm": file_meta.rpm,
                        "channel": self.config.channel,
                        "window_index": window_index,
                        "kurtosis": record.features.kurtosis,
                        "skewness": record.features.skewness,
                        "is_informative": record.features.is_informative,
                    }
                )
                feature_rows.append(
                    [
                        record.features.kurtosis,
                        record.features.skewness,
                        float(record.features.is_informative),
                    ]
                )

            print(
                f"[File] {file_meta.filename:<16} "
                f"label={label_name:<28} windows={len(window_records):>4} "
                f"informative={informative_count:>4}"
            )

        if not X_parts:
            raise RuntimeError("No windows were generated. Check channel, window_size, and raw files.")

        label_names = [label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])]
        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        feature_array = np.asarray(feature_rows, dtype=np.float32)

        self.writer.save(
            self.config.output_path,
            X,
            y,
            label_names,
            metadata_rows,
            feature_array,
            self.config,
        )

        print(f"\n[Done] Saved dataset to {self.config.output_path}")
        print(f"[Done] Samples: {len(y)} | Classes: {len(label_names)} | Shape: {X.shape}")
        informative_total = int(feature_array[:, 2].sum()) if len(feature_array) else 0
        print(f"[Done] Informative windows: {informative_total} / {len(feature_array)}")
        for label_index, label_name in enumerate(label_names):
            print(f"       {label_name:<28} {(y == label_index).sum():>6}")

    def _build_window_records(self, signal: np.ndarray) -> list[WindowRecord]:
        raw_windows = self.windowizer.transform(signal)
        records: list[WindowRecord] = []
        for window in raw_windows:
            features = self.feature_extractor.analyze(window)
            if not self.selector.keep(features):
                continue
            records.append(WindowRecord(window=window, features=features))
        return records

    def _discover_files(self) -> list[Path]:
        if not self.config.data_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {self.config.data_root}")

        files = [
            path
            for path in sorted(self.config.data_root.rglob("*"))
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        ]
        if self.config.limit_files is not None:
            files = files[: self.config.limit_files]
        return files

    def _print_summary(self, files: list[Path]) -> None:
        counts: dict[str, int] = {}
        for path in files:
            metadata = self.metadata_parser.parse(path)
            label = self.label_scheme.make_label(metadata)
            counts[label] = counts.get(label, 0) + 1

        print("[Summary] Raw files per label")
        for label, count in sorted(counts.items()):
            print(f"  {label:<28} {count:>4}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess raw CWRU .mat files into windowed arrays.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("processed/cwru_preprocessed.npz"))
    parser.add_argument("--channel", choices=sorted(CHANNEL_KEY_SUFFIX), default="DE")
    parser.add_argument(
        "--label-mode",
        choices=sorted(LabelSchemeFactory.SCHEMES),
        default="fault_type",
        help="How to turn file metadata into class labels.",
    )
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--normalization", choices=["zscore", "minmax", "none"], default="zscore")
    parser.add_argument(
        "--window-outcome-mode",
        choices=sorted(WINDOW_OUTCOME_MODES),
        default="keep_all",
        help="keep all windows, keep only informative windows, or keep all with informative tagging",
    )
    parser.add_argument("--kurtosis-threshold", type=float, default=3.5)
    parser.add_argument("--skewness-threshold", type=float, default=1.0)
    parser.add_argument("--min-windows-per-file", type=int, default=1)
    parser.add_argument("--limit-files", type=int)
    parser.add_argument("--summary-only", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = PreprocessConfig(
        data_root=args.data_root,
        output_path=args.output,
        channel=args.channel,
        label_mode=args.label_mode,
        window_size=args.window_size,
        stride=args.stride,
        normalization=args.normalization,
        window_outcome_mode=args.window_outcome_mode,
        kurtosis_threshold=args.kurtosis_threshold,
        skewness_threshold=args.skewness_threshold,
        min_windows_per_file=args.min_windows_per_file,
        limit_files=args.limit_files,
        summary_only=args.summary_only,
    )

    pipeline = CWRUPreprocessor(
        config=config,
        metadata_parser=CWRUMetadataParser(),
        signal_reader=MatSignalReader(config.channel),
        windowizer=SignalWindowizer(
            window_size=config.window_size,
            stride=config.stride,
            normalization=config.normalization,
        ),
        feature_extractor=WindowFeatureExtractor(
            kurtosis_threshold=config.kurtosis_threshold,
            skewness_threshold=config.skewness_threshold,
        ),
        selector=InformativeWindowSelector(config.window_outcome_mode),
        label_scheme=LabelSchemeFactory.create(config.label_mode),
        writer=DatasetWriter(),
    )
    pipeline.run()


if __name__ == "__main__":
    main()
