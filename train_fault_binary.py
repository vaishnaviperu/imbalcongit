"""
Train a simple binary Fault vs Normal classifier from preprocessed CWRU data.

Example:
    python3 train_fault_binary.py \
        --input processed/cwru_binary_fault_de_2048.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = np.load(path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    label_names = [str(name) for name in data["label_names"].tolist()]

    if label_names != ["normal", "fault"]:
        raise ValueError(
            "This training script expects a binary_fault dataset with "
            "label_names exactly equal to ['normal', 'fault']."
        )

    features = X.reshape(len(X), -1)
    if "stat_features" in data.files:
        stat_features = np.asarray(data["stat_features"], dtype=np.float32)
        features = np.concatenate([features, stat_features], axis=1)

    return features, y, label_names


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a simple Fault vs Normal model on preprocessed CWRU data.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the preprocessed .npz file")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    X, y, label_names = load_dataset(args.input)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("=" * 60)
    print("Binary Fault vs Normal Results")
    print("=" * 60)
    print(f"Input file : {args.input}")
    print(f"Train size : {len(X_train)}")
    print(f"Test size  : {len(X_test)}")
    print(f"Accuracy   : {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=label_names, digits=4))
    print("Confusion matrix:")
    print(matrix)


if __name__ == "__main__":
    main()
