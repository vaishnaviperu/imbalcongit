# Programmer Guide

This file is the human-oriented map of the codebase. It is meant for anyone who opens the repo and needs to understand what exists, what is experimental, and where to extend things next.

## Purpose

This repository is for bearing-fault experiments on the CWRU dataset, with two main tracks:

1. Raw dataset acquisition from the CWRU website.
2. Preprocessing and modeling for imbalance-aware representation learning.

Right now, the repo contains:

- a downloader for raw `.mat` files
- a starter contrastive-learning experiment
- a reusable preprocessing pipeline for turning raw signals into windowed arrays
- a beginner-friendly binary Fault vs Normal training script

## Current File Layout

Top-level files:

- [cwru_downloader.py](/Users/vaishnaviperu/Documents/cwr_ballbearing/cwru_downloader.py)
  Downloads labeled CWRU `.mat` files from the official dataset pages and stores them under `data/`.

- [preprocess_cwru.py](/Users/vaishnaviperu/Documents/cwr_ballbearing/preprocess_cwru.py)
  Main preprocessing pipeline. Converts raw `.mat` files into windowed datasets with labels, metadata, and per-window statistical features such as kurtosis and skewness.

- [imbalcon.py](/Users/vaishnaviperu/Documents/cwr_ballbearing/imbalcon.py)
  Self-contained starter experiment for contrastive learning under class imbalance. At the moment this is still a standalone prototype rather than a fully integrated training pipeline over the real CWRU preprocessing outputs.

- [train_fault_binary.py](/Users/vaishnaviperu/Documents/cwr_ballbearing/train_fault_binary.py)
  Loads a `binary_fault` preprocessed dataset, performs a train/test split, trains a simple RandomForest model, and prints accuracy, a classification report, and a confusion matrix.

- [setup.sh](/Users/vaishnaviperu/Documents/cwr_ballbearing/setup.sh)
  Creates a virtual environment and installs the Python dependencies.

- [requirements.txt](/Users/vaishnaviperu/Documents/cwr_ballbearing/requirements.txt)
  Frozen Python package list from the environment setup.

- [imbalcon_results.png](/Users/vaishnaviperu/Documents/cwr_ballbearing/imbalcon_results.png)
  Output visualization from the starter experiment.

Folders:

- [data](/Users/vaishnaviperu/Documents/cwr_ballbearing/data)
  Raw CWRU dataset files grouped by download category.

- [processed](/Users/vaishnaviperu/Documents/cwr_ballbearing/processed)
  Intended home for preprocessed `.npz` datasets and their manifest `.json` files.

## Raw Data Structure

The downloader organizes data into:

- `data/normal`
- `data/drive_end_12k`
- `data/drive_end_48k`
- `data/fan_end_12k`

Examples of filenames:

- `Normal_0.mat`
- `B007_0.mat`
- `IR014_2.mat`
- `OR021@6_1.mat`

Current filename meaning:

- `Normal`, `B`, `IR`, `OR`
  Fault family: normal, ball, inner race, outer race

- `007`, `014`, `021`
  Fault size in mils when present

- `@3`, `@6`, `@12`
  Outer-race fault position when present

- trailing `_0`, `_1`, `_2`, `_3`
  Motor load condition

## How Preprocessing Works

The preprocessing code is intentionally modular so later changes can be local instead of invasive.

### Main Classes

Inside [preprocess_cwru.py](/Users/vaishnaviperu/Documents/cwr_ballbearing/preprocess_cwru.py):

- `PreprocessConfig`
  Immutable run configuration for data root, output path, channel, label mode, window size, stride, and normalization.

- `FileMetadata`
  Parsed metadata for one raw file, including fault family, load, sample rate, optional fault size, optional outer-race position, and optional RPM.

- `CWRUMetadataParser`
  Converts CWRU filenames into structured metadata.

- `MatSignalReader`
  Loads a `.mat` file and extracts one vibration channel such as `DE`, `FE`, or `BA`.

- `SignalWindowizer`
  Splits one long 1D signal into fixed windows and normalizes each window.

- `WindowFeatureExtractor`
  Computes per-window kurtosis and skewness, then decides whether a window is informative based on configurable thresholds.

- `InformativeWindowSelector`
  Applies the configured outcome mode:
  keep all windows, filter to informative windows only, or keep all while tagging informative status.

- `LabelScheme` plus implementations
  Encodes metadata into class names. Current implementations:
  `binary_fault`, `fault_type`, `fault_type_load`, and `composite`.

- `DatasetWriter`
  Saves final arrays to compressed `.npz` and writes a `.json` manifest.

- `CWRUPreprocessor`
  Orchestrates discovery, parsing, signal loading, windowing, label assignment, and writing.

### Data Flow

The intended flow is:

1. Discover all `.mat` files under `data/`.
2. Parse filename metadata.
3. Load one chosen channel from the file contents.
4. Convert the long time series into overlapping or non-overlapping windows.
5. Normalize each window.
6. Compute per-window kurtosis and skewness.
7. Decide whether the window is informative using threshold rules.
8. Map metadata into label names.
7. Save:
   `X`: windowed signal array
   `y`: integer class ids
   `label_names`: class-name lookup
   `metadata`: per-window provenance rows
   `stat_features`: per-window statistical features with kurtosis and skewness
   `window_features`: per-window feature array with kurtosis, skewness, and informative flag

## Running The Pipeline

Environment setup:

```bash
./setup.sh
source .venv/bin/activate
```

Download raw data:

```bash
python3 cwru_downloader.py
```

Inspect raw-label distribution only:

```bash
python3 preprocess_cwru.py --summary-only --label-mode fault_type
```

Build a preprocessed dataset:

```bash
python3 preprocess_cwru.py \
  --data-root data \
  --output processed/cwru_fault_type_de_2048.npz \
  --channel DE \
  --label-mode fault_type \
  --window-size 2048 \
  --stride 1024 \
  --window-outcome-mode tag_only \
  --kurtosis-threshold 3.5 \
  --skewness-threshold 1.0
```

Alternative label setups:

```bash
python3 preprocess_cwru.py --label-mode binary_fault
python3 preprocess_cwru.py --label-mode fault_type_load
python3 preprocess_cwru.py --label-mode composite
```

Build a binary Fault vs Normal dataset:

```bash
python3 preprocess_cwru.py \
  --data-root data \
  --output processed/cwru_binary_fault_de_2048.npz \
  --channel DE \
  --label-mode binary_fault \
  --window-size 2048 \
  --stride 1024 \
  --window-outcome-mode tag_only \
  --kurtosis-threshold 3.5 \
  --skewness-threshold 1.0
```

Train a simple binary model:

```bash
python3 train_fault_binary.py \
  --input processed/cwru_binary_fault_de_2048.npz
```

Filter to informative windows only:

```bash
python3 preprocess_cwru.py \
  --window-outcome-mode filter_informative \
  --kurtosis-threshold 4.0 \
  --skewness-threshold 1.25
```

## Extending The Codebase

This repo is set up so new work usually belongs in one of these places:

### Add a new labeling strategy

Edit [preprocess_cwru.py](/Users/vaishnaviperu/Documents/cwr_ballbearing/preprocess_cwru.py):

- add a new class implementing `make_label`
- register it in `LabelSchemeFactory.SCHEMES`

This is the right place for experiments like:

- normal vs fault
- fault family only
- fault family plus load
- fault family plus severity
- domain labels for transfer learning

### Add a new signal channel or sensor rule

Edit `CHANNEL_KEY_SUFFIX` and `MatSignalReader`.

This is the right place if later files include new signal names or if channel selection becomes more complex than suffix matching.

### Add a new windowing or normalization method

Edit `SignalWindowizer`.

Good future additions:

- RMS normalization
- bandpass filtering before windowing
- resampling between `12k` and `48k`
- window rejection for low-energy segments

### Add a new informative-window rule

Edit `WindowFeatureExtractor` or `InformativeWindowSelector`.

This is the right place for:

- different kurtosis or skewness logic
- AND-vs-OR threshold rules
- extra features like RMS, crest factor, or entropy
- learned ranking scores instead of hard thresholds

### Add train/validation/test splitting

Right now preprocessing writes one dataset with labels and metadata. If we want reproducible splits, the cleanest path is to add a dedicated splitter layer rather than mixing split logic into the raw file reader.

Possible future module:

- `dataset_splits.py`

Possible responsibilities:

- stratified splitting
- group-aware splitting by source file
- load-aware or RPM-aware evaluation
- imbalance-preserving train/test generation

### Integrate with the modeling code

Right now [imbalcon.py](/Users/vaishnaviperu/Documents/cwr_ballbearing/imbalcon.py) is still a prototype experiment. The long-term direction should be:

1. preprocess raw data with `preprocess_cwru.py`
2. load the resulting `.npz`
3. train baseline SimCLR
4. train imbalance-aware variant
5. compare macro-F1 and representation quality

That integration has not been done yet in a clean reusable module.

## Known Project State

Important current truths:

- The raw downloader and preprocessing pipeline are real and usable.
- The preprocessing pipeline can now emit a direct `binary_fault` dataset for quick ML validation.
- The repository now has a simple binary training script for quick sanity checks.
- The starter `imbalcon.py` experiment exists and has shown a positive synthetic result, but it is not yet the final real-data training pipeline.
- The repository does not yet have a unified training entry point that consumes `processed/*.npz`.
- There are no formal tests yet.

## Suggested Next Refactor Steps

If development continues, the most valuable structural improvements are:

1. Add a `train_cwru.py` that reads preprocessed outputs.
2. Move reusable dataset-loading helpers into a small `src/` or `cwru/` package.
3. Add split logic with group-aware evaluation by source file.
4. Add a small validation script that checks dataset shapes, class counts, and label mappings.
5. Add tests for filename parsing and channel extraction.

## Maintenance Rule

This file should be updated whenever the repo structure, data flow, or extension points change.

In practice, update this guide when:

- a new script is added
- a new preprocessing class or label mode is introduced
- a new window feature, threshold, or outcome mode is introduced
- the training pipeline changes
- outputs or folder conventions change
- experiments move from prototype to production-style structure
