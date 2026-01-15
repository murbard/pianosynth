# Dataset Setup Instructions

This project uses the University of Iowa Musical Instrument Samples (Piano).

## 1. Download Data
We provide a script to automatically download the piano samples (~260 files).

```bash
PYTHONPATH=src poetry run python scripts/download_data.py
```

This will:
*   Identify all Piano AIFF files from the Iowa website.
*   Download them to `data/raw/iowa`.

## 2. Preprocess Data
After downloading, run the preprocessing script to trim silence, normalize audio, and align onsets.

```bash
PYTHONPATH=src poetry run python scripts/preprocess.py
```

This will:
*   Load raw AIFF files.
*   Detect onsets and trim leading silence.
*   Normalize amplitude (-1dB).
*   Save processed tensors to `data/processed`.

## 3. Analyze Features (Optional)
If you wish to re-run the initial feature extraction (for the heuristic optimization phase):

```bash
PYTHONPATH=src poetry run python scripts/analyze_samples.py
```

## 4. Train Model
To reproduce the spectral optimization:

```bash
PYTHONPATH=src poetry run python scripts/train_spectral.py
```
This trains the model for 1000 epochs using the processed data and saves `src/pianosynth/params_spectral.pt`.
