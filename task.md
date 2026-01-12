# Piano Synthesis Optimization Tasks

- [x] **Project Structure & Data**
    - [x] Create project skeleton with `poetry`
    - [x] Implement data download script (`scripts/download_data.py`)
    - [x] Implement preprocessing script (`scripts/preprocess.py`)
    - [x] Create initialization/documentation for data (`DATA_SETUP.md`)

- [x] **Synthesis Engine**
    - [x] Implement additive synthesis in `src/pianosynth/synth.py`
    - [x] Refactor for full parameterization (Dictionary/JSON config)
    - [x] Create default parameters (`src/pianosynth/default_params.json`)
    - [x] Implement vectorized physics (`src/pianosynth/physics.py`)

- [x] **Optimization (Phase 1: Feature Matching)**
    - [x] Implement feature extraction (`scripts/analyze_samples.py`)
    - [ ] Optimize parameters against features (Pivot to Spectral due to artifacts)

- [x] **Optimization (Phase 2: Spectral Matching)**
    - [x] Implement Differentiable Renderer (`src/pianosynth/spectral.py`)
    - [x] Implement Multi-Resolution STFT Loss
    - [x] Implement Training Script (`scripts/train_spectral.py`)
    - [x] Refine Loss: Linear Amplitude Weighting (Solved "Spectral Soup")
    - [x] Refine Phase: Coherent Attack (Solved "Phaser" artifact)
    - [x] Refine Physics: Unison Support (Solved "Dead Tone" artifact)

- [x] **Verification & Reports**
    - [x] Generate scale comparison (`scripts/generate_scale_comparison.py`)
    - [x] Extract and print optimized parameters (`scripts/extract_report_data.py`)
    - [x] Create Walkthrough (`walkthrough.md`)
