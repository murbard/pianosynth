# Implementation Plan - Data-Driven Piano Optimization

We will tune the parametric piano model by optimizing its parameters to match the properties of the University of Iowa Piano dataset.

## User Review Required

> [!IMPORTANT]
> - **Optimization Strategy**: Resynthesis/Analysis-first.
> - **Compute**: GPU (`cuda`) if available.
> - **Dynamics**: Using `pp`, `mf`, `ff`.
> - **Alignment Strategy**:
>     - **Volume**: We will **learn** the exact velocity $v$ for each sample (initialized at 0.3/0.6/0.9).
>     - **Silence**: Rigid onset detection during preprocessing to trim start.
>     - **Length**: Trim end based on Noise Floor (-60dB); synthesis generates exact duration of trimmed sample.

## Proposed Changes

### 1. Data Acquisition & Preprocessing
- **Script**: `scripts/download_data.py`
    - Parse `iowa_piano.html`.
    - Parallel download all `pp`, `mf`, `ff` samples.
- **Script**: `scripts/preprocess.py`
    - **Onset Detection**: Use `librosa.onset.onset_detect` (or energy threshold) + backtrack to finding the "start of attack".
        - Hard trim the file so sample 0 is the attack.
    - **Silence Trimming**: Find the point where energy drops below -60dB (or noise floor) and trim the end.
    - **Normalization**: Normalize peak to -1.0 dB.
    - Output: `data/processed/{note}_{dynamic}.pt` (Tensor)

### 2. Feature Extraction (The "Analysis" Step)
- **Script**: `scripts/analyze_samples.py`
    - For each processed sample:
        - **Refined $f_0$**: FFT peak interpolation.
        - **Partial Tracking**:
            - Track partials $n=1..N$.
            - Fit linear regression to $\log(\text{amplitude})$ of each partial to find **Decay Rate** $\tau_n$ and **Initial Amplitude** $A_{0,n}$.
            - Measure precise frequency $f_n$ for **Inharmonicity** fitting.
    - Save dataset: `Dict[Note, Dynamics] -> {partials_freq, partials_decay, partials_amp}`.

### 3. Direct Spectral Optimization (New Strategy)
> [!NOTE]
> We pivoted to this strategy after feature-based matching produced "plucked" artifacts.

- **File**: `pianosynth/train_spectral.py` (New)
    - **Methodology**:
        - Optimize parameters by minimizing the **Multi-Resolution STFT distance** between Synthesized Audio and Real Audio.
        - Bypasses intermediate feature extraction errors.
    - **Differentiable Renderer**:
        - Implement a fully differentiable `piano_additive_render` function compatible with Autograd.
        - Must be fast enough for inner-loop training.
    - **Loss Function**:
        - Multi-Resolution STFT Loss (sc\_loss + mag\_loss).
        - Focus on spectral envelope and temporal decay matching.

### 4. Verification
- **Script**: `scripts/compare.py`
    - Visualize Spectrograms (Model vs Real).
    - Listen to `scale_comparison.wav`.

## Verification Plan

### Automated
- [ ] `pytest tests/test_preprocess.py`: Verify onset detector doesn't cut off attacks.
- [ ] `pytest tests/test_fitting.py`: Verify we can overfit a single note perfectly.

### Manual
- [ ] **Alignment Check**: Visually inspect waveforms of preprocessed data to ensure concise start times.
- [ ] **Velocity Check**: Inspect the learned $v$ values. Do `ff` samples actually map to high $v$ (~0.9)?
