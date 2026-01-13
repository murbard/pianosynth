# Project Status & Resumption Guide

## Current Architecture: Per-Key Optimization
We have transitioned from global functional forms to a **Flattend Per-Key Parameter** model (`PianoParamPerKey`). Each of the 88 piano keys now has its own independent set of physical parameters (decay, stiffness, hammer properties, etc.), allowing for perfect fitting of individual notes without being constrained by global polynomial curves.

## Status: Sparse Training Compelte
We have successfully implemented and verified the pipeline using a **sparse training strategy**:
1. **Refactoring**:
    - `physics.py` rewritten to accept flat per-key parameters.
    - **Strict ET**: Removed legacy stretch tuning logic; synthesis is now strictly 12-TET.
    - **Preprocessing**: Input audio is automatically trimmed of silence and resampled to exact 12-TET frequencies during loading.
2. **Sparse Run**:
    - Trained on a **subset of 27 notes** (equally spaced, alternating dynamics) to ~2000 active epochs.
    - Convergence verified (Loss ~2.4, down from >6.0).
3. **Current Artifacts** (Saved in Repo):
    - **Checkpoint**: `src/pianosynth/checkpoints/params_all_keys.pt` (Contains trained values for the 27 sparse notes; others are default).
    - **Plots**: `plots/*.png` showing the distribution of physical parameters across the keyboard.
    - **Audio**: `full_comparison.wav` demonstrating the current quality.

## Immediate Next Step: Full Dataset Training
The codebase is primed to train on the **full dataset** (all ~260 available audio samples). We have implemented a **Smart Initialization** logic that uses the results from the sparse run to accelerate the full run.

### How to Resume on New Machine
1. **Clone & Install**:
   ```bash
   git clone <repo_url>
   cd pianosynth
   poetry install
   ```
   *(Ensure `data/processed/` contains the `.pt` audio tensors. If not, re-run `scripts/preprocess.py` or transfer data).*

2. **Run Full Training**:
   ```bash
   PYTHONPATH=src poetry run python scripts/train_all_notes.py --full
   ```
   **Automatic Smart Init**: The script detects the `--full` flag and the existing `params_all_keys.pt` checkpoint. It will **automatically initialize** the untrained keys by interpolating parameter values from their nearest trained neighbors (from the sparse run). This provides a significantly better starting point than random initialization.

3. **Verify**:
   - Training should progress stably.
   - After completion, run `scripts/plot_parameters.py` to see the full, dense parameter distributions.
   - Run `scripts/generate_full_comparison.py` to render the final result.
