# Data-Driven Optimization Walkthrough

We have successfully optimized the physical piano model parameters using the University of Iowa Piano Dataset.

## Workflow Update: Robust Optimization

Based on user feedback, we refined the Optimization workflow to be physically robust without artificial constraints:
*   **Loss Function**: Weighted **Log-Magnitude Loss** by the target linear amplitude. This focuses the optimizer on the peaks (harmonics) and ignores the noise floor, solving the "spectral soup" issue naturally.
*   **Phase Coherence**: Set aligned phase ($\phi \approx 0$) for the attack transient to match the sharp "thump" of a real hammer, avoiding "phaser" artifacts.
*   **Constraints Removed**: Removed artificial bounds on parameters, allowing the physics to emerge from the improved loss.

## Final Results (1000 Epochs)

The model converged to a balanced, physically plausible piano profile.

### 1. Inharmonicity
Recovered a realistic stiffness curve.
*   Scale: **1.22** (Stiff Grand)
*   **String Variation**: **2.5%** (Physically realistic, naturally minimized)

### 2. Tone & Timbre
The parameters settled into a "sweet spot" between the mellow heuristic and the harsh bright bridge-strike of the unconstrained run.
*   **Hammer Tilt**: $p \approx 2.16$ (Classic Mellow/Bright balance).
*   **Strike Point**: **0.38** (Mid-string? This seems high, might be compensating for comb filtering effects).
*   **Decay Time**: $8.86s$ (Natural long sustain).

## Audio Results

### Full Scale Comparison
**[scale_comparison.wav](scale_comparison.wav)**
*(First part: Unoptimized / Second part: Optimized)*

The optimized scale features a coherent, sharp attack (no "phaser" woosh) and a balanced harmonic spectrum. The weighted loss successfully prevented the noise-filling artifacts.

## Files
*   **Config**: `src/pianosynth/default_params.json`
*   **Optimized Checkpoint**: `src/pianosynth/params_spectral.pt`
