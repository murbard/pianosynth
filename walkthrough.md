# WaveRNN Implementation Walkthrough

## Overview
We implemented a small autoregressive WaveRNN model to generate piano waveforms.
The model conditions on:
- **Past Samples**: 2000 samples context (encoded via CNN).
- **Metadata**: Velocity, Log Pitch, Time.
- **Hidden State**: Recurrent state from previous step.

## Architecture
- **Context CNN**: 3-layer 1D CNN reducing 2000 samples to a size-16 vector.
- **RNN Core**: GRU (Hidden Size 80).
    - Input: [Context(16), Vel(1), Pitch(1), Time(1)] = 19 dim.
- **Output**: Gaussian Distribution (Mu, Sigma).

**Total Parameters**: ~25,930 (Hidden Size 80).

## Training
- **Dataset**: Full University of Iowa Piano Dataset (All notes, pp/mf/ff).
- **Strategy**: 
    - Random 100-step sequences.
    - Full-batch (Sequence training).
    - 5000 Epochs.
- **Progress**: Currently running full scale experiment.

## Results
- **Training Loss**: Converged to ~ -5.46 after 50 epochs (Single Note).
- **Artifacts**:
    - `results_single_note/wavernn_final.pt`: Trained Model
    - `results_single_note/preview_epoch_50.wav`: Generated Preview (0.5s)
    - `results_single_note/wavernn_loss.png`: Loss Curve

Train loss consistently decreased, indicating the model learned to model the waveform distribution. Previews generated every 10 epochs allow monitoring of audio quality evolution.

## Usage
### Train
```bash
PYTHONPATH=src poetry run python scripts/train_wavernn.py
```

### Generate
```bash
PYTHONPATH=src poetry run python scripts/generate_wavernn.py
```
