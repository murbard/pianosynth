import sys
import os
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from pianosynth.synth import piano_additive
from pianosynth.io import save_wav
from pianosynth.optimization import PianoParam

PARAM_PATH = Path("src/pianosynth/params_spectral.pt")

def generate_scale():
    print("Generating Optimized C Major scale (Full Engine)...")
    
    # Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not PARAM_PATH.exists():
        print("Error: Optimized params not found. Run train_params.py first.")
        return

    checkpoint = torch.load(PARAM_PATH, map_location=device)
    # Extract learned params from state dict
    state = checkpoint['model_state']
    
    # Pack into overrides dict for piano_additive
    # Keys must match what we added in synth.py
    overrides = {}
    
    if "logB_knots" in state:
        overrides["B_knots"] = torch.pow(10.0, state["logB_knots"])
        
    if "decay_A" in state: overrides["decay_A"] = state["decay_A"]
    if "decay_p" in state: overrides["decay_p"] = state["decay_p"]
    
    if "k0" in state: overrides["k0"] = state["k0"]
    if "k1" in state: overrides["k1"] = state["k1"]
    
    if "tilt_base" in state: overrides["tilt_base"] = state["tilt_base"]
    if "tilt_slope" in state: overrides["tilt_slope"] = state["tilt_slope"]

    print("Learned Parameters:")
    for k, v in overrides.items():
        if v.numel() > 1:
            print(f"  {k}: [Tensor shape {v.shape}]")
        else:
            print(f"  {k}: {v.item():.4f}")

    # C major scale: C4, D4, E4, F4, G4, A4, B4, C5
    midis = [60, 62, 64, 65, 67, 69, 71, 72]
    
    notes = []
    
    for i, m in enumerate(midis):
        print(f"Synthesizing midi {m}...")
        # 0.5s duration
        # Using full engine
        note = piano_additive(
            midi=m, 
            velocity=0.7, 
            dur=0.5, 
            device=device,
            params_override=overrides
        )
        notes.append(note.cpu())
    
    # Concatenate
    full_scale = torch.cat(notes)
    
    output_path = "scale_optimized.wav"
    print(f"Saving to {output_path}...")
    save_wav(output_path, full_scale)
    print("Done!")

if __name__ == "__main__":
    generate_scale()
