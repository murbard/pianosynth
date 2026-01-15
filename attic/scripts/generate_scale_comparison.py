import sys
import os
import torch
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from pianosynth.synth import piano_additive
from pianosynth.io import save_wav

PARAM_PATH = Path("src/pianosynth/params_spectral.pt")

def generate_segment(midis, overrides=None, device="cpu"):
    notes = []
    # dur=0.5 for scale speed
    for m in midis:
        print(f"  Midi {m}...")
        note = piano_additive(
            midi=m, 
            velocity=0.7, 
            dur=0.5, 
            device=device, 
            params_override=overrides
        )
        notes.append(note.cpu())
    return torch.cat(notes)

def main():
    print("Generating Comparison Scale...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    midis = [60, 62, 64, 65, 67, 69, 71, 72] # C Maj
    
    # 1. Unoptimized (Heuristic)
    print("Synthesizing Unoptimized...")
    scale_unopt = generate_segment(midis, overrides=None, device=device)
    
    # 2. Optimized
    print("Loading Optimized Params...")
    if not PARAM_PATH.exists():
        print("Params not found.")
        return
        
    # Load model structure
    from pianosynth.optimization import PianoParam
    model = PianoParam(device=device)
    
    checkpoint = torch.load(PARAM_PATH, map_location=device)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        # Fallback if saved differently
        model.load_state_dict(checkpoint)
        
    overrides = model.get_overrides()
    
    # We need to detach tensors in overrides for synthesis if they have grad
    overrides = {k: v.detach() for k, v in overrides.items()}

    print("Synthesizing Optimized...")
    scale_opt = generate_segment(midis, overrides=overrides, device=device)
    
    # Normalize segments independently to fair comparison
    scale_unopt = scale_unopt / (scale_unopt.abs().max() + 1e-6) * 0.9
    scale_opt = scale_opt / (scale_opt.abs().max() + 1e-6) * 0.9
    
    # Silence separator (1s)
    silence = torch.zeros(44100, dtype=torch.float32)
    
    combined = torch.cat([scale_unopt, silence, scale_opt])
    
    out_path = "scale_comparison.wav"
    save_wav(out_path, combined)
    print(f"Saved {out_path} (Unoptimized -> Silence -> Optimized)")

if __name__ == "__main__":
    main()
