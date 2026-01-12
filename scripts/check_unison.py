import torch
import json
from pianosynth.optimization import PianoParam
from pathlib import Path

PARAM_PATH = Path("src/pianosynth/params_spectral.pt")

def main():
    if not PARAM_PATH.exists():
        print("Params not found.")
        return
        
    model = PianoParam(device="cpu")
    checkpoint = torch.load(PARAM_PATH, map_location="cpu")
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
        
    ov = model.get_overrides()
    ov = {k: v.detach().item() if v.numel() == 1 else v.detach() for k, v in ov.items()}
    
    print("--- Diagnostic: Unison & Tuning ---")
    print(f"DC Base: {ov['dc_base']:.4f} (Default ~1.2)")
    print(f"DC Slope: {ov['dc_slope']:.4f} (Default ~1.8)")
    print(f"Random Detune Base: {ov['random_detune_base']:.4f} (Default ~0.1)")
    print(f"Random Detune Slope: {ov['random_detune_slope']:.4f} (Default ~0.1)")
    
    print("\n--- Diagnostic: Stretches ---")
    print(f"Bass Stretch: {ov['stretch_bass_amount']:.4f}")
    print(f"Treble Stretch: {ov['stretch_treble_amount']:.4f}")

if __name__ == "__main__":
    main()
