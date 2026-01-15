import torch
from pianosynth.optimization import PianoParam
from pathlib import Path

PARAM_PATH = Path("src/pianosynth/params_spectral.pt")

def main():
    if not PARAM_PATH.exists():
        print("Optimized params not found.")
        return
    model = PianoParam(device="cpu")
    checkpoint = torch.load(PARAM_PATH, map_location="cpu")
    
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
        
    ov = model.get_overrides()
    # Detach
    ov = {k: v.detach().item() if v.numel() == 1 else v.detach() for k, v in ov.items()}
    
    print("\n--- Optimized Parameters (Spectral) ---")
    
    print(f"\nInharmonicity B (Knots):")
    print(f"{ov['B_knots']}")
    print(f"Scale: {ov['B_scale']:.4f}, Var: {ov['string_variation_std']:.4f}")
    
    print(f"\nDecay:")
    print(f"A={ov['A']:.2f}, p={ov['p']:.2f}")
    print(f"k0={ov['k0']:.4f}, k1={ov['k1']:.4f}")
    print(f"Fast Divisor: {ov['tau_fast_divisor']:.2f}")

    print(f"\nHammer:")
    print(f"Tilt: base={ov['tilt_base']:.2f}, slope={ov['tilt_slope']:.2f}")
    print(f"Comb: base={ov['comb_base']:.2f}, mix={ov['comb_mix']:.2f}")
    print(f"Cutoff: min={ov['fc_min']:.0f}, max={ov['fc_max']:.0f}")
    
    print(f"\nStrike Point:")
    print(f"Bass={ov['xh_bass']:.4f}, Treble={ov['xh_treble']:.4f}")
    
    print(f"\nBody Filter:")
    print(f"HP: {ov['highpass_freq']:.0f}Hz, LP: {ov['lowpass_freq']:.0f}Hz")

    print("\n---------------------------------------")

if __name__ == "__main__":
    main()
