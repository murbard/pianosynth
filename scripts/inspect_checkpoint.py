import torch
import torch.nn.functional as F

def main():
    d = torch.load('src/pianosynth/params_spectral.pt')
    st = d['model_state']
    
    # Helper to print un-transformed latent params if needed, or transform them
    # decay_A in optimization.py was: 0.1 + softplus(param)
    dA = F.softplus(st['params.decay_A']) + 0.1
    print(f"Decay A (Optimized): {dA.item()}")
    
    # hammer_tilt_base was: param (no transform in getter, but softplus in get_overrides? No, just P())
    # Wait, optimization.py lines: overrides["tilt_base"] = P("hammer.tilt_base")
    # if I reverted constraints.
    # Let's just print the raw value.
    print(f"Hammer Tilt Base: {st['params.hammer_tilt_base'].item()}")
    print(f"Comb Mix: {st['params.hammer_comb_mix'].item()}")
    
    # Check Inharmonicity Scale (B_scale)
    # overrides["B_scale"] = SP("inharmonicity.B_scale")
    b_scale = F.softplus(st['params.inharmonicity_B_scale'])
    print(f"B Scale: {b_scale.item()}")
    
if __name__ == "__main__":
    main()
