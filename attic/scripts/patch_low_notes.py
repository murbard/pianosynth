import torch
from pathlib import Path

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CPT_PATH = Path("src/pianosynth/checkpoints/params_best_ever.pt")
    
    print(f"Loading {CPT_PATH}")
    cpt = torch.load(CPT_PATH, map_location=device)
    state = cpt["model_state"]
    
    # We want to copy from MIDI 23 (Index 23-21=2) to MIDI 21 (0) and 22 (1)
    # Params are shape [88, 3] usually.
    # Start Note is 21.
    
    idx_src = 23 - 21 # 2
    idx_tgt1 = 21 - 21 # 0
    idx_tgt2 = 22 - 21 # 1
    
    modified_keys = []
    
    for k, v in state.items():
        # Check if param is per-key [88, ...]
        if v.ndim >= 1 and v.shape[0] == 88:
            # Copy src to tgt
            # We clone to avoid memory sharing? Not strictly needed for state_dict, but safer.
            v[idx_tgt1] = v[idx_src].clone()
            v[idx_tgt2] = v[idx_src].clone()
            modified_keys.append(k)
        elif v.ndim == 0:
            # Scalar param (if any, usually none in PianoParamPerKey)
            pass
        else:
             print(f"Skipping {k} with shape {v.shape}")

    print(f"Patched {len(modified_keys)} parameters for MIDI 21 and 22 using MIDI 23.")
    
    # Save
    # Backup first?
    backup = CPT_PATH.parent / "params_best_ever.bak"
    torch.save(cpt, backup)
    print(f"Backed up to {backup}")
    
    torch.save(cpt, CPT_PATH)
    print(f"Overwrote {CPT_PATH}")

if __name__ == "__main__":
    main()
