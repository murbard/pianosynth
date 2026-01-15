import torch
from pathlib import Path

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CPT_PATH = Path("src/pianosynth/checkpoints/params_best_ever.pt")
    
    print(f"Loading {CPT_PATH}")
    cpt = torch.load(CPT_PATH, map_location=device)
    state = cpt["model_state"]
    
    # Missing Keys: 21_pp, 21_mf, 22_mf, 102_mf
    missing_map = [
        # (Target Note, Target DynIdx, Source Note, Source DynIdx)
        # 21 (A0) missing pp(0), mf(1).
        # We can copy from 23 (B0) which has pp and mf? Or 22?
        # 22_pp exists. 22_ff exists. 22_mf missing.
        
        # Strategy:
        # 21_pp <- 22_pp
        # 21_mf <- 23_mf (since 22_mf is also missing)
        # 22_mf <- 23_mf
        # 102_mf <- 101_mf (Interpolate 101/103? Just copy 101 for safety)
        
        (21, 0, 22, 0), # 21pp from 22pp
        (21, 1, 23, 1), # 21mf from 23mf
        (22, 1, 23, 1), # 22mf from 23mf
        (102, 1, 101, 1) # 102mf from 101mf
    ]
    
    modified_keys = []
    
    for (tgt_m, tgt_d, src_m, src_d) in missing_map:
        idx_tgt_m = tgt_m - 21
        idx_src_m = src_m - 21
        
        print(f"Patching {tgt_m} [{tgt_d}] from {src_m} [{src_d}]...")
        
        for k, v in state.items():
            if v.ndim >= 1 and v.shape[0] == 88:
                # Shape [88, 3] usually
                # Copy src to tgt
                v[idx_tgt_m, tgt_d] = v[idx_src_m, src_d].clone()
                if k not in modified_keys: modified_keys.append(k)

    print(f"Patched {len(modified_keys)} parameters.")
    
    # Save
    backup = CPT_PATH.parent / "params_best_ever_pre_full_patch.bak"
    torch.save(cpt, backup)
    print(f"Backed up to {backup}")
    
    torch.save(cpt, CPT_PATH)
    print(f"Overwrote {CPT_PATH}")

if __name__ == "__main__":
    main()
