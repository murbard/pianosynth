import torch
import pandas as pd
from pathlib import Path
from pianosynth.optimization_batch import PianoParamPerKey

CHECKPOINT = Path("src/pianosynth/checkpoints/params_categorical.pt")
Patched_CHECKPOINT = Path("src/pianosynth/checkpoints/params_categorical_patched.pt")
CSV_PATH = Path("results_categorical/neighbor_improvements.csv")

def main():
    device = "cpu"
    
    if not CHECKPOINT.exists():
        print("Original checkpoint not found.")
        return
        
    if not CSV_PATH.exists():
        print("Improvements CSV not found.")
        return
        
    print(f"Loading {CHECKPOINT}...")
    cpt = torch.load(CHECKPOINT, map_location=device)
    state = cpt["model_state"]
    
    # Init model structure to know param shapes
    model = PianoParamPerKey(device=device)
    # We work directly on the state dict
    
    df = pd.read_csv(CSV_PATH)
    print(f"Found {len(df)} proposed improvements.")
    
    dyn_map = {'pp': 0, 'mf': 1, 'ff': 2}
    
    # Sort to apply? Or apply simultaneously?
    # To avoid chaining updates (copying already-copied garbage or good stuff),
    # we should read from 'old_state' and write to 'new_state'.
    # BUT, the parameters are tensors. We need to clone the state first.
    
    new_state = {k: v.clone() for k, v in state.items()}
    
    apply_count = 0
    
    for idx, row in df.iterrows():
        d_name = row['Dynamic']
        target_note = int(row['Note'])
        source_note = int(row['SourceNote'])
        imp = row['Improvement']
        
        # Threshold? User said "update based on best".
        # Let's apply all improvements > 0.
        if imp <= 0: continue
        
        d_idx = dyn_map[d_name]
        
        # Indices in tensor (Note 21 is index 0)
        t_idx = target_note - 21
        s_idx = source_note - 21
        
        # Validate
        if t_idx < 0 or t_idx > 87 or s_idx < 0 or s_idx > 87:
            continue
            
        # Copy for ALL parameters
        # Params are shape [88, 3] or [88] sometimes?
        # Model definition says (88, 3) for most.
        # Physics constants might be different but PianoParamPerKey mostly uses (88,3).
        
        for p_name, p_val in state.items():
            # Check shape
            # Most params are [88, 3].
            # Some might be different?
            if p_val.ndim == 2 and p_val.shape == (88, 3):
                # Copy src -> target
                # We read from ORIGINAL state to avoid dependency chains for now
                src_val = state[p_name][s_idx, d_idx]
                new_state[p_name][t_idx, d_idx] = src_val
            elif p_val.ndim == 1 and p_val.shape[0] == 88:
                 # Should we update shared params? 
                 # PianoParamPerKey generally expands everything to (88,3) in the register functions
                 # but internally store them.
                 # Actually, look at optimization_batch.py:
                 # self.params[name] = nn.Parameter(tensor_val) where tensor_val is (88,3).
                 # Some might be sigmoid/logits.
                 pass
            
        apply_count += 1
        
    print(f"Applied {apply_count} substitutions.")
    
    torch.save({"model_state": new_state}, Patched_CHECKPOINT)
    print(f"Saved patched checkpoint to {Patched_CHECKPOINT}")

if __name__ == "__main__":
    main()
