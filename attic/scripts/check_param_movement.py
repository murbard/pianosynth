import torch
import numpy as np
import pandas as pd
from pathlib import Path
from pianosynth.optimization_batch import PianoParamPerKey

CHECKPOINT = Path("src/pianosynth/checkpoints/params_categorical.pt")

def main():
    device = "cpu"
    
    # 1. Fresh Model (Initialization)
    model_init = PianoParamPerKey(device=device)
    state_init = model_init.state_dict()
    
    # 2. Trained Model
    if not CHECKPOINT.exists():
        print("Checkpoint not found.")
        return
    cpt = torch.load(CHECKPOINT, map_location=device)
    state_trained = cpt["model_state"]
    
    results = []
    
    print("Parameter Movement Analysis (Trained - Init):")
    print(f"{'Parameter':<30} | {'Mean Abs Diff':<15} | {'Max Abs Diff':<15} | {'Rel Diff (%)':<15}")
    print("-" * 85)
    
    for name, p_init in state_init.items():
        if name not in state_trained:
            print(f"Skipping {name} (not in checkpoint)")
            continue
            
        p_trained = state_trained[name]
        
        # Calculate diffs
        diff = p_trained - p_init
        abs_diff = torch.abs(diff)
        
        mean_diff = abs_diff.mean().item()
        max_diff = abs_diff.max().item()
        
        # Relative diff (avoid div by zero)
        denom = torch.abs(p_init).mean().item() + 1e-9
        rel_diff = (mean_diff / denom) * 100.0
        
        results.append({
            "Parameter": name,
            "MeanAbsDiff": mean_diff,
            "MaxAbsDiff": max_diff,
            "RelDiffPct": rel_diff
        })
        
        print(f"{name:<30} | {mean_diff:<15.6f} | {max_diff:<15.6f} | {rel_diff:<15.2f}")

    # Save to CSV
    df = pd.DataFrame(results)
    df = df.sort_values("RelDiffPct", ascending=True)
    df.to_csv("results_categorical/param_movement.csv", index=False)
    print("\nSaved to results_categorical/param_movement.csv")
    
    # Heuristic for "Untuned"
    untuned = df[df["RelDiffPct"] < 0.1] # Less than 0.1% change
    if not untuned.empty:
        print("\nPotentially Untuned Parameters (< 0.1% movement):")
        print(untuned[["Parameter", "RelDiffPct"]].to_string(index=False))

if __name__ == "__main__":
    main()
