import torch
import numpy as np
from pathlib import Path

PROCESSED_DATA_DIR = Path("data/processed")

def load(m, d):
    p = PROCESSED_DATA_DIR / f"{m}_{d}.pt"
    if not p.exists(): return None
    return torch.load(p).float()

def main():
    print("Checking Data Similarity...")
    
    # Load suspects
    t21_ff = load(21, 'ff')
    t22_ff = load(22, 'ff')
    t23_ff = load(23, 'ff')
    t23_mf = load(23, 'mf')
    
    # Check Silence
    if t23_mf is not None:
        print(f"23_mf Max Amp: {t23_mf.abs().max().item()}")
        if t23_mf.abs().max() < 1e-4:
            print("23_mf is effectively SILENT.")
    else:
        print("23_mf Not Found")

    # Check Duplicates (22_ff vs 23_ff)
    if t22_ff is not None and t23_ff is not None:
        min_len = min(len(t22_ff), len(t23_ff))
        diff = (t22_ff[:min_len] - t23_ff[:min_len]).abs().mean()
        print(f"Diff 22_ff vs 23_ff: {diff.item():.6f}")
        if diff < 1e-4:
             print("ALERT: 22_ff and 23_ff are IDENTICAL.")
             
    # Check 21_ff vs 23_mf (User suspected similarity)
    if t21_ff is not None and t23_mf is not None:
         min_len = min(len(t21_ff), len(t23_mf))
         diff = (t21_ff[:min_len] - t23_mf[:min_len]).abs().mean()
         print(f"Diff 21_ff vs 23_mf: {diff.item():.6f}")
         
    # Check Pitch consistency again with better method?
    # nah, similarity is enough to prove corruption.

if __name__ == "__main__":
    main()
