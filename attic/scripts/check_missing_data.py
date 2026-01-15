from pathlib import Path
import os

PROCESSED_DATA_DIR = Path("data/processed")

def main():
    start_note = 21
    end_note = 108
    dyns = ['pp', 'mf', 'ff']
    
    missing = []
    
    print(f"Scanning {PROCESSED_DATA_DIR} for missing keys (Total {88*3} expected)...")
    
    for m in range(start_note, end_note + 1):
        for d in dyns:
            f_path = PROCESSED_DATA_DIR / f"{m}_{d}.pt"
            if not f_path.exists():
                missing.append((m, d))
                print(f"MISSING: {m}_{d}")

    print("-" * 30)
    print(f"Total Missing: {len(missing)}")
    if len(missing) > 0:
        print("These parameters are UNTRAINED (Random/Default Init).")
    else:
        print("All data present. No untrained parameters.")

if __name__ == "__main__":
    main()
