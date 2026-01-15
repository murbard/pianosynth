import torch
import numpy as np

def main():
    try:
        data = torch.load("raw_file_map.pt")
    except:
        print("Map not found.")
        return
        
    # Filter valid
    valid = [d for d in data if d["freq"] > 20]
    valid.sort(key=lambda x: x["midi"])
    
    print(f"Total Valid Files: {len(valid)}")
    if len(valid) > 0:
        print("--- Lowest 10 ---")
        for i in range(min(10, len(valid))):
            d = valid[i]
            print(f"{d['midi']:.2f} ({d['freq']:.1f} Hz) - {d['label']} ({d['path']})")
            
        print("\n--- Highest 5 ---")
        for i in range(1, 6):
            d = valid[-i]
            print(f"{d['midi']:.2f} ({d['freq']:.1f} Hz) - {d['label']} ({d['path']})")
            
    # Check specific problematic files
    print("\n--- Diagnostic: B0 ---")
    b0 = [d for d in data if "B0" in str(d["label"])]
    for d in b0:
         print(f"{d['label']} -> {d['freq']:.1f} Hz (Midi {d['midi']:.2f}) Path: {d['path']}")

if __name__ == "__main__":
    main()
