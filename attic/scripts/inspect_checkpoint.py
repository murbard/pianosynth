import torch
import sys

def main():
    f = "src/pianosynth/checkpoints/params_best_ever.pt"
    try:
        cpt = torch.load(f, map_location='cpu')
        print(f"File: {f}")
        if 'loss' in cpt:
            print(f"Loss: {cpt['loss']}")
        else:
            print("Loss: NOT FOUND")
        print(f"Keys: {cpt.keys()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
