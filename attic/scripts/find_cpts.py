from pathlib import Path

def main():
    p = Path("src/pianosynth/checkpoints")
    print(f"Checking {p.absolute()}")
    if not p.exists():
        print("Dir not found!")
        # Try finding where checkpoints are
        for path in Path(".").rglob("checkpoints"):
            print(f"Found: {path}")
    else:
        print("Files:")
        for f in p.glob("params_best_loss_*.pt"):
            print(f.name)
        for f in p.glob("params_best_ever.pt"):
            print(f.name)

if __name__ == "__main__":
    main()
