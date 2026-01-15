import json
import matplotlib.pyplot as plt
from pathlib import Path

HISTORY_PATH = Path("loss_history.json")
OUTPUT_PATH = Path("loss_curves.png")

def main():
    if not HISTORY_PATH.exists():
        print("History not found.")
        return
        
    with open(HISTORY_PATH, "r") as f:
        history = json.load(f)
        
    epochs = [h["epoch"] for h in history]
    loss_total = [h["loss_total"] for h in history]
    loss_freq = [h["loss_freq"] for h in history]
    loss_decay = [h["loss_decay"] for h in history]
    loss_amp = [h["loss_amp"] for h in history]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, loss_total, label="Total Loss", linewidth=2, color="k")
    plt.plot(epochs, loss_freq, label="Freq Loss (Scaled)", linestyle="--", alpha=0.7)
    plt.plot(epochs, loss_decay, label="Decay Loss", linestyle="--", alpha=0.7)
    plt.plot(epochs, loss_amp, label="Amp Loss", linestyle="--", alpha=0.7)
    
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.title("Optimization Loss Curves")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
