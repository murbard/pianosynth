import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from pianosynth.optimization_batch import PianoParamPerKey

CHECKPOINT = Path("src/pianosynth/checkpoints/params_best_ever.pt")
OUT_DIR = Path("results_categorical/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    device = "cpu" # sufficient for plotting
    
    model = PianoParamPerKey(device=device)
    if CHECKPOINT.exists():
        print(f"Loading {CHECKPOINT}...")
        cpt = torch.load(CHECKPOINT, map_location=device)
        model.load_state_dict(cpt["model_state"])
    else:
        print("Checkpoint not found! Plotting initialized values.")
        
    # Generate data
    midis = torch.arange(21, 109, device=device).float()
    
    # Run for each dynamic
    dyn_data = {}
    for dyn_name, dyn_idx in [('pp', 0), ('mf', 1), ('ff', 2)]:
        dyn_t = torch.full_like(midis, dyn_idx).long()
        
        with torch.no_grad():
            # Get physical values (softplus etc applied)
            overrides = model(midis, dyn_t)
            
        dyn_data[dyn_name] = overrides
        
    # Keys to plot
    # We'll plot everything in the returned dictionary
    keys = list(dyn_data['mf'].keys())
    
    print(f"Plotting {len(keys)} parameters...")
    
    for k in keys:
        plt.figure(figsize=(10, 6))
        
        # Extract arrays
        y_pp = dyn_data['pp'][k].numpy()
        y_mf = dyn_data['mf'][k].numpy()
        y_ff = dyn_data['ff'][k].numpy()
        x = midis.numpy()
        
        # Handle scalar expansion if necessary (though model should return [88])
        if y_pp.ndim == 0: y_pp = np.full_like(x, y_pp)
        if y_mf.ndim == 0: y_mf = np.full_like(x, y_mf)
        if y_ff.ndim == 0: y_ff = np.full_like(x, y_ff)
            
        plt.plot(x, y_pp, label='pp', color='blue', linewidth=1.5)
        plt.plot(x, y_mf, label='mf', color='green', linewidth=1.5)
        plt.plot(x, y_ff, label='ff', color='red', linewidth=1.5)
        
        plt.title(f"Parameter: {k}")
        plt.xlabel("Midi Note")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Sanitize filename
        safe_k = k.replace('.', '_')
        plt.savefig(OUT_DIR / f"{safe_k}.png")
        plt.close()
        
    print(f"Saved plots to {OUT_DIR}")

if __name__ == "__main__":
    main()
