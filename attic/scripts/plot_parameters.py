import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict
from tqdm import tqdm

from pianosynth.optimization_batch import PianoParamPerKey

CHECKPOINT_DIR = Path("src/pianosynth/checkpoints")

def main():
    device = "cpu"
    
    checkpoint_path = CHECKPOINT_DIR / "params_all_keys.pt"
    if not checkpoint_path.exists():
        print("Master params_all_keys.pt not found.")
        return
        
    print(f"Loading {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PianoParamPerKey(device=device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    
    # Get all physical parameters for 88 keys
    inputs = torch.arange(21, 109, device=device).float()
    
    with torch.no_grad():
        overrides = model(inputs)
    
    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)
    
    print(f"Generating plots in {plot_dir}...")
    
    for k, v in overrides.items():
        if not torch.is_tensor(v): continue
        if v.ndim > 1:
             # e.g. [88, 1]? Flatten
             v = v.view(-1)
        if v.shape[0] != 88:
            print(f"Skipping {k}: shape {v.shape}")
            continue
            
        vals = v.numpy()
        midis = inputs.numpy()
        
        plt.figure(figsize=(10, 6))
        plt.plot(midis, vals, 'o-', alpha=0.8, markersize=4)
        plt.title(f"Parameter: {k}")
        plt.xlabel("Midi Note")
        plt.ylabel("Value")
        plt.grid(True, which='both', alpha=0.3)
        
        # Highlight sparse trained notes? 
        # 21, 24, ...
        # sparse_midis = np.linspace(21, 108, 27).round().astype(int)
        # plt.plot(sparse_midis, vals[sparse_midis-21], 'rx')
        
        safe_name = k.replace(".", "_")
        plt.savefig(plot_dir / f"param_{safe_name}.png")
        plt.close()
        
    print("Done.")

if __name__ == "__main__":
    main()
