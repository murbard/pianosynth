import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from pianosynth.optimization_batch import PianoParamPerKey
from pianosynth.spectral import MultiResSTFTLoss, diff_piano_render
from pianosynth.physics import calculate_partials

PROCESSED_DATA_DIR = Path("data/processed")
CHECKPOINT = Path("src/pianosynth/checkpoints/params_categorical.pt")
OUT_DIR = Path("results_categorical")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def preprocess_sample_simple(audio, midi, sr=44100):
    threshold = 1e-3
    mask = torch.abs(audio) > threshold
    if mask.any():
         idx = torch.where(mask)[0][0]
         idx = max(0, idx - 100)
         audio = audio[idx:]
    else:
         return None
    return audio

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Model
    model = PianoParamPerKey(device=device)
    if not CHECKPOINT.exists():
        print("Checkpoint not found.")
        return
    cpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(cpt["model_state"])
    
    loss_fn = MultiResSTFTLoss(device=device)
    
    # Targets
    midis = list(range(21, 109))
    dyns_map = {'pp': 0, 'mf': 1, 'ff': 2}
    
    # Storage
    # { 'pp': { midi: { 'self': L, 'prev': L, 'next': L } } }
    results = {d: {} for d in dyns_map.keys()}
    
    print("Running Neighbor Substitution Test...")
    
    for m in tqdm(midis):
        # 0. Load Ground Truth
        # We need a valid sample for this pitch to compute ANY loss
        # Ideally we check all dynamics
        pass
    
    # New loop structure to process one clip at a time
    for d_name, d_idx in dyns_map.items():
        print(f"Processing {d_name}...")
        for m in tqdm(midis):
            # Load Audio
            f_path = PROCESSED_DATA_DIR / f"{m}_{d_name}.pt"
            if not f_path.exists(): continue
            try:
                audio_gt = torch.load(f_path).float().to(device)
            except: continue
            
            audio_gt = preprocess_sample_simple(audio_gt, m)
            if audio_gt is None or len(audio_gt) < 1000: continue
            
            CLIP_LEN = 44100 * 2
            if len(audio_gt) > CLIP_LEN:
                audio_gt = audio_gt[:CLIP_LEN]
            else:
                pad = CLIP_LEN - len(audio_gt)
                audio_gt = torch.cat([audio_gt, torch.zeros(pad, device=device)])
            
            # Prepare Batch Tensors
            audio_gt = audio_gt.unsqueeze(0) # [1, T]
            m_t = torch.tensor([m], device=device).float()
            d_t = torch.tensor([d_idx], device=device).long()
            
            # Function to calculate loss with specific overrides
            def get_loss(m_target_params):
                # m_target_params: MIDI note index to pull parameters FROM
                # But we render at pitch 'm' (current note)
                
                # Boundary check
                if m_target_params < 21 or m_target_params > 108:
                    return float('inf')
                
                m_src_t = torch.tensor([m_target_params], device=device).float()
                
                with torch.no_grad():
                    # Get PARAMS from Source
                    overrides = model(m_src_t, d_t)
                    
                    # Render at TARGET Pitch (m)
                    # Note: calculate_partials uses 'midi' argument for F0 calculation
                    # and uses 'overrides' for the parameters. This is exactly what we want.
                    phys_out = calculate_partials(m_t, overrides, device=device)
                    
                    y_pred = diff_piano_render(
                        freqs=phys_out["freqs"],
                        tau_s=phys_out["tau_s"],
                        tau_f=phys_out["tau_f"],
                        amps=phys_out["amps"],
                        w_curve=phys_out["w_curve"],
                        dur_samples=CLIP_LEN,
                        reverb_wet=phys_out.get("reverb_wet"),
                        reverb_decay=phys_out.get("reverb_decay")
                    )
                    
                    if y_pred.ndim == 1: y_pred = y_pred.unsqueeze(0)
                    
                    l = loss_fn(y_pred, audio_gt)
                    return l.item()

            l_self = get_loss(m)
            l_prev = get_loss(m - 1)
            l_next = get_loss(m + 1)
            
            results[d_name][m] = {
                "self": l_self,
                "prev": l_prev,
                "next": l_next
            }
            
    # --- Analysis & Plotting ---
    plt.figure(figsize=(14, 8))
    
    colors = {'pp': 'blue', 'mf': 'green', 'ff': 'red'}
    
    for i, d_name in enumerate(dyns_map.keys()):
        data = results[d_name]
        midis_sorted = sorted(data.keys())
        if not midis_sorted: continue
        
        gaps = []
        valid_midis = []
        
        for m in midis_sorted:
            entry = data[m]
            l_s = entry["self"]
            l_p = entry["prev"]
            l_n = entry["next"]
            
            best_neighbor = min(l_p, l_n)
            
            # Gap = Self - Neighbor
            # Positive = Neighbor is better (Bad Minima)
            # Negative = Self is better (Good)
            gap = l_s - best_neighbor
            
            # Filter huge outliers (e.g. inf)
            if abs(gap) > 10.0: continue 
            
            gaps.append(gap)
            valid_midis.append(m)
            
        plt.subplot(3, 1, i+1)
        plt.bar(valid_midis, gaps, color=colors[d_name], alpha=0.7)
        plt.axhline(0, color='k', linewidth=1)
        plt.title(f"Optimization Gap ({d_name}): Positive = Neighbor params are better")
        plt.ylabel("Loss Difference")
        plt.ylim(-0.2, 0.5) # Zoom in on improvements
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(OUT_DIR / "neighbor_substitution_gap.png")
    print(f"Saved plot to {OUT_DIR / 'neighbor_substitution_gap.png'}")
    
    # Save a summary CSV of improving substitutions
    import pandas as pd
    rows = []
    for d_name in dyns_map.keys():
        for m, entry in results[d_name].items():
            l_s = entry["self"]
            l_p = entry["prev"]
            l_n = entry["next"]
            
            best_n = l_p
            best_src = m - 1
            if l_n < l_p:
                best_n = l_n
                best_src = m + 1
            
            if best_n < l_s:
                rows.append({
                    "Dynamic": d_name,
                    "Note": m,
                    "SelfLoss": l_s,
                    "BestNeighborLoss": best_n,
                    "SourceNote": best_src,
                    "Improvement": l_s - best_n
                })
                
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Improvement", ascending=False)
        df.to_csv(OUT_DIR / "neighbor_improvements.csv", index=False)
        print("Analysis Summary (Top Improvements):")
        print(df.head(10).to_string(index=False))
        print(f"Full CSV saved to {OUT_DIR / 'neighbor_improvements.csv'}")

if __name__ == "__main__":
    main()
