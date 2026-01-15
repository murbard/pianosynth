import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pianosynth.optimization_batch import PianoParamPerKey
from pianosynth.spectral import MultiResSTFTLoss, diff_piano_render
from pianosynth.physics import calculate_partials

PROCESSED_DATA_DIR = Path("data/processed")
CHECKPOINT = Path("src/pianosynth/checkpoints/params_outliers_fixed.pt")

def preprocess_sample_simple(audio, midi, sr=44100):
   # Simplified preprocessing matches training
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
    print(f"Using device: {device}", flush=True)
    
    # Load Model
    model = PianoParamPerKey(device=device)
    if not CHECKPOINT.exists():
        print("Checkpoint not found.", flush=True)
        return
    cpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(cpt["model_state"])
    
    loss_fn = MultiResSTFTLoss(device=device)
    
    # Iterate all notes/dynamics
    midis = list(range(21, 109))
    dyns = ['pp', 'mf', 'ff']
    dyn_map = {'pp': 0, 'mf': 1, 'ff': 2}
    
    # Store losses: {(midi, dyn_str): loss}
    losses = {}
    
    print("Calculating Loss per Note...", flush=True)
    
    for m in midis:
        for d_name in dyns:
            f_path = PROCESSED_DATA_DIR / f"{m}_{d_name}.pt"
            if not f_path.exists():
                continue
                
            try:
                audio_gt = torch.load(f_path).float().to(device)
            except:
                continue
                
            audio_gt = preprocess_sample_simple(audio_gt, m)
            if audio_gt is None or len(audio_gt) < 1000:
                continue
                
            # Render Length
            CLIP_LEN = 44100 * 2
            if len(audio_gt) > CLIP_LEN:
                audio_gt = audio_gt[:CLIP_LEN]
            else:
                pad = CLIP_LEN - len(audio_gt)
                audio_gt = torch.cat([audio_gt, torch.zeros(pad, device=device)])
                
            # Run Model
            m_t = torch.tensor([m], device=device).float()
            d_t = torch.tensor([dyn_map[d_name]], device=device).long()
            
            with torch.no_grad():
                overrides = model(m_t, d_t)
                phys_out = calculate_partials(m_t, overrides, device=device)
                y_pred = diff_piano_render(
                    phys_out["freqs"], phys_out["tau_s"], phys_out["tau_f"],
                    phys_out["amps"], phys_out["w_curve"], CLIP_LEN,
                    reverb_wet=phys_out.get("reverb_wet"), reverb_decay=phys_out.get("reverb_decay")
                )
                if y_pred.ndim == 1: y_pred = y_pred.unsqueeze(0)
                if audio_gt.ndim == 1: audio_gt = audio_gt.unsqueeze(0)
                
                loss = loss_fn(y_pred, audio_gt)
                losses[(m, d_name)] = loss.item()

    # Calculate Ratios
    # Ratio = L(m) / Mean(L(m-1), L(m+1))
    # We need to handle boundary conditions and missing neighbors
    
    outliers = []
    
    for key, loss in losses.items():
        m, d = key
        
        l_prev = losses.get((m-1, d))
        l_next = losses.get((m+1, d))
        
        neighbors = []
        if l_prev is not None: neighbors.append(l_prev)
        if l_next is not None: neighbors.append(l_next)
        
        if not neighbors:
            continue
            
        avg_neighbor = sum(neighbors) / len(neighbors)
        if avg_neighbor < 1e-6: continue # Avoid div/0
        
        ratio = loss / avg_neighbor
        outliers.append({
            'midi': m,
            'dyn': d,
            'loss': loss,
            'avg_neighbor': avg_neighbor,
            'ratio': ratio
        })
        
    # Sort by ratio descending
    outliers.sort(key=lambda x: x['ratio'], reverse=True)
    
    print("\nTop 10 Outliers (Loss vs Neighbors):", flush=True)
    print(f"{'Note':<10} {'Dyn':<5} {'Loss':<10} {'Neighbor':<10} {'Ratio':<10}", flush=True)
    print("-" * 50, flush=True)
    for i in range(min(10, len(outliers))):
        o = outliers[i]
        try:
             # Just formatting safe note name
             note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
             octave = (o['midi'] // 12) - 1
             note = f"{note_names[o['midi'] % 12]}{octave}"
        except: note = str(o['midi'])
        
        print(f"{note:<10} {o['dyn']:<5} {o['loss']:.4f}     {o['avg_neighbor']:.4f}     {o['ratio']:.4f}", flush=True)
if __name__ == "__main__": main()
