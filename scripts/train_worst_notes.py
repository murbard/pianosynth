import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from pianosynth.optimization_batch import PianoParamPerKey
from pianosynth.spectral import MultiResSTFTLoss, diff_piano_render
from pianosynth.physics import calculate_partials

PROCESSED_DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("src/pianosynth/checkpoints")
START_CPT = CHECKPOINT_DIR / "params_worst_fixed.pt"
FINAL_CPT = CHECKPOINT_DIR / "params_worst_lbfgs.pt"

def preprocess_sample(audio, midi, device="cpu"):
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
    
    # 1. Load Model
    model = PianoParamPerKey(device=device)
    loss_fn = MultiResSTFTLoss(device=device)
    
    if START_CPT.exists():
        print(f"Loading {START_CPT}")
        cpt = torch.load(START_CPT, map_location=device)
        model.load_state_dict(cpt["model_state"])
    else:
        print("Checkpoint not found!")
        return
        
    CLIP_LEN = 44100 * 2
    
    # 2. Find Worst 10 Notes
    print("Scanning dataset for worst 10 notes...")
    losses = []
    
    dyn_map = {'pp': 0, 'mf': 1, 'ff': 2}
    
    # We can perform a quick scan or trust the previous candidates?
    # Better to re-scan in case they changed.
    for m in tqdm(range(21, 109)):
        for d_str, d_idx in dyn_map.items():
            f = PROCESSED_DATA_DIR / f"{m}_{d_str}.pt"
            if not f.exists(): continue
            try:
                audio = torch.load(f).float().to(device)
                audio = preprocess_sample(audio, m)
                if audio is None or len(audio) < 1000: continue
                
                if len(audio) > CLIP_LEN: target = audio[:CLIP_LEN]
                else: target = F.pad(audio, (0, CLIP_LEN - len(audio)))
                target = target.unsqueeze(0)
                
                m_t = torch.tensor([m], device=device).float()
                d_t = torch.tensor([d_idx], device=device).long()
                
                with torch.no_grad():
                    overrides = model(m_t, d_t)
                    phys = calculate_partials(m_t, overrides, device=device)
                    y_pred = diff_piano_render(
                        phys["freqs"], phys["tau_s"], phys["tau_f"],
                        phys["amps"], phys["w_curve"], CLIP_LEN,
                        reverb_wet=phys.get("reverb_wet"), reverb_decay=phys.get("reverb_decay")
                    )
                    if y_pred.ndim == 1: y_pred = y_pred.unsqueeze(0)
                    l = loss_fn(y_pred, target).item()
                    
                losses.append({
                    'midi_t': m_t, 'dyn_t': d_t, 'target': target, 'loss': l,
                    'desc': f"{m} {d_str}"
                })
            except: pass
            
    losses.sort(key=lambda x: x['loss'], reverse=True)
    worst_10 = losses[:10]
    
    print("\nTop 10 Worst Candidates:")
    for item in worst_10:
        print(f"Note {item['desc']}: Loss {item['loss']:.4f}")
        
    # Prepare Full Batch Tensors
    b_midi = torch.cat([x['midi_t'] for x in worst_10])
    b_dyn = torch.cat([x['dyn_t'] for x in worst_10])
    b_targets = torch.cat([x['target'] for x in worst_10])
    
    # 3. Optim Setup (LBFGS)
    # LBFGS standard: LR=1.0 with Strong Wolfe line search
    optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, history_size=100, line_search_fn='strong_wolfe')
    
    print("\nStarting L-BFGS Training (LR=1.0, Strong Wolfe)...")
    
    # 1000 'Epochs' of LBFGS steps? Or 1000 iterations?
    # optim.LBFGS step() performs multiple iterations (max_iter).
    # If we want 1000 total evaluations, we can loop fewer times.
    # Let's loop 50 times with max_iter=20 inside step.
    
    outer_steps = 50 
    
    for step in range(outer_steps):
        
        def closure():
            optimizer.zero_grad()
            # Determinism is CRITICAL for L-BFGS
            torch.manual_seed(42) 
            overrides = model(b_midi, b_dyn)
            phys = calculate_partials(b_midi, overrides, device=device)
            y_pred = diff_piano_render(
                phys["freqs"], phys["tau_s"], phys["tau_f"],
                phys["amps"], phys["w_curve"], CLIP_LEN,
                reverb_wet=phys.get("reverb_wet"), reverb_decay=phys.get("reverb_decay")
            )
            loss = loss_fn(y_pred, b_targets)
            loss.backward()
            return loss
            
        loss = optimizer.step(closure)
        print(f"Step {step+1}/{outer_steps}: Loss {loss.item():.4f}")
        
    torch.save({"model_state": model.state_dict()}, FINAL_CPT)
    print(f"Saved {FINAL_CPT}")

if __name__ == "__main__":
    main()
