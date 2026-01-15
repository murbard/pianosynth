import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import random
import numpy as np
import torch.nn.functional as F

from pianosynth.optimization_batch import PianoParamPerKey
from pianosynth.spectral import MultiResSTFTLoss, diff_piano_render
from pianosynth.physics import calculate_partials

PROCESSED_DATA_DIR = Path("data/clean_et")
CHECKPOINT_DIR = Path("src/pianosynth/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def preprocess_sample(audio, midi, device="cpu", sr=44100):
    # 1. Trim Silence (Still useful if clean_et has silence)
    threshold = 1e-3
    mask = torch.abs(audio) > threshold
    if mask.any():
         idx = torch.where(mask)[0][0]
         # Keep a bit of pre-onset
         idx = max(0, idx - 100)
         audio = audio[idx:]
    else:
         return None

    # 2. Tune to ET - REMOVED (Data is now clean)
    # audio is already 12-TET tuned from data/clean_et
        
    return audio

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Train on full 260-sample dataset")
    args = parser.parse_args()
    
    if args.full:
        print("Training on FULL dataset.")
        target_midis = list(range(21, 109))
        sparse_mode = False
    else:
        print("Training on SPARSE dataset (27 notes).")
        target_midis = np.linspace(21, 108, 27).round().astype(int)
        sparse_mode = True
    
    dataset = []
    print("Loading dataset...")
    
    dyn_map = {'pp': 0, 'mf': 1, 'ff': 2}
    
    loop_iter = list(enumerate(target_midis))
    target_dyns = ['pp', 'mf', 'ff']

    for i, m in loop_iter:
        if sparse_mode:
            req_dyns = [target_dyns[i % 3]]
        else:
            req_dyns = ['pp', 'mf', 'ff']
            
        for d_req in req_dyns:
            f = PROCESSED_DATA_DIR / f"{m}_{d_req}.pt"
            
            # Fallback (Sparse only)
            if not f.exists() and sparse_mode:
                 for alt in ['mf', 'ff', 'pp']:
                     if (PROCESSED_DATA_DIR / f"{m}_{alt}.pt").exists():
                         f = PROCESSED_DATA_DIR / f"{m}_{alt}.pt"
                         break
            
            if not f.exists(): continue
        
            try:
                audio = torch.load(f).float().to(device)
            except:
                continue
                
            audio = preprocess_sample(audio, m, device=device)
            if audio is None or len(audio) < 1000: continue
            
            dataset.append({
                "midi": int(m),
                "dyn_idx": dyn_map.get(d_req, 1), # Default mf
                "audio": audio
            })

    print(f"Loaded {len(dataset)} samples.")
    
    # Model
    model = PianoParamPerKey(device=device)
    loss_fn = MultiResSTFTLoss(device=device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Checkpoint logic: 
    # Legacy checkpoint is incompatible with (88, 3).
    # We start fresh or implement fancy migration.
    # Given the major refactor, FRESH start is safer, relying on the new init logic.
    # Resuming from patched checkpoint
    MASTER_CPT = CHECKPOINT_DIR / "params_categorical_patched.pt"
    
    if MASTER_CPT.exists():
         print(f"Resuming from {MASTER_CPT}")
         cpt = torch.load(MASTER_CPT, map_location=device)
         model.load_state_dict(cpt["model_state"])

    BATCH_SIZE = 4
    EPOCHS = 10
    CLIP_LEN = 44100 * 2
    
    targets_padded = []
    midis = []
    dyn_indices = []
    
    for d in dataset:
        audio = d['audio']
        if len(audio) > CLIP_LEN:
            t = audio[:CLIP_LEN]
        else:
            t = torch.cat([audio, torch.zeros(CLIP_LEN - len(audio), device=device)])
        targets_padded.append(t)
        midis.append(d['midi'])
        dyn_indices.append(d['dyn_idx'])
        
    targets_padded = torch.stack(targets_padded)
    midis_tensor = torch.tensor(midis, device=device).float()
    dyns_tensor = torch.tensor(dyn_indices, device=device).long()
    
    print("Starting Training...")
    n_samples = len(dataset)
    pbar = tqdm(range(EPOCHS))
    
    for epoch in pbar:
        perm = torch.randperm(n_samples, device=device)
        epoch_loss = 0
        steps = 0
        
        for i in range(0, n_samples, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            
            batch_midi = midis_tensor[idx]
            batch_dyns = dyns_tensor[idx]
            batch_targets = targets_padded[idx]
            
            optimizer.zero_grad()
            
            # Forward taking dyn indices
            overrides = model(batch_midi, batch_dyns)
            
            # Physics (No Velocity)
            phys_out = calculate_partials(
                midi=batch_midi,
                overrides=overrides,
                n_partials=64,
                device=device
            )
            
            y_pred = diff_piano_render(
                freqs=phys_out["freqs"],
                tau_s=phys_out["tau_s"],
                tau_f=phys_out["tau_f"],
                amps=phys_out["amps"],
                w_curve=phys_out["w_curve"],
                dur_samples=CLIP_LEN,
                reverb_wet=phys_out["reverb_wet"],
                reverb_decay=phys_out["reverb_decay"]
            )
            
            loss = loss_fn(y_pred, batch_targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            steps += 1
            
        avg_L = epoch_loss / steps
        pbar.set_description(f"L: {avg_L:.4f}")
        
        if epoch % 100 == 0:
            torch.save({ "model_state": model.state_dict() }, MASTER_CPT)
            
    torch.save({ "model_state": model.state_dict() }, MASTER_CPT)
    print("Training Complete.")

if __name__ == "__main__":
    main()
