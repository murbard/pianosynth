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
# Start from the patched state
START_CPT = CHECKPOINT_DIR / "params_categorical_patched.pt"
SMART_CPT = CHECKPOINT_DIR / "params_smart_smooth.pt"

def preprocess_sample(audio, midi, device="cpu", sr=44100):
    # Trim Silence
    threshold = 1e-3
    mask = torch.abs(audio) > threshold
    if mask.any():
         idx = torch.where(mask)[0][0]
         idx = max(0, idx - 100)
         audio = audio[idx:]
    else:
         return None

    # Tune (Simplified) - Assume mostly tuned or handled by data prep
    # Just ensure length
    return audio

def check_and_replace_neighbors(model, dataset_map, device, loss_fn, clip_len):
    """
    Iterate all keys/dyns.
    Calculate L_self.
    Calculate L_avg (params = avg(prev, next)).
    If L_avg < L_self, replace params.
    """
    replacements = 0
    total_checks = 0
    state = model.params
    
    # We need to perform this check carefully. 
    # We can't modify 'state' while iterating if we rely on neighbors?
    # Actually, "Gauss-Seidel" (replace immediately) might propagate smoothness faster.
    # But "Jacobi" (replace after checking all) is more stable?
    # Let's do immediate replacement for faster convergence towards smoothness.
    
    midis = sorted(list(dataset_map.keys()))
    
    dyns_map = {'pp': 0, 'mf': 1, 'ff': 2}
    
    # Iterate
    # Skip boundaries for averaging (21 and 108 don't have 2 neighbors)
    # Range 22 to 107
    
    candidates = [m for m in midis if m > 21 and m < 108]
    
    for m in candidates:
        for d_str, d_idx in dyns_map.items():
            if d_str not in dataset_map[m]: continue
            
            # Get Ground Truth
            audio_gt = dataset_map[m][d_str]
            if len(audio_gt) > clip_len: audio_gt = audio_gt[:clip_len]
            else: audio_gt = F.pad(audio_gt, (0, clip_len - len(audio_gt)))
            
            audio_gt = audio_gt.unsqueeze(0)
            
            m_t = torch.tensor([m], device=device).float()
            d_t = torch.tensor([d_idx], device=device).long()
            
            # --- 1. Calculate L_self ---
            with torch.no_grad():
                ov_self = model(m_t, d_t)
                phys_self = calculate_partials(m_t, ov_self, device=device)
                y_self = diff_piano_render(
                    phys_self["freqs"], phys_self["tau_s"], phys_self["tau_f"],
                    phys_self["amps"], phys_self["w_curve"], clip_len,
                    reverb_wet=phys_self.get("reverb_wet"), reverb_decay=phys_self.get("reverb_decay")
                )
                if y_self.ndim == 1: y_self = y_self.unsqueeze(0)
                l_self = loss_fn(y_self, audio_gt).item()
                
            # --- 2. Calculate L_avg ---
            # Construct Avg Params
            # We must manually average the underlying parameters in the model.params Dict
            # Param indices: m-21. Neighbor indices: (m-1)-21, (m+1)-21
            idx_curr = m - 21
            idx_prev = (m - 1) - 21
            idx_next = (m + 1) - 21
            
            # Temporary override dict? No, we need to pass parameters to calculate_partials.
            # But calculate_partials calls overrides...
            # We can manually construct the 'overrides' dict with averaged values.
            
            # But the model() forward pass applies softplus/sigmoid/etc.
            # We should average the UNCONSTRAINED parameters (in self.params), then pass through forward logic?
            # User said "test if a note's parameter would be better off replaced by the average".
            # Averaging the raw logits/unconstrained params is usually safer for optimization.
            
            # Let's construct a "virtual" model output at 'm' using averaged raw params.
            
            ov_avg = {}
            with torch.no_grad():
                # We need to manually invoke the logic inside model.forward but with averaged params
                # To avoid duplicating forward logic, we can temporarily WRITE average to the model at index m,
                # compute loss, then revert if it wasn't better.
                
                # Backup current params
                backup = {}
                for name, param in model.params.items():
                    # param shape [88, 3] or [88]
                    if param.ndim == 2:
                        backup[name] = param[idx_curr, d_idx].clone()
                        # Set to Average
                        avg_val = 0.5 * (param[idx_prev, d_idx] + param[idx_next, d_idx])
                        param[idx_curr, d_idx] = avg_val
                    # Ignore scalar/1D params for now (or treat as shared)
                
                # Compute Loss with Averaged Params (which are now in the model slot m)
                ov_test = model(m_t, d_t)
                phys_test = calculate_partials(m_t, ov_test, device=device)
                y_test = diff_piano_render(
                    phys_test["freqs"], phys_test["tau_s"], phys_test["tau_f"],
                    phys_test["amps"], phys_test["w_curve"], clip_len,
                    phys_test.get("reverb_wet"), phys_test.get("reverb_decay")
                )
                if y_test.ndim == 1: y_test = y_test.unsqueeze(0)
                l_avg = loss_fn(y_test, audio_gt).item()
                
                # Decision
                if l_avg < l_self:
                    # Keep the change! (Already written to model)
                    replacements += 1
                else:
                    # Revert
                    for name, param in model.params.items():
                        if param.ndim == 2:
                             param[idx_curr, d_idx] = backup[name]
                             
            total_checks += 1
            
    return replacements, total_checks


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Data
    # Pre-load all samples into RAM for speed
    dataset_map = {} # { midi: { 'pp': tensor, ... } }
    
    print("Loading Dataset...")
    dyn_strs = ['pp', 'mf', 'ff']
    
    for m in range(21, 109):
        dataset_map[m] = {}
        for d in dyn_strs:
            f = PROCESSED_DATA_DIR / f"{m}_{d}.pt"
            # Fallback
            if not f.exists():
                for alt in ['mf', 'ff', 'pp']:
                    if (PROCESSED_DATA_DIR / f"{m}_{alt}.pt").exists():
                        f = PROCESSED_DATA_DIR / f"{m}_{alt}.pt"
                        break
            
            if f.exists():
                try:
                    audio = torch.load(f).float().to(device)
                    audio = preprocess_sample(audio, m)
                    if audio is not None and len(audio) > 1000:
                        dataset_map[m][d] = audio
                except: pass
                
    # 2. Init Model
    model = PianoParamPerKey(device=device)
    loss_fn = MultiResSTFTLoss(device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    if START_CPT.exists():
        print(f"Loading {START_CPT}")
        cpt = torch.load(START_CPT, map_location=device)
        model.load_state_dict(cpt["model_state"])
    else:
        print("Starting from scratch (WARNING: Expected patched checkpoint)")
        
    EPOCHS = 100
    CLIP_LEN = 44100 * 2
    
    print(f"Starting Smart Smoothing Training ({EPOCHS} Epochs)...")
    
    for epoch in range(1, EPOCHS + 1):
        # A. Train Step (Standard Gradient Descent)
        model.train()
        train_loss = 0
        steps = 0
        
        # Random batching for training
        # Flatten dataset for loader
        flat_data = []
        for m, d_dict in dataset_map.items():
            for d, aud in d_dict.items():
                flat_data.append((m, d, aud))
        
        # Shuffle
        # Simple batch loop
        BATCH_SIZE = 4
        import random
        random.shuffle(flat_data)
        
        dyn_map = {'pp': 0, 'mf': 1, 'ff': 2}
        
        for i in range(0, len(flat_data), BATCH_SIZE):
            batch = flat_data[i:i+BATCH_SIZE]
            
            # Prepare batch
            midis = torch.tensor([b[0] for b in batch], device=device).float()
            dyns = torch.tensor([dyn_map[b[1]] for b in batch], device=device).long()
            
            targets = []
            for b in batch:
                aud = b[2]
                if len(aud) > CLIP_LEN: targets.append(aud[:CLIP_LEN])
                else: targets.append(F.pad(aud, (0, CLIP_LEN - len(aud))))
            targets = torch.stack(targets)
            
            optimizer.zero_grad()
            overrides = model(midis, dyns)
            phys = calculate_partials(midis, overrides, device=device)
            y_pred = diff_piano_render(
                phys["freqs"], phys["tau_s"], phys["tau_f"],
                phys["amps"], phys["w_curve"], CLIP_LEN,
                reverb_wet=phys.get("reverb_wet"), reverb_decay=phys.get("reverb_decay")
            )
            
            loss = loss_fn(y_pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            steps += 1
            
        avg_loss = train_loss / steps
        
        # B. Smart Smoothing Step
        rep, checks = check_and_replace_neighbors(model, dataset_map, device, loss_fn, CLIP_LEN)
        
        print(f"Epoch {epoch}: L={avg_loss:.4f} | Replaced {rep}/{checks} params")
        
        if epoch % 10 == 0:
            torch.save({"model_state": model.state_dict()}, SMART_CPT)

    torch.save({"model_state": model.state_dict()}, SMART_CPT)
    print("Smart Smoothing Complete.")

if __name__ == "__main__":
    main()
