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
START_CPT = CHECKPOINT_DIR / "params_lbfgs_global_deep.pt" 
FINAL_CPT = CHECKPOINT_DIR / "params_lbfgs_global_robust.pt"

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
        print(f"Checkpoint {START_CPT} not found! Fallback to patched?")
        return
        
    CLIP_LEN = 44100 * 2
    
    # 2. Gather All Tasks
    print("Gathering all valid samples...")
    tasks = []
    
    dyn_map = {'pp': 0, 'mf': 1, 'ff': 2}
    
    for m in range(21, 109):
        for d_str, d_idx in dyn_map.items():
            f = PROCESSED_DATA_DIR / f"{m}_{d_str}.pt"
            f_prev = PROCESSED_DATA_DIR / f"{m-1}_{d_str}.pt"
            f_next = PROCESSED_DATA_DIR / f"{m+1}_{d_str}.pt"
            
            if f.exists():
                tasks.append({
                    'midi': m,
                    'dyn_str': d_str,
                    'dyn_idx': d_idx,
                    'path': f,
                    'prev_path': f_prev if f_prev.exists() else None,
                    'next_path': f_next if f_next.exists() else None
                })
                
    print(f"Found {len(tasks)} samples.")
    
    # 3. Batch Loop
    # RTX 3060 (12GB) Limit: Batch 32 caused OOM. Reducing to 10.
    # 3. Batch Loop
    # Robust Training: Batch 8 to safely handle 3x target buffers
    BATCH_SIZE = 8
    
    # Sort for deterministic processing order (Sequential)
    tasks.sort(key=lambda x: (x['midi'], x['dyn_idx']))
    
    total_batches = (len(tasks) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"Scanning {total_batches} batches (Size {BATCH_SIZE})...")
    
    for b_idx in range(total_batches):
        torch.cuda.empty_cache() # Keep memory clean
        
        start = b_idx * BATCH_SIZE
        end = min(len(tasks), start + BATCH_SIZE)
        batch_tasks = tasks[start:end]
        
        # Load Batch Data
        b_midi_list = []
        b_dyn_list = []
        
        # Targets: List of tuples (main, prev, next)
        # We will stack them later or handle inside closure
        b_target_main = []
        b_target_prev = []
        b_target_next = []
        
        valid_count = 0
        
        for item in batch_tasks:
            try:
                # Load Main
                func_load = lambda p: torch.load(p).float().to(device) if p and p.exists() else None
                
                audio_main = func_load(item['path'])
                audio_prev = func_load(item.get('prev_path'))
                audio_next = func_load(item.get('next_path'))
                
                if audio_main is None: continue

                # Preprocess Main
                audio_main = preprocess_sample(audio_main, item['midi'])
                if audio_main is None or len(audio_main) < 1000: continue
                
                # Preprocess & Resample Neighbors
                # Prev (M-1) -> Want M (Higher Pitch) -> Play Faster -> Shorter Duration
                # Ratio = 2^(1/12) approx 1.05946
                # New Len = Old Len / Ratio
                if audio_prev is not None:
                     audio_prev = preprocess_sample(audio_prev, item['midi'] - 1)
                     if audio_prev is not None:
                         # Resample
                         # Input [1, 1, T]
                         ratio = 2**(1/12)
                         new_len = int(len(audio_prev) / ratio)
                         audio_prev = audio_prev.view(1, 1, -1)
                         audio_prev = F.interpolate(audio_prev, size=new_len, mode='linear', align_corners=False).squeeze()
                
                # Next (M+1) -> Want M (Lower Pitch) -> Play Slower -> Longer Duration
                # Ratio = 2^(-1/12) approx 0.94387
                # New Len = Old Len / 0.94387... = Old Len * 1.05946
                if audio_next is not None:
                     audio_next = preprocess_sample(audio_next, item['midi'] + 1)
                     if audio_next is not None:
                         # Resample
                         ratio = 2**(-1/12) # < 1
                         new_len = int(len(audio_next) / ratio)
                         audio_next = audio_next.view(1, 1, -1)
                         audio_next = F.interpolate(audio_next, size=new_len, mode='linear', align_corners=False).squeeze()

                # Pad/Crop Helper
                def pad_crop(a):
                    if a is None: return None
                    if len(a) > CLIP_LEN: return a[:CLIP_LEN]
                    return F.pad(a, (0, CLIP_LEN - len(a)))

                t_main = pad_crop(audio_main)
                t_prev = pad_crop(audio_prev)
                t_next = pad_crop(audio_next) # Can be None
                
                b_target_main.append(t_main)
                
                # If neighbor missing, use Main as fallback (Loss(y, y_target) is just repeated, no harm, just weight)
                # Or better: Handling inside loss? 
                # Simplest: If missing, duplicate Main. "Robustness" falls back to standard loss.
                b_target_prev.append(t_prev if t_prev is not None else t_main)
                b_target_next.append(t_next if t_next is not None else t_main)

                b_midi_list.append(item['midi'])
                b_dyn_list.append(item['dyn_idx'])
                valid_count += 1
            except Exception as e: 
                # print(f"Error loading {item['midi']}: {e}")
                pass
            
        if valid_count == 0: continue
        
        # Prepare Tensors
        b_midi = torch.tensor(b_midi_list, device=device).float()
        b_dyn = torch.tensor(b_dyn_list, device=device).long()
        
        t_main_stack = torch.stack(b_target_main)
        t_prev_stack = torch.stack(b_target_prev)
        t_next_stack = torch.stack(b_target_next)
        
        print(f"Batch {b_idx+1}/{total_batches}: Notes {b_midi_list[0]:.0f}-{b_midi_list[-1]:.0f} ({valid_count} samples)")
        
        # L-BFGS for this batch
        optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, history_size=100, line_search_fn='strong_wolfe')
        
        def closure():
            optimizer.zero_grad()
            torch.manual_seed(42) # Determinism
            overrides = model(b_midi, b_dyn)
            phys = calculate_partials(b_midi, overrides, device=device)
            y_pred = diff_piano_render(
                phys["freqs"], phys["tau_s"], phys["tau_f"],
                phys["amps"], phys["w_curve"], CLIP_LEN,
                reverb_wet=phys.get("reverb_wet"), reverb_decay=phys.get("reverb_decay")
            )
            
            # Robust Loss: Average of 3
            # We used Main as fallback for missing neighbors, so simply averaging stacks works.
            l_main = loss_fn(y_pred, t_main_stack)
            l_prev = loss_fn(y_pred, t_prev_stack)
            l_next = loss_fn(y_pred, t_next_stack)
            
            loss = (l_main + l_prev + l_next) / 3.0
            
            loss.backward()
            return loss

        # Run Steps
        # Robust Refinement: 10 L-BFGS steps.
        
        initial_loss = closure().item()
        
        for step in range(10):
             loss = optimizer.step(closure)
             
        final_loss = loss.item()
        print(f"  Loss: {initial_loss:.4f} -> {final_loss:.4f}")
        
        # Periodic Save
        if b_idx % 5 == 0:
            torch.save({"model_state": model.state_dict()}, FINAL_CPT)

    torch.save({"model_state": model.state_dict()}, FINAL_CPT)
    print(f"Global Robust Refinement Complete. Saved {FINAL_CPT}")

if __name__ == "__main__":
    main()
