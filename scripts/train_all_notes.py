import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import random

from pianosynth.optimization_batch import PianoParamPerKey
from pianosynth.spectral import MultiResSTFTLoss, diff_piano_render
from pianosynth.physics import calculate_partials

PROCESSED_DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("src/pianosynth/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def load_all_data(device="cpu"):
    """
    Loads all available processed note data.
    Returns: list of dicts
    """
    files = list(PROCESSED_DATA_DIR.glob("*.pt"))
    dataset = []
    
    print("Loading dataset...")
    for f in tqdm(files):
        # 60_mf.pt
        try:
            parts = f.stem.split('_')
            midi = int(parts[0])
            dyn = parts[1]
            if 21 <= midi <= 108:
                audio = torch.load(f).float().to(device)
                dataset.append({
                    "midi": midi,
                    "dyn": dyn,
                    "audio": audio
                })
        except:
            pass
            
    return dataset

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Train on full 260-sample dataset")
    parser.add_argument("--sparse", action="store_true", help="Train on sparse 27-sample dataset (Default)", default=True) 
    # Logic: if full is set, sparse is False.
    args = parser.parse_args()
    
    if args.full:
        print("Training on FULL dataset (all available notes/dynamics).")
        # Generate all 88 notes as targets
        import numpy as np
        target_midis = list(range(21, 109))
        # Dynamics: All
        target_dyns = ['pp', 'mf', 'ff'] # We will just look for all of them
        sparse_mode = False
    else:
        print("Training on SPARSE dataset (27 notes).")
        import numpy as np
        target_midis = np.linspace(21, 108, 27).round().astype(int)
        target_dyns = ['pp', 'mf', 'ff']
        sparse_mode = True
    
    # Load all and filter? Or specific load.
    # load_all_data iterates files. Let's filter inside extraction or after.
    # Faster to just load what we need if we knew filenames.
    # But files are "{midi}_{dyn}.pt".
    
    import torch.nn.functional as F
    
    def preprocess_sample(audio, midi, device="cpu", sr=44100):
        # 1. Trim Silence
        # Threshold: -60dB? 
        threshold = 1e-3
        # find first sample > threshold
        mask = torch.abs(audio) > threshold
        if mask.any():
             idx = torch.where(mask)[0][0]
             # Backtrack slightly to catch attack transient?
             idx = max(0, idx - 100)
             audio = audio[idx:]
        else:
             return None # Silent file?

        # 2. Tune to ET
        # Expected f0
        f_et = 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
        
        # Estimate f0
        # Simple HPS or just Max FFT peak in expected range
        # Only check first 0.5s ? 
        n_fft = 2048*4
        if len(audio) < n_fft: n_fft = len(audio)
        
        # Windowed
        w = torch.hann_window(n_fft, device=device)
        spec = torch.fft.rfft(audio[:n_fft] * w)
        mag = torch.abs(spec)
        freqs = torch.fft.rfftfreq(n_fft, d=1/sr).to(device)
        
        # Search peak near f_et (+- 1 semitone)
        # f_min = f_et * 2^(-1/12), f_max = f_et * 2^(1/12)
        idx_min = int((f_et * 0.94) / (sr/n_fft))
        idx_max = int((f_et * 1.06) / (sr/n_fft))
        idx_min = max(1, idx_min)
        idx_max = min(len(mag)-1, idx_max)
        
        if idx_max <= idx_min:
             # Just assume tuned if freq very low or resolution bad
             f_meas = f_et
        else:
            peak_idx = torch.argmax(mag[idx_min:idx_max]) + idx_min
            
            # Parabolic interpolation
            alpha = mag[peak_idx-1]
            beta = mag[peak_idx]
            gamma = mag[peak_idx+1]
            p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
            f_meas = (peak_idx + p) * (sr / n_fft)
            
        rate = f_et / f_meas
        
        # Resample if deviation > 1 cent?
        # 1 cent = 2^(1/1200) ~ 1.0005
        if abs(rate - 1.0) > 0.0005:
            # Resample
            # New length
            # If rate > 1 (Target > Meas), we need to speed up (shorter) ? 
            # No, if we play back faster it goes higher. 
            # Resampling: We have audio at sr. It has pitch f_meas.
            # We want it to be f_et.
            # Pitch shift = f_et / f_meas.
            # Ideally we play it at rate. 
            # So we interpolate: new_len = old_len / rate
            
            new_len = int(len(audio) / rate)
            # Use interpolate
            audio = audio.view(1, 1, -1)
            audio = F.interpolate(audio, size=new_len, mode='linear', align_corners=False)
            audio = audio.view(-1)
            
            # print(f"Resampled {midi}: {f_meas:.2f} -> {f_et:.2f} (r={rate:.4f})")
            
        return audio

    dataset = []
    print("Loading dataset...")
    
    if sparse_mode:
        # Loop specific targets
        loop_iter = list(enumerate(target_midis))
    else:
        # Loop all midis
        loop_iter = list(enumerate(target_midis))

    for i, m in loop_iter:
        if sparse_mode:
            # Sparse: Specific dynamic per note
            req_dyns = [target_dyns[i % 3]]
        else:
            # Full: All dynamics for this note
            req_dyns = ['pp', 'mf', 'ff']
            
        for d_req in req_dyns:
            f = PROCESSED_DATA_DIR / f"{m}_{d_req}.pt"
            
            # fallback if exact dyn missing?
            if not f.exists():
                if sparse_mode:
                    # Try others just to get something
                    found = False
                    for alt in ['mf', 'ff', 'pp']:
                        f_alt = PROCESSED_DATA_DIR / f"{m}_{alt}.pt"
                        if f_alt.exists():
                            f = f_alt
                            found = True
                            print(f"Substituted {alt} for {d_req} on {m}")
                            break
                    if not found:
                         # print(f"Missing data for {m}")
                         continue
                else:
                    # In full mode, if a dyn is missing, we just skip it.
                    continue
            
            # Additional check for file existence (redundant but safe)
            if not f.exists(): continue
        
        audio = torch.load(f).float().to(device)
        
        # Preprocess: Trim Silence + Tune to ET
        audio = preprocess_sample(audio, m, device=device)
        
        if audio is None or len(audio) < 1000:
             print(f"Audio invalid/silent after preprocessing for {m}")
             continue
             
        if torch.isnan(audio).any():
            print(f"NAN detected in loaded audio for {m} {d_req}!")
            continue
            
        dataset.append({
            "midi": int(m),
            "dyn": d_req,
            "audio": audio
        })

    if not dataset:
        print("No data found.")
        return
        
    print(f"Loaded {len(dataset)} sparse samples.")
    
    # Model
    model = PianoParamPerKey(device=device)
    loss_fn = MultiResSTFTLoss(device=device)
    
    # Velocities Initialization
    n_samples = len(dataset)
    init_vels = []
    for d in dataset:
        if d['dyn'] == 'pp': v=0.2
        elif d['dyn'] == 'mf': v=0.5
        elif d['dyn'] == 'ff': v=0.8
        else: v=0.5
        init_vels.append(v)
        
    vel_logits = torch.logit(torch.tensor(init_vels, device=device).clamp(0.01, 0.99))
    vel_logits = torch.nn.Parameter(vel_logits, requires_grad=True)
    
    optimizer = optim.Adam(list(model.parameters()) + [vel_logits], lr=1e-3)
    
    # Resume?
    MASTER_CPT = CHECKPOINT_DIR / "params_all_keys.pt"
    if MASTER_CPT.exists():
         print(f"Loading checkpoint {MASTER_CPT}")
         try:
             cpt = torch.load(MASTER_CPT, map_location=device)
             
             # Smart Init Logic for FULL run
             if args.full:
                 print("Applying Nearest-Neighbor Initialization from Sparse Checkpoint...")
                 state = cpt["model_state"]
                 
                 # Define sparse grid (must match what was trained)
                 # 27 notes from 21 to 108
                 sparse_midis = np.linspace(21, 108, 27).round().astype(int)
                 sparse_indices = sparse_midis - 21
                 
                 # Full indices
                 all_indices = np.arange(88)
                 
                 # Build map: full_idx -> nearest_sparse_idx
                 idx_map = {}
                 for idx in all_indices:
                     # Find nearest in sparse_indices
                     dist = np.abs(sparse_indices - idx)
                     nearest_idx = sparse_indices[np.argmin(dist)]
                     idx_map[idx] = nearest_idx
                     
                 # Apply to each param
                 new_state = {}
                 for k, v in state.items():
                     if v.numel() == 88:
                         # Interpolate
                         new_v = v.clone()
                         # CPU handling for indexing speed?
                         v_cpu = v.cpu()
                         new_v_cpu = new_v.cpu()
                         
                         for idx in all_indices:
                             source = idx_map[idx]
                             new_v_cpu[idx] = v_cpu[source]
                             
                         new_state[k] = new_v_cpu.to(device)
                     else:
                         new_state[k] = v
                 
                 model.load_state_dict(new_state)
                 print("Smart initialization complete.")
                 
             else:
                 model.load_state_dict(cpt["model_state"])

             # Velocities
             # If switching from sparse to full, velocities shape will mismatch (27 * 3 vs 88 * 3 approx)
             # Sparse dataset len != Full dataset len.
             # So we cannot load velocities if dataset size changed.
             saved_vels = cpt["velocities"]
             if saved_vels.shape == vel_logits.shape:
                 vel_logits.data = torch.logit(saved_vels.clamp(1e-6, 1-1e-6))
                 print("Loaded velocities.")
             else:
                 print(f"Velocities shape mismatch ({saved_vels.shape} vs {vel_logits.shape}). Resetting velocities (expected for sparse->full switch).")

         except Exception as e:
             print(f"Failed to load checkpoint: {e}")
             # traceback?
             import traceback
             traceback.print_exc()

    BATCH_SIZE = 4 # Small batch for debugging
    EPOCHS = 2000 
    CLIP_LEN = 44100 * 2
    
    targets_padded = []
    midis = []
    for d in dataset:
        audio = d['audio']
        if len(audio) > CLIP_LEN:
            t = audio[:CLIP_LEN]
        else:
            t = torch.cat([audio, torch.zeros(CLIP_LEN - len(audio), device=device)])
        targets_padded.append(t)
        midis.append(d['midi'])
        
    targets_padded = torch.stack(targets_padded)
    midis_tensor = torch.tensor(midis, device=device).float()
    
    print("Starting Training...")
    # torch.autograd.set_detect_anomaly(True) 
    
    # Debug: Check initial overrides
    dummy_midi = midis_tensor[:4]
    overrides = model(dummy_midi)
    print("Checking initial parameters...")
    for k, v in overrides.items():
        if torch.isnan(v).any():
            print(f"NAN in INIT param {k}")
        else:
            # print stats
            print(f"{k}: min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")
            
    pbar = tqdm(range(EPOCHS))
    
    for epoch in pbar:
        perm = torch.randperm(n_samples, device=device)
        epoch_loss = 0
        steps = 0
        
        for i in range(0, n_samples, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            
            batch_midi = midis_tensor[idx]
            batch_targets = targets_padded[idx]
            batch_vel_logits = vel_logits[idx]
            
            optimizer.zero_grad()
            
            vs = torch.sigmoid(batch_vel_logits)
            overrides = model(batch_midi)
            
            # Check params for nan
            for k, v in overrides.items():
                if torch.isnan(v).any():
                    print(f"NAN in override {k} at epoch {epoch}")
                    
            phys_out = calculate_partials(
                midi=batch_midi,
                velocity=vs,
                overrides=overrides,
                n_partials=64,
                device=device
            )
            
            # Check phys output
            for k, v in phys_out.items():
                 if torch.isnan(v).any():
                     print(f"NAN in phys_out[{k}] at epoch {epoch}")
                     # Break to avoid crash
                     return

            y_pred = diff_piano_render(
                freqs=phys_out["freqs"],
                tau_s=phys_out["tau_s"],
                tau_f=phys_out["tau_f"],
                amps=phys_out["amps"],
                w_curve=phys_out["w_curve"],
                dur_samples=CLIP_LEN,
                # reverb_wet=overrides.get("reverb_wet"),
                # reverb_decay=overrides.get("reverb_decay")
            )
            
            if torch.isnan(y_pred).any():
                print(f"NAN in y_pred at epoch {epoch}")
                return
            
            loss = loss_fn(y_pred, batch_targets)
            
            if torch.isnan(loss):
                print(f"NAN Loss at epoch {epoch}")
                return
                
            loss.backward()
            
            # Clip grads
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            steps += 1

            
        avg_L = epoch_loss / steps
        pbar.set_description(f"L: {avg_L:.4f}")
        
        if epoch % 100 == 0:
            torch.save({
                "model_state": model.state_dict(),
                "velocities": torch.sigmoid(vel_logits.detach())
            }, MASTER_CPT)
            
    # Final Save
    torch.save({
        "model_state": model.state_dict(),
        "velocities": torch.sigmoid(vel_logits.detach())
    }, MASTER_CPT)
    print("Training Complete.")

    # Post-process: Split params back to individual files for legacy scripts?
    # Or just use the new model for generation.
    # The generation script expects params_M.pt.
    # We should update generation script or split the checkppoint.
    # Let's split it for compatibility.
    
    print("Splitting checkpoints...")
    for m in range(21, 109):
        # Extract params for midi m
        idx = m - 21
        single_state = {}
        
        # We need to construct a "PianoParam" state dict from "PianoParamPerKey"
        # or just save "PianoParamPerKey" state and have a loader know what to do.
        # But legacy scripts load PianoParam (scalars).
        # We can extract the scalars for this key.
        
        # This is tricky because model structure differs.
        # But we wrote a new generate_full_comparison that loads per key.
        # We should update THAT script to load the MASTER checkpoint instead.
        pass

if __name__ == "__main__":
    main()

