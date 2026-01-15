import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pianosynth.optimization_batch import PianoParamPerKey
from pianosynth.spectral import MultiResSTFTLoss, diff_piano_render
from pianosynth.physics import calculate_partials

PROCESSED_DATA_DIR = Path("data/processed")
CHECKPOINT = Path("src/pianosynth/checkpoints/params_neighbor_fixed_final.pt")

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
    print(f"Using device: {device}", flush=True)
    
    # Load Model
    model = PianoParamPerKey(device=device)
    if not CHECKPOINT.exists():
        print("Checkpoint not found.", flush=True)
        return
    cpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(cpt["model_state"])
    
    loss_fn = MultiResSTFTLoss(device=device)
    
    midis = list(range(22, 108)) # Skip 21 and 108 boundaries for simplicity (need 2 neighbors)
    dyns = ['pp', 'mf', 'ff']
    dyn_map = {'pp': 0, 'mf': 1, 'ff': 2}
    
    improvements = []
    
    print("Checking Neighbor Averaging...", flush=True)
    
    params_cache = {} 
    # Pre-fetch all params to avoid repeated lookups if it's slow?
    # Actually direct access is fast.
    
    for m in tqdm(midis):
        for d_name in dyns:
            d_idx = dyn_map[d_name]
            
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
            
            if audio_gt.ndim == 1: audio_gt = audio_gt.unsqueeze(0)
            
            # 1. Current Loss
            m_t = torch.tensor([m], device=device).float()
            d_t = torch.tensor([d_idx], device=device).long()
            
            # Helper to run loss
            def get_loss(override_Model=None):
                with torch.no_grad():
                    # If we simply run model(m_t, d_t), we get current params.
                    # To test averaged params, we need to temporarily inject them OR
                    # manually construct the "overrides" dictionary.
                    # Constructing overrides manually is safer/cleaner than mutating model state.
                    
                    # BUT get_loss needs "overrides".
                    # Let's write a helper to fetch params for (m,d).
                    
                    def fetch_params(midi_val, dyn_val):
                         # Extract raw scaler values for this key/dyn
                         # Returns dict {param_name: scalar_tensor}
                         # We need to act as if we are inside PianoParamPerKey.forward but picking specific indices
                         idx = midi_val - 21
                         vals = {}
                         for name, param in model.params.items():
                             # param is (88, 3)
                             vals[name] = param[idx, dyn_val]
                         return vals
                    
                    # Compute averaged raw params
                    p_prev = fetch_params(m-1, d_idx)
                    p_next = fetch_params(m+1, d_idx)
                    
                    p_avg = {}
                    for k in p_prev:
                        p_avg[k] = (p_prev[k] + p_next[k]) / 2.0
                        
                    # Now we need to pass these raw params through the activation functions (softplus etc)
                    # to get the actual physics overrides.
                    # We can use model._apply_activations(p_avg)? No, logic is inside forward.
                    # We have to copy forward logic.
                    # Or we can temporarily patch the model parameter for index m.
                    
                    # Patching is easiest to ensure we use EXACT logic.
                    # Save old values
                    idx_m = m - 21
                    old_values = {}
                    for name, param in model.params.items():
                        old_values[name] = param.data[idx_m, d_idx].clone()
                        param.data[idx_m, d_idx] = p_avg[name]
                        
                    # Forward
                    overrides = model(m_t, d_t)
                    
                    # Restore
                    for name, param in model.params.items():
                         param.data[idx_m, d_idx] = old_values[name]
                         
                    phys = calculate_partials(m_t, overrides, device=device)
                    y_pred = diff_piano_render(
                        phys["freqs"], phys["tau_s"], phys["tau_f"],
                        phys["amps"], phys["w_curve"], CLIP_LEN,
                        reverb_wet=phys.get("reverb_wet"), reverb_decay=phys.get("reverb_decay")
                    )
                    if y_pred.ndim == 1: y_pred = y_pred.unsqueeze(0)
                    return loss_fn(y_pred, audio_gt).item()

            # Current Loss (Standard forward)
            with torch.no_grad():
                overrides = model(m_t, d_t)
                phys = calculate_partials(m_t, overrides, device=device)
                y_pred = diff_piano_render(
                    phys["freqs"], phys["tau_s"], phys["tau_f"],
                    phys["amps"], phys["w_curve"], CLIP_LEN,
                    reverb_wet=phys.get("reverb_wet"), reverb_decay=phys.get("reverb_decay")
                )
                if y_pred.ndim == 1: y_pred = y_pred.unsqueeze(0)
                l_curr = loss_fn(y_pred, audio_gt).item()
            
            # Avg Neighbor Loss
            l_avg = get_loss()
            
            if l_avg < l_curr:
                improvements.append({
                    'midi': m,
                    'dyn': d_name,
                    'l_curr': l_curr,
                    'l_avg': l_avg,
                    'diff': l_curr - l_avg
                })

    improvements.sort(key=lambda x: x['diff'], reverse=True)
    
    print("\nTop Improvements from Neighbor Averaging:", flush=True)
    print(f"{'Note':<10} {'Dyn':<5} {'Curr':<10} {'Avg':<10} {'Diff':<10}", flush=True)
    print("-" * 50, flush=True)
    
    for i in range(min(20, len(improvements))):
        o = improvements[i]
        try:
             note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
             octave = (o['midi'] // 12) - 1
             note = f"{note_names[o['midi'] % 12]}{octave}"
        except: note = str(o['midi'])
        
        print(f"{note:<10} {o['dyn']:<5} {o['l_curr']:.4f}     {o['l_avg']:.4f}     {o['diff']:.4f}", flush=True)

if __name__ == "__main__":
    main()
