import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pianosynth.optimization_batch import PianoParamPerKey
from pianosynth.spectral import MultiResSTFTLoss, diff_piano_render
from pianosynth.physics import calculate_partials

PROCESSED_DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("src/pianosynth/checkpoints")
INPUT_CPT = CHECKPOINT_DIR / "params_lbfgs_global_deep.pt"
OUTPUT_CPT = CHECKPOINT_DIR / "params_patched_neighbors.pt"

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
    if not INPUT_CPT.exists():
        print("Checkpoint not found.", flush=True)
        return
    cpt = torch.load(INPUT_CPT, map_location=device)
    model.load_state_dict(cpt["model_state"])
    
    loss_fn = MultiResSTFTLoss(device=device)
    
    midis = list(range(22, 108)) 
    dyns = ['pp', 'mf', 'ff']
    dyn_map = {'pp': 0, 'mf': 1, 'ff': 2}
    
    patched_count = 0
    patched_notes = []
    
    print("Evaluating and Patching Neighbor Averages...", flush=True)
    
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
            
            m_t = torch.tensor([m], device=device).float()
            d_t = torch.tensor([d_idx], device=device).long()
            
            # Helper to fetch and average params
            def get_avg_params():
                 idx = m - 21
                 vals = {}
                 for name, param in model.params.items():
                     p_prev = param.data[idx-1, d_idx]
                     p_next = param.data[idx+1, d_idx]
                     vals[name] = (p_prev + p_next) / 2.0
                 return vals
            
            # Helper to eval loss
            def eval_loss(params_dict=None):
                 # If params_dict provided, inject it temporarily
                 if params_dict:
                    idx_m = m - 21
                    old_values = {}
                    for name, param in model.params.items():
                        old_values[name] = param.data[idx_m, d_idx].clone()
                        param.data[idx_m, d_idx] = params_dict[name]
                 
                 with torch.no_grad():
                    overrides = model(m_t, d_t)
                    phys = calculate_partials(m_t, overrides, device=device)
                    y_pred = diff_piano_render(
                        phys["freqs"], phys["tau_s"], phys["tau_f"],
                        phys["amps"], phys["w_curve"], CLIP_LEN,
                        reverb_wet=phys.get("reverb_wet"), reverb_decay=phys.get("reverb_decay")
                    )
                    if y_pred.ndim == 1: y_pred = y_pred.unsqueeze(0)
                    loss = loss_fn(y_pred, audio_gt).item()
                    
                 if params_dict:
                    # Restore (for checking, we don't commit yet)
                    for name, param in model.params.items():
                         param.data[idx_m, d_idx] = old_values[name]
                 return loss

            l_curr = eval_loss()
            avg_params = get_avg_params()
            l_avg = eval_loss(avg_params)
            
            # Threshold: Improvement must be > 0.05
            if (l_curr - l_avg) > 0.05:
                # Apply Patch PERMANENTLY
                idx_m = m - 21
                for name, param in model.params.items():
                    param.data[idx_m, d_idx] = avg_params[name]
                
                patched_count += 1
                patched_notes.append({
                    'midi': m, 'dyn': d_name, 
                    'l_old': l_curr, 'l_new': l_avg, 
                    'gain': l_curr - l_avg
                })

    print(f"\nPatched {patched_count} notes.", flush=True)
    if patched_count > 0:
        print(f"{'Note':<10} {'Dyn':<5} {'Old':<10} {'New':<10} {'Gain':<10}", flush=True)
        print("-" * 50, flush=True)
        patched_notes.sort(key=lambda x: x['gain'], reverse=True)
        for p in patched_notes:
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            octave = (p['midi'] // 12) - 1
            n_str = f"{note_names[p['midi'] % 12]}{octave}"
            print(f"{n_str:<10} {p['dyn']:<5} {p['l_old']:.4f}     {p['l_new']:.4f}     {p['gain']:.4f}", flush=True)
            
        torch.save({"model_state": model.state_dict()}, OUTPUT_CPT)
        print(f"Saved patched checkpoint to {OUTPUT_CPT}", flush=True)
    else:
        print("No patches applied.", flush=True)

if __name__ == "__main__":
    main()
