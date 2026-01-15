import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
from pathlib import Path
from tqdm import tqdm
from pianosynth.optimization_batch import PianoParamPerKey
from pianosynth.spectral import MultiResSTFTLoss, diff_piano_render
from pianosynth.physics import calculate_partials

PROCESSED_DATA_DIR = Path("data/processed")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output checkpoint")
    parser.add_argument("--threshold", type=float, default=0.01, help="Improvement threshold")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    
    # 1. Load Model
    model = PianoParamPerKey(device=device)
    if not Path(args.input).exists():
        print(f"Checkpoint {args.input} not found.", flush=True)
        return
    
    print(f"Loading {args.input}...", flush=True)
    cpt = torch.load(args.input, map_location=device)
    model.load_state_dict(cpt["model_state"])
    
    loss_fn = MultiResSTFTLoss(device=device)
    
    midis = list(range(22, 108)) 
    dyns = ['pp', 'mf', 'ff']
    dyn_map = {'pp': 0, 'mf': 1, 'ff': 2}
    
    patched_notes = []
    
    print("Scanning for Neighbor Averaging improvements...", flush=True)
    
    CLIP_LEN = 44100 * 2
    
    # We need to gather audio for potential polish later, so cache paths?
    # Actually just re-load later to save memory.
    
    for m in midis: # tqdm(midis):
        # Skip outputting progress bar for loop cleanliness
        for d_name in dyns:
            d_idx = dyn_map[d_name]
            f_path = PROCESSED_DATA_DIR / f"{m}_{d_name}.pt"
            if not f_path.exists(): continue
            
            try:
                audio_gt = torch.load(f_path).float().to(device)
            except: continue
            
            audio_gt = preprocess_sample_simple(audio_gt, m)
            if audio_gt is None or len(audio_gt) < 1000: continue
            if len(audio_gt) > CLIP_LEN: audio_gt = audio_gt[:CLIP_LEN]
            else: audio_gt = F.pad(audio_gt, (0, CLIP_LEN - len(audio_gt)))
            if audio_gt.ndim == 1: audio_gt = audio_gt.unsqueeze(0)
            
            m_t = torch.tensor([m], device=device).float()
            d_t = torch.tensor([d_idx], device=device).long()
            
            # Helper: Get interpolated params
            idx = m - 21
            avg_params = {}
            for name, param in model.params.items():
                p_prev = param.data[idx-1, d_idx]
                p_next = param.data[idx+1, d_idx]
                avg_params[name] = (p_prev + p_next) / 2.0
            
            # Helper: Eval Loss
            def eval_loss(p_overrides=None):
                 if p_overrides:
                    old_vals = {}
                    for name, param in model.params.items():
                        old_vals[name] = param.data[idx, d_idx].clone()
                        param.data[idx, d_idx] = p_overrides[name]
                 
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
                    
                 if p_overrides:
                    for name, param in model.params.items():
                         param.data[idx, d_idx] = old_vals[name]
                 return loss

            l_curr = eval_loss()
            l_new = eval_loss(avg_params)
            
            if (l_curr - l_new) > args.threshold:
                # Apply Patch
                for name, param in model.params.items():
                    param.data[idx, d_idx] = avg_params[name]
                
                patched_notes.append({
                    'midi': m, 'dyn': d_name, 'd_idx': d_idx,
                    'gain': l_curr - l_new,
                    'audio': audio_gt
                })
                # print(f"  Patching {m} {d_name}: {l_curr:.4f} -> {l_new:.4f} (Gain {l_curr-l_new:.4f})", flush=True)

    if not patched_notes:
        print("No patches applied. Converged.", flush=True)
        torch.save({"model_state": model.state_dict()}, args.output)
        return

    print(f"Patched {len(patched_notes)} notes. Polishing...", flush=True)
    
    # Polish Phase
    b_midi = torch.tensor([p['midi'] for p in patched_notes], device=device).float()
    b_dyn = torch.tensor([p['d_idx'] for p in patched_notes], device=device).long()
    b_targets = torch.cat([p['audio'] for p in patched_notes]) # (B, T)
    
    optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, history_size=100, line_search_fn='strong_wolfe')
    
    def closure():
        optimizer.zero_grad()
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

    l_start = closure().item()
    print(f"  Polish Start Loss: {l_start:.4f}", flush=True)
    
    for _ in range(20):
        optimizer.step(closure)
        
    l_end = closure().item()
    print(f"  Polish End Loss:   {l_end:.4f}", flush=True)
    
    torch.save({"model_state": model.state_dict()}, args.output)
    print(f"Saved {args.output}", flush=True)
    
    # Verify improvements
    print("  Top Fixes:", flush=True)
    patched_notes.sort(key=lambda x: x['gain'], reverse=True)
    for p in patched_notes[:5]:
        print(f"   M={p['midi']} {p['dyn']} Gain={p['gain']:.4f}", flush=True)

if __name__ == "__main__":
    main()
