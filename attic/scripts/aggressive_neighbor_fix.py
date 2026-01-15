import torch
import torch.nn.functional as F
import argparse
import json
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
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--json_out", type=str, required=True, help="List of modified notes")
    parser.add_argument("--threshold", type=float, default=0.01)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    
    model = PianoParamPerKey(device=device)
    if not Path(args.input).exists():
        print(f"Checkpoint not found: {args.input}", flush=True)
        return
    cpt = torch.load(args.input, map_location=device)
    model.load_state_dict(cpt["model_state"])
    loss_fn = MultiResSTFTLoss(device=device)
    
    # --- START FROM BEST LOGIC ---
    best_cpt_path = Path(args.output).parent / "params_best_ever.pt"
    best_loss = float('inf')
    best_state = None
    
    # Check "Best Ever" file
    if best_cpt_path.exists():
        try:
            b = torch.load(best_cpt_path, map_location=device)
            if 'loss' in b:
                best_loss = b['loss']
                best_state = b['model_state']
                print(f"Found Best Ever: {best_loss:.4f}", flush=True)
            else:
                # Fallback: Treat as unknown, don't use unless valid
                pass
        except: pass
        
    # Check Input Loss (if metadata exists, else we might need to eval?)
    # For now, if Input has no metadata, we assume it's NOT better than best_loss unless proven.
    # But Input is usually the Result of Polish of the previous step.
    # Polish script NOW saves metadata.
    input_loss = float('inf')
    if 'loss' in cpt:
        input_loss = cpt['loss']
        print(f"Input Loss: {input_loss:.4f}", flush=True)
    
    # Decide Starting Point
    # CRITICAL CHANGE: User wants to continue "Polish -> Scan" cycle even if Polish increased loss.
    # Strict reversion to Best Ever kills exploration (since we just scanned Best Ever and found 0 changes).
    # We will Start from Input, but keep tracking Best Ever.
    
    # if best_state is not None and best_loss < input_loss:
    #     print(f"Using Best Ever (Loss {best_loss:.4f}) instead of Input (Loss {input_loss:.4f})", flush=True)
    #     model.load_state_dict(best_state)
    #     current_global_loss = best_loss
    # else:
    
    print(f"Using Input (Loss {input_loss}) - Exploration Mode", flush=True)
    current_global_loss = input_loss # Might be inf if not metadated
        
    best_global_loss = best_loss if best_loss != float('inf') else current_global_loss
    
    midis = list(range(22, 108))
    dyns = ['pp', 'mf', 'ff']
    dyn_map = {'pp': 0, 'mf': 1, 'ff': 2}
    CLIP_LEN = 44100 * 2
    
    # Pre-cache audio to avoid disk I/O in inner loop
    # 88*3 audio clips might be too big for RAM? (88*3*88200*4 bytes) ~ 80MB. No problem.
    cache_audio = {}
    print("Caching audio targets...", flush=True)
    for m in midis:
        for d in dyns:
            f = PROCESSED_DATA_DIR / f"{m}_{d}.pt"
            if f.exists():
                try:
                    a = torch.load(f).float().to(device)
                    a = preprocess_sample_simple(a, m)
                    if a is not None and len(a) >= 1000:
                         if len(a) > CLIP_LEN: a = a[:CLIP_LEN]
                         else: a = F.pad(a, (0, CLIP_LEN - len(a)))
                         if a.ndim == 1: a = a.unsqueeze(0)
                         cache_audio[(m, d)] = a
                except: pass

    modified_notes = set() # (midi, dyn_str)
    
    best_global_loss = float('inf')
    
    loop_iter = 0
    while True:
        loop_iter += 1
        changes_this_pass = 0
        total_gain = 0.0
        pass_total_loss = 0.0
        pass_notes_count = 0
        
        print(f"--- Inner Replacement Loop {loop_iter} ---", flush=True)
        
        # We iterate sequentially.
        # Order matters! Forward then backward? Random?
        # Let's just do sequential for now.
        
        for m in midis:
            for d_name in dyns:
                if (m, d_name) not in cache_audio: continue
                
                audio_gt = cache_audio[(m, d_name)]
                d_idx = dyn_map[d_name]
                m_t = torch.tensor([m], device=device).float()
                d_t = torch.tensor([d_idx], device=device).long()
                idx = m - 21
                
                # Fetch Current Params
                # Helper to get params
                def get_params(midi_idx, dyn_idx):
                    p = {}
                    for n, param in model.params.items():
                        # CRITICAL: Must clone to avoid view mutation!
                        p[n] = param.data[midi_idx, dyn_idx].clone()
                    return p
                
                curr_params = get_params(idx, d_idx)
                
                # Candidates
                candidates = []
                
                # Helper for combinations
                def mix(param_dicts, weights):
                    # param_dicts: list of dicts
                    # weights: list of floats, sum doesn't have to be 1 (e.g. extrapolation)
                    if len(param_dicts) != len(weights): return None
                    res = {}
                    keys = param_dicts[0].keys()
                    for k in keys:
                        val = 0
                        valid = True
                        for i, p in enumerate(param_dicts):
                            if p is None: 
                                valid = False
                                break
                            val += p[k] * weights[i]
                        if not valid: return None
                        res[k] = val
                    return res

                # Fetch Neighbors
                p_curr = get_params(idx, d_idx)
                
                # We need safe access for L-1, R+1 etc.
                # idx corresponds to m. m=22 -> idx=1.
                # L=idx-1 (0). L-1=idx-2 (-1).
                # Max m=107 -> idx=86. (Len 88: 0..87).
                # R=idx+1 (87). R+1=idx+2 (88 -> Invalid).
                
                def get_p_safe(i):
                    if i < 0 or i >= 88: return None
                    return get_params(i, d_idx)

                p_L   = get_p_safe(idx-1)
                p_R   = get_p_safe(idx+1)
                p_LL  = get_p_safe(idx-2)
                p_RR  = get_p_safe(idx+2)
                
                # Define 12+ Candidates
                candidates = []
                # 1. Current (Baseline)
                candidates.append({'type': 'curr', 'params': p_curr})
                
                # Basic Neighbors
                if p_L: candidates.append({'type': 'L', 'params': p_L})
                if p_R: candidates.append({'type': 'R', 'params': p_R})
                
                # Averages
                if p_L and p_R:
                    candidates.append({'type': 'Avg_LR', 'params': mix([p_L, p_R], [0.5, 0.5])})
                    
                # Weighted Averages
                if p_L and p_R:
                    candidates.append({'type': 'WAvg_2L_1R', 'params': mix([p_L, p_R], [2/3, 1/3])})
                    candidates.append({'type': 'WAvg_1L_2R', 'params': mix([p_L, p_R], [1/3, 2/3])})
                    
                # Wider Smoothing (4-point)
                if p_L and p_R and p_LL and p_RR:
                    candidates.append({'type': 'Smooth_4', 'params': mix([p_LL, p_L, p_R, p_RR], [0.25, 0.25, 0.25, 0.25])})
                
                # 3-Point Smoothing
                if p_LL and p_L and p_R:
                     candidates.append({'type': 'Smooth_3L', 'params': mix([p_LL, p_L, p_R], [1/3, 1/3, 1/3])})
                if p_L and p_R and p_RR:
                     candidates.append({'type': 'Smooth_3R', 'params': mix([p_L, p_R, p_RR], [1/3, 1/3, 1/3])})

                # Linear Extrapolation (Trend)
                if p_LL and p_L:
                    # Pred = L + (L - LL) = 2L - LL
                    candidates.append({'type': 'Extrap_L', 'params': mix([p_L, p_LL], [2.0, -1.0])})
                if p_RR and p_R:
                    # Pred = R + (R - RR) = 2R - RR
                    candidates.append({'type': 'Extrap_R', 'params': mix([p_R, p_RR], [2.0, -1.0])})
                    
                # Wide Average (LL, RR) - skipping immediate
                if p_LL and p_RR:
                    candidates.append({'type': 'Avg_Wide', 'params': mix([p_LL, p_RR], [0.5, 0.5])})

                # Mix with Current (Relaxation)
                candidates.append({'type': 'Relax_L', 'params': mix([p_curr, p_L], [0.5, 0.5])}) if p_L else None
                candidates.append({'type': 'Relax_R', 'params': mix([p_curr, p_R], [0.5, 0.5])}) if p_R else None
                if p_L and p_R:
                    candidates.append({'type': 'Relax_All', 'params': mix([p_L, p_curr, p_R], [0.33, 0.33, 0.33])})

                # Filter Nones (if mix returned None)
                candidates = [c for c in candidates if c['params'] is not None]
                
                # Eval
                best_loss = float('inf')
                best_cand = None
                curr_loss = float('inf')
                
                # We need to eval efficiently.
                # Inject params into model for rendering.
                # To avoid tons of memory ops, we reuse space.
                
                # Save Original (it's already in curr_params, but we need to restore model state)
                # Actually, we can just overwrite model.params.data
                
                for cand in candidates:
                    # Inject
                    for k, v in cand['params'].items():
                        model.params[k].data[idx, d_idx] = v
                        
                    with torch.no_grad():
                        # Fix seed for determinism in both physics and rendering
                        # Use a seed specific to the note/dyn to avoid correlation, but consistent across passes?
                        # Actually just consistent is enough. 
                        # Use 42 + m + d_idx * 1000
                        seed = 42 + m + d_idx * 1000
                        torch.manual_seed(seed)
                        
                        ov = model(m_t, d_t)
                        phys = calculate_partials(m_t, ov, device=device)
                        y = diff_piano_render(
                            phys["freqs"], phys["tau_s"], phys["tau_f"],
                            phys["amps"], phys["w_curve"], CLIP_LEN,
                            reverb_wet=phys.get("reverb_wet"), reverb_decay=phys.get("reverb_decay")
                        )
                        if y.ndim == 1: y = y.unsqueeze(0)
                        loss = loss_fn(y, audio_gt).item()
                        
                    cand['loss'] = loss
                    if cand['type'] == 'curr':
                        curr_loss = loss
                    
                    if loss < best_loss:
                        best_loss = loss
                        best_cand = cand
                
                # Check outcome
                loss_for_note = curr_loss
                if best_cand['type'] != 'curr' and (curr_loss - best_loss) > args.threshold:
                    # Apply Change
                    for k, v in best_cand['params'].items():
                        model.params[k].data[idx, d_idx] = v
                    
                    changes_this_pass += 1
                    total_gain += (curr_loss - best_loss)
                    modified_notes.add((m, d_name))
                    loss_for_note = best_loss
                else:
                    # Restore Current
                    for k, v in curr_params.items():
                        model.params[k].data[idx, d_idx] = v
                
                pass_total_loss += loss_for_note
                pass_notes_count += 1
                        
        avg_loss = pass_total_loss / pass_notes_count if pass_notes_count > 0 else 0.0
        print(f"  Pass {loop_iter}: {changes_this_pass} changes, Global Avg Loss {avg_loss:.4f}, Total Gain {total_gain:.4f}", flush=True)
        
        # Track Best
        if avg_loss < best_global_loss:
            best_global_loss = avg_loss
            
            # Save "Best Ever" (Symlink-like)
            best_path_fixed = Path(args.output).parent / "params_best_ever.pt"
            torch.save({
                "model_state": model.state_dict(),
                "loss": best_global_loss
            }, best_path_fixed)
            
            # Save "Best Value" (History)
            best_path_val = Path(args.output).parent / f"params_best_loss_{best_global_loss:.6f}.pt"
            torch.save({
                "model_state": model.state_dict(),
                "loss": best_global_loss
            }, best_path_val)
            
            print(f"  [New Best] Saved {best_path_val}", flush=True)
        
        if changes_this_pass == 0:
            break
            
        if loop_iter >= 1000: # Increased limit
             print("  Inner loop safety limit (1000) reached.")
             break

    print(f"Total Unique Notes Modified: {len(modified_notes)}", flush=True)
    
    # Save Output
    torch.save({
        "model_state": model.state_dict(),
        "loss": avg_loss # Save current loss
    }, args.output)
    print(f"Saved checkpoint: {args.output}", flush=True)
    
    # Save JSON list
    # Convert set to list of dicts
    out_list = [{'midi': m, 'dyn': d} for (m, d) in modified_notes]
    with open(args.json_out, 'w') as f:
        json.dump(out_list, f, indent=2)
    print(f"Saved modified notes list: {args.json_out}", flush=True)

if __name__ == "__main__":
    main()
