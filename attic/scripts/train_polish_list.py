import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
import json
from pathlib import Path
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
    parser.add_argument("--json_list", type=str, required=True)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    
    # Load List
    with open(args.json_list, 'r') as f:
        notes_to_polish = json.load(f)
        
    if not notes_to_polish:
        print("No notes to polish.", flush=True)
        # Copy input to output
        if args.input != args.output:
             import shutil
             shutil.copy(args.input, args.output)
        return

    # Load Model
    model = PianoParamPerKey(device=device)
    cpt = torch.load(args.input, map_location=device)
    model.load_state_dict(cpt["model_state"])
    loss_fn = MultiResSTFTLoss(device=device)
    
    dyn_map = {'pp': 0, 'mf': 1, 'ff': 2}
    CLIP_LEN = 44100 * 2
    
    # Prepare Batch
    b_midi_list = []
    b_dyn_list = []
    b_targets_list = []
    
    print(f"Loading audio for {len(notes_to_polish)} notes...", flush=True)
    
    for item in notes_to_polish:
        m = item['midi']
        d = item['dyn']
        f = PROCESSED_DATA_DIR / f"{m}_{d}.pt"
        if f.exists():
            try:
                a = torch.load(f).float().to(device)
                a = preprocess_sample_simple(a, m)
                if a is not None and len(a) >= 1000:
                     if len(a) > CLIP_LEN: a = a[:CLIP_LEN]
                     else: a = F.pad(a, (0, CLIP_LEN - len(a)))
                     b_targets_list.append(a)
                     b_midi_list.append(m)
                     b_dyn_list.append(dyn_map[d])
            except: pass
            
    if not b_targets_list:
        print("No valid audio found.", flush=True)
        torch.save({"model_state": model.state_dict()}, args.output)
        return

    b_midi = torch.tensor(b_midi_list, device=device).float()
    b_dyn = torch.tensor(b_dyn_list, device=device).long()
    b_targets = torch.stack(b_targets_list)
    
    # Chunking to avoid OOM
    CHUNK_SIZE = 16
    total_notes = len(b_midi)
    
    print(f"Polishing {total_notes} notes in chunks of {CHUNK_SIZE}...", flush=True)
    
    for i in range(0, total_notes, CHUNK_SIZE):
        chunk_indices = slice(i, min(i + CHUNK_SIZE, total_notes))
        
        c_midi = b_midi[chunk_indices]
        c_dyn = b_dyn[chunk_indices]
        c_targets = b_targets[chunk_indices]
        
        print(f"  Chunk {i//CHUNK_SIZE + 1}: Notes {i} to {i+len(c_midi)}", flush=True)
        
        # L-BFGS for this chunk
        # We need to optimize ONLY parameters relevant to this chunk?
        # L-BFGS optimizes all parameters passed to it.
        # If we pass model.parameters(), it tracks everything.
        # But gradients for non-active notes will be zero.
        # However, L-BFGS history vectors will be huge (full model size).
        # And step computation will update ALL params (even if grad is zero, updates might happen due to history?)
        # Actually, with limited history L-BFGS, if grad is 0, direction might be 0? 
        # Standard L-BFGS might drift unchanged params if not careful? 
        # No, if grad is 0, and prev updates were 0, it stays 0.
        # BUT, to be safe and cleaner, we can just use the global optimizer. 
        # The main memory cost is the RENDER (intermediates).
        # The model size is small (88*3 params). 88x3 x 20(history) is tiny.
        # The OOM was in `diff_piano_render`.
        # So using global optimizer with Mini-Batch Forward is the trick?
        # NO. L-BFGS requires the SAME batch for closure evaluation.
        # We cannot change the batch inside one optimization step.
        # So we MUST run separate optimization loops for each chunk.
        
        optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, history_size=100, line_search_fn='strong_wolfe')
        
        def closure():
            optimizer.zero_grad()
            torch.manual_seed(42)
            overrides = model(c_midi, c_dyn)
            phys = calculate_partials(c_midi, overrides, device=device)
            y = diff_piano_render(
                phys["freqs"], phys["tau_s"], phys["tau_f"],
                phys["amps"], phys["w_curve"], CLIP_LEN,
                reverb_wet=phys.get("reverb_wet"), reverb_decay=phys.get("reverb_decay")
            )
            if y.ndim == 1: y = y.unsqueeze(0)
            loss = loss_fn(y, c_targets)
            loss.backward()
            return loss
            
        try:
            l_start = closure().item()
            for _ in range(20):
                optimizer.step(closure)
            l_end = closure().item()
            print(f"    Loss: {l_start:.4f} -> {l_end:.4f}", flush=True)
        except Exception as e:
            print(f"    Error in chunk: {e}", flush=True)

    # Final Global Eval for Metadata
    print("Evaluating Global Loss...", flush=True)
    # We need to eval ALL notes (not just polished ones? Or just polished ones?)
    # "Global Average Loss" implies all notes?
    # But this script only sees "notes_to_polish".
    # The output checkpoint is the full model.
    # The user wants "start from best loss ever". This implies Global Loss over 88 keys.
    # Evaluating 88 keys takes time. But it's safer.
    
    # We can reuse the code from aggressive_fix or just iterate.
    # Let's simple-eval 88 keys if possible.
    # We need audio for all 88 keys.
    # Reuse `preprocess_sample_simple`.
    
    global_loss_sum = 0.0
    global_count = 0
    full_midis = list(range(21, 109))
    full_dyns = ['pp', 'mf', 'ff']
    
    # Simple loop
    for m in full_midis:
         for d_name in full_dyns:
             f = PROCESSED_DATA_DIR / f"{m}_{d_name}.pt"
             if not f.exists(): continue
             try: 
                 a = torch.load(f).float().to(device)
                 if len(a) > CLIP_LEN: a = a[:CLIP_LEN]
                 else: a = F.pad(a, (0, CLIP_LEN - len(a)))
                 if a.ndim == 1: a = a.unsqueeze(0)
                 
                 m_t = torch.tensor([m], device=device).float()
                 d_t = torch.tensor([dyn_map[d_name]], device=device).long()
                 
                 with torch.no_grad():
                     # Fix seed
                     seed = 42 + m + dyn_map[d_name] * 1000
                     torch.manual_seed(seed)
                     
                     ov = model(m_t, d_t)
                     phys = calculate_partials(m_t, ov, device=device)
                     y = diff_piano_render(
                        phys["freqs"], phys["tau_s"], phys["tau_f"],
                        phys["amps"], phys["w_curve"], CLIP_LEN,
                        reverb_wet=phys.get("reverb_wet"), reverb_decay=phys.get("reverb_decay")
                     )
                     if y.ndim == 1: y = y.unsqueeze(0)
                     l = loss_fn(y, a).item()
                     global_loss_sum += l
                     global_count += 1
             except: pass
             
    final_global_loss = global_loss_sum / global_count if global_count > 0 else float('inf')
    print(f"Global Loss: {final_global_loss:.4f}", flush=True)
            
    # Check vs Best Ever
    best_path_fixed = Path(args.output).parent / "params_best_ever.pt"
    best_loss = float('inf')
    if best_path_fixed.exists():
        try:
             b = torch.load(best_path_fixed, map_location=device)
             if 'loss' in b: best_loss = b['loss']
        except: pass
        
    if final_global_loss < best_loss:
        print(f"  [New Best from Polish] Saving Best Ever (Loss {final_global_loss:.4f})", flush=True)
        # Save Best
        torch.save({
            "model_state": model.state_dict(),
            "loss": final_global_loss
        }, best_path_fixed)
        # Save History
        best_path_val = Path(args.output).parent / f"params_best_loss_{final_global_loss:.6f}.pt"
        torch.save({
            "model_state": model.state_dict(),
            "loss": final_global_loss
        }, best_path_val)

    torch.save({
        "model_state": model.state_dict(),
        "loss": final_global_loss
    }, args.output)
    print(f"Saved {args.output}", flush=True)

if __name__ == "__main__":
    main()
