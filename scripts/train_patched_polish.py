import torch
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from pianosynth.optimization_batch import PianoParamPerKey
from pianosynth.spectral import MultiResSTFTLoss, diff_piano_render
from pianosynth.physics import calculate_partials

PROCESSED_DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("src/pianosynth/checkpoints")
START_CPT = CHECKPOINT_DIR / "params_patched_neighbors.pt"
FINAL_CPT = CHECKPOINT_DIR / "params_patched_polished.pt"

# List from the previous run
PATCHED_NOTES = [
    {'midi': 96, 'dyn': 'mf'}, # C7
    {'midi': 104, 'dyn': 'pp'}, # G#7
    {'midi': 24, 'dyn': 'mf'}, # C2
    {'midi': 76, 'dyn': 'ff'}, # E5
    {'midi': 81, 'dyn': 'ff'}, # A5
    {'midi': 83, 'dyn': 'mf'}, # B5
    {'midi': 98, 'dyn': 'mf'}, # D7
    {'midi': 72, 'dyn': 'pp'}, # C5
    {'midi': 23, 'dyn': 'pp'}, # B0
    {'midi': 21, 'dyn': 'mf'}, # A1 - Note: MIDI 21 is A0 actually? 
    # Wait, A0 is 21. A1 is 33.
    # The printed log said "A1". Let's verify MIDI conversion.
    # log: A1 mf, Gain 0.0503.
    # My previous script used: octave = (m // 12) - 1.
    # 21 // 12 = 1. 1-1 = 0. A0.
    # 33 // 12 = 2. 2-1 = 1. A1.
    # The loop was range(22, 108).
    # So 33 is valid.
    # Let's just hardcode the list based on the output for safety/speed.
    # Actually, I can just reconstruct it or make `apply_neighbor_subst.py` save a JSON list.
    # But hardcoding is faster for this one-off interactions.
    # Wait, the log printed: A1.
    # Let's trust the name calculaton:
    # A1 -> M=33.
]
# I will make the script flexible: pass the notes as list.
# Based on output:
# C7 (96), G#7 (104), C2 (36), E5 (76), A5 (81), B5 (83), D7 (98), C5 (72), B0 (23), A1 (33).
PARAMS_LIST = [
    (96, 'mf'), (104, 'pp'), (36, 'mf'), (76, 'ff'), (81, 'ff'), 
    (83, 'mf'), (98, 'mf'), (72, 'pp'), (23, 'pp'), (33, 'mf')
]

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
    print(f"Using device: {device}", flush=True)
    
    # 1. Load Model
    model = PianoParamPerKey(device=device)
    loss_fn = MultiResSTFTLoss(device=device)
    
    if START_CPT.exists():
        print(f"Loading {START_CPT}", flush=True)
        cpt = torch.load(START_CPT, map_location=device)
        model.load_state_dict(cpt["model_state"])
    else:
        print(f"Checkpoint {START_CPT} not found!", flush=True)
        return
        
    CLIP_LEN = 44100 * 2
    dyn_map = {'pp': 0, 'mf': 1, 'ff': 2}
            
    # 3. Prepare Batch
    b_midi_list = []
    b_dyn_list = []
    b_target_list = []
    
    print("Loading patched audio...", flush=True)
    for m, d_name in PARAMS_LIST:
        f = PROCESSED_DATA_DIR / f"{m}_{d_name}.pt"
        if f.exists():
            audio = torch.load(f).float().to(device)
            audio = preprocess_sample(audio, m)
            if audio is None or len(audio) < 1000: continue
            
            if len(audio) > CLIP_LEN:
                t = audio[:CLIP_LEN]
            else:
                t = F.pad(audio, (0, CLIP_LEN - len(audio)))
            
            b_target_list.append(t)
            b_midi_list.append(m)
            b_dyn_list.append(dyn_map[d_name])
            
    if not b_target_list:
        print("No valid audio found for patched notes.", flush=True)
        return
        
    b_midi = torch.tensor(b_midi_list, device=device).float()
    b_dyn = torch.tensor(b_dyn_list, device=device).long()
    b_targets = torch.stack(b_target_list)
    
    print(f"Polishing {len(b_midi)} patched notes...", flush=True)
    
    # 4. Train with L-BFGS
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

    initial_loss = closure().item()
    print(f"Initial Loss (Pre-Polish): {initial_loss:.4f}", flush=True)
    
    # Run loop
    for step in range(20):
         loss = optimizer.step(closure)
         if step % 5 == 0:
             print(f"Step {step}: {loss.item():.4f}", flush=True)
             
    final_loss = loss.item()
    print(f"Final Loss: {final_loss:.4f}", flush=True)
    
    torch.save({"model_state": model.state_dict()}, FINAL_CPT)
    print(f"Saved {FINAL_CPT}", flush=True)

if __name__ == "__main__":
    main()
