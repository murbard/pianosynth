import torch
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from pianosynth.optimization_batch import PianoParamPerKey
from pianosynth.spectral import MultiResSTFTLoss, diff_piano_render
from pianosynth.physics import calculate_partials

PROCESSED_DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("src/pianosynth/checkpoints")
START_CPT = CHECKPOINT_DIR / "params_lbfgs_global_deep.pt" 
FINAL_CPT = CHECKPOINT_DIR / "params_outliers_fixed.pt"

# Defined from the analysis
OUTLIERS = [
    {'midi': 79, 'dyn': 'ff'}, # G5
    {'midi': 66, 'dyn': 'pp'}, # F#4
    {'midi': 74, 'dyn': 'pp'}, # D5
    {'midi': 69, 'dyn': 'pp'}, # A4
    {'midi': 97, 'dyn': 'pp'}, # C#7
    {'midi': 62, 'dyn': 'pp'}, # D4
    {'midi': 87, 'dyn': 'pp'}, # D#6
    {'midi': 97, 'dyn': 'ff'}, # C#7
    {'midi': 60, 'dyn': 'pp'}, # C4
    {'midi': 89, 'dyn': 'pp'}, # F6
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
    print(f"Using device: {device}")
    
    # 1. Load Model
    model = PianoParamPerKey(device=device)
    loss_fn = MultiResSTFTLoss(device=device)
    
    if START_CPT.exists():
        print(f"Loading {START_CPT}")
        cpt = torch.load(START_CPT, map_location=device)
        model.load_state_dict(cpt["model_state"])
    else:
        print(f"Checkpoint {START_CPT} not found!")
        return
        
    CLIP_LEN = 44100 * 2
    dyn_map = {'pp': 0, 'mf': 1, 'ff': 2}
    
    # 2. Reset Parameters for Outliers
    print("Resetting parameters for outliers...")
    with torch.no_grad():
        for o in OUTLIERS:
            m = o['midi']
            d_idx = dyn_map[o['dyn']]
            
            # Index in parameter tensor: [midi - 21, dyn_idx, :]
            idx_m = m - 21
            
            # Reset all parameters for this key/dyn
            for name, param in model.params.items():
                # param shape is (88, 3)
                # We reset to random normal.
                # Careful: param.data[idx_m, d_idx] is a scalar (0-d tensor)
                # assigning a 1-element random tensor is fine.
                param.data[idx_m, d_idx] = torch.randn([], device=device)
            
    # 3. Prepare Batch
    b_midi_list = []
    b_dyn_list = []
    b_target_list = []
    
    print("Loading outlier audio...")
    for o in OUTLIERS:
        f = PROCESSED_DATA_DIR / f"{o['midi']}_{o['dyn']}.pt"
        if f.exists():
            audio = torch.load(f).float().to(device)
            audio = preprocess_sample(audio, o['midi'])
            if audio is None or len(audio) < 1000: continue
            
            if len(audio) > CLIP_LEN:
                t = audio[:CLIP_LEN]
            else:
                t = F.pad(audio, (0, CLIP_LEN - len(audio)))
            
            b_target_list.append(t)
            b_midi_list.append(o['midi'])
            b_dyn_list.append(dyn_map[o['dyn']])
            
    if not b_target_list:
        print("No valid audio found for outliers.")
        return
        
    b_midi = torch.tensor(b_midi_list, device=device).float()
    b_dyn = torch.tensor(b_dyn_list, device=device).long()
    b_targets = torch.stack(b_target_list)
    
    print(f"Training {len(b_midi)} outliers...")
    
    # 4. Train with L-BFGS
    # Use standard settings LR=1.0, Strong Wolfe
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

    # Run for a good amount of steps. 
    # Since we reset them, 20 might not be enough. Let's do 50.
    
    initial_loss = closure().item()
    print(f"Initial Loss (Post-Reset): {initial_loss:.4f}")
    
    for step in range(50):
         loss = optimizer.step(closure)
         if step % 10 == 0:
             print(f"Step {step}: {loss.item():.4f}")
             
    final_loss = loss.item()
    print(f"Final Loss: {final_loss:.4f}")
    
    torch.save({"model_state": model.state_dict()}, FINAL_CPT)
    print(f"Saved {FINAL_CPT}")

if __name__ == "__main__":
    main()
