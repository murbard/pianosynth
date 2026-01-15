import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import json
from pianosynth.optimization import PianoParam

DATA_PATH = Path("data/processed/analysis_data.pt")
PARAM_OUT = Path("src/pianosynth/params_optimized.pt")

def main():
    if not DATA_PATH.exists():
        print("Data not found.")
        return
        
    raw_data = torch.load(DATA_PATH)
    # raw_data[midi][dyn] = dict
    
    # Flatten data
    # We need: midi_batch, vel_indices, target_freqs, target_decays, target_amps
    midis = []
    dyns = [] # strings
    target_f = []
    target_d = []
    target_a = []
    
    dyn_map = {'pp': 0, 'mf': 1, 'ff': 2}
    
    print("Loading data...")
    for m in raw_data:
        for d in raw_data[m]:
            midis.append(m)
            dyns.append(d)
            
            entry = raw_data[m][d]
            target_f.append(entry['freqs'])
            target_d.append(entry['decays'])
            target_a.append(entry['amps'])
            
    # To Tensor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    midi_t = torch.tensor(midis, device=device, dtype=torch.float32)
    
    # Pad if lengths differ? Assuming analyze_samples padded to N=64
    target_f = torch.stack(target_f).to(device)
    target_d = torch.stack(target_d).to(device)
    target_a = torch.stack(target_a).to(device)
    
    # Create Model
    model = PianoParam(device=device)
    
    # Learnable Velocities
    # We have N samples. Each has a velocity.
    # Init: pp=0.15, mf=0.5, ff=0.9
    n_samples = len(midis)
    init_vels = []
    for d in dyns:
        if d == 'pp': init_vels.append(0.15)
        elif d == 'mf': init_vels.append(0.5)
        elif d == 'ff': init_vels.append(0.9)
    
    # Learnable parameter constrained to [0, 1]
    # We use sigmoid(logit)
    vel_logits = torch.tensor(init_vels, device=device)
    vel_logits = torch.logit(vel_logits) # inverse sigmoid
    vel_logits = nn.Parameter(vel_logits, requires_grad=True)
    
    optimizer = optim.Adam(
        list(model.parameters()) + [vel_logits],
        lr=1e-2
    )
    
    print(f"Starting optimization on {n_samples} samples...")
    pbar = tqdm(range(5000))
    history = []
    
    for epoch in pbar:
        optimizer.zero_grad()
        
        # Get v
        velocities = torch.sigmoid(vel_logits)
        
        # Predict
        pred_f, pred_d, pred_a = model(midi_t, velocities)
        
        # Loss Masks
        # Mask out missing partials (target_f == 0)
        mask = (target_f > 1.0) 
        
        # 1. Frequency Loss (MSE)
        # Scale: ~1000Hz.
        diff_f = (pred_f - target_f) * mask
        loss_f = torch.mean(diff_f**2)
        
        # 2. Decay Loss (Log MSE) - WEIGHTED BY AMPLITUDE
        # We trust decay estimates of loud partials more than quiet ones.
        # target_d can be 100.0 (dummy). Mask those?
        d_mask = mask & (target_d < 50.0) & (target_d > 0.001)
        
        # Prevent log(0)
        p_d_safe = pred_d.clamp(min=1e-4)
        t_d_safe = target_d.clamp(min=1e-4)
        
        # Use target amplitude as confidence weight for decay
        # Normalize weights to sum to 1 (or mean)
        w_decay = target_a.clamp(min=0.0)
        w_decay = w_decay * d_mask.float()
        w_sum = w_decay.sum().clamp(min=1e-6)
        
        diff_d_sq = (torch.log(p_d_safe) - torch.log(t_d_safe))**2
        loss_d = torch.sum(diff_d_sq * w_decay) / w_sum
        
        # 3. Amplitude Loss (Log MSE)
        a_mask = mask & (target_a > 1e-6)
        p_a_safe = pred_a.clamp(min=1e-6)
        t_a_safe = target_a.clamp(min=1e-6)
        diff_a = (torch.log(p_a_safe) - torch.log(t_a_safe)) * a_mask
        loss_a = torch.mean(diff_a**2)
        
        # Combine
        # Relative Freq Loss
        diff_f_rel = ((pred_f - target_f) / (target_f + 1.0)) * mask
        loss_f_rel = torch.mean(diff_f_rel**2)
        
        loss = 100.0 * loss_f_rel + 1.0 * loss_d + 1.0 * loss_a
        
        loss.backward()
        optimizer.step()
        
        # Log history
        history.append({
            "epoch": epoch,
            "loss_total": loss.item(),
            "loss_freq": loss_f_rel.item() * 100.0,
            "loss_decay": loss_d.item(),
            "loss_amp": loss_a.item()
        })
        
        pbar.set_description(f"L_f: {loss_f_rel.item():.2e} L_d: {loss_d.item():.4f} L_a: {loss_a.item():.4f}")
        
    print("Optimization complete.")
    
    # Save parameters
    state = {
        "model_state": model.state_dict(),
        "velocities": torch.sigmoid(vel_logits).detach().cpu(),
        "midi_indices": midis,
        "dynamics": dyns
    }
    torch.save(state, PARAM_OUT)
    print(f"Saved to {PARAM_OUT}")
    
    # Save History
    with open("loss_history.json", "w") as f:
        json.dump(history, f)
    print("Saved loss_history.json")

if __name__ == "__main__":
    main()
