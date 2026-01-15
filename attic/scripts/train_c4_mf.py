import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import json
import random

from pianosynth.optimization import PianoParam
from pianosynth.spectral import MultiResSTFTLoss, diff_piano_render
from pianosynth.physics import calculate_partials

PROCESSED_DATA_DIR = Path("data/processed")
PARAM_OUT = Path("src/pianosynth/params_c4_mf.pt")

def load_audio_data(device="cpu"):
    """
    Loads C4 mf audio file into memory.
    Returns: list of (midi, dynamic_str, audio_tensor)
    """
    files = list(PROCESSED_DATA_DIR.glob("*.pt"))
    dataset = []
    
    # We specifically want midi 60 (C4) and dynamic 'mf'
    target_midi = 60
    target_dyn = 'mf'
        
    for f in tqdm(files, desc="Loading Audio"):
        if f.name == "metadata.pt" or f.name == "analysis_data.pt":
            continue
            
        # filename fmt: {midi}_{dyn}.pt
        parts = f.stem.split('_')
        midi = int(parts[0])
        dyn = parts[1]
        
        if midi == target_midi and dyn == target_dyn:
            audio = torch.load(f).float().to(device)
            dataset.append({
                "midi": midi,
                "dyn": dyn,
                "audio": audio
            })
            print(f"Loaded target file: {f.name}")
            break # Found it
            
    return dataset

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    dataset = load_audio_data(device=device)
    if not dataset:
        print("No C4 mf data found. Run preprocess.py first?")
        return
        
    loss_fn = MultiResSTFTLoss(device=device)
    n_samples = len(dataset)

    # Init guess for mf = 0.5
    init_vels = [0.5] * n_samples
        
    vel_logits = torch.logit(torch.tensor(init_vels, device=device))
    vel_logits = torch.nn.Parameter(vel_logits, requires_grad=True)
    
    model = PianoParam(device=device)

    # Load Checkpoint logic
    start_epoch = 0
    if PARAM_OUT.exists():
        print(f"Resuming from {PARAM_OUT}...")
        checkpoint = torch.load(PARAM_OUT, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=False)
        
        # Recover velocity logits from sigmoid values
        saved_vels = checkpoint["velocities"] 
        # saved_vels is sigmoid(logits)
        # logits = logit(saved_vels)
        # Avoid numerical instability at 0/1
        saved_vels_clamped = saved_vels.clamp(1e-6, 1-1e-6)
        vel_logits.data = torch.logit(saved_vels_clamped)
        
        # If we saved epoch/history, we could resume count, but simpler to just run N more
        print("Checkpoint loaded. Continuing training...")

    optimizer = optim.Adam(list(model.parameters()) + [vel_logits], lr=1e-3)
    
    # Training Loop
    BATCH_SIZE = 1 # Single note
    CLIP_LEN_samples = 44100 * 2 # 2 seconds
    EPOCHS = 2000
    
    pbar = tqdm(range(EPOCHS))
    history = []
    
    for epoch in pbar:
        # Full batch every time for single sample
        batch_indices = [0]
        
        batch_midi = []
        batch_vel_idx = []
        batch_target_audio = []
        
        for idx in batch_indices:
            batch_midi.append(dataset[idx]['midi'])
            batch_vel_idx.append(idx)
            
            # Slice audio
            audio = dataset[idx]['audio']
            if len(audio) > CLIP_LEN_samples:
                # Always take start for stability on single note
                target = audio[:CLIP_LEN_samples]
            else:
                # Pad
                pad = torch.zeros(CLIP_LEN_samples - len(audio), device=device)
                target = torch.cat([audio, pad])
            batch_target_audio.append(target)
            
        m_t = torch.tensor(batch_midi, device=device, dtype=torch.float32)
        v_idx = torch.tensor(batch_vel_idx, device=device)
        targets = torch.stack(batch_target_audio)
        
        optimizer.zero_grad()
        
        # Get Velocities
        vs = torch.sigmoid(vel_logits[v_idx])
        
        # 1. Get Params Overrides from Model
        overrides = model.get_overrides()
        
        # 2. Physics Calculation
        phys_out = calculate_partials(
            midi=m_t,
            velocity=vs,
            overrides=overrides,
            n_partials=64, 
            device=device
        )
        
        # 3. Render
        y_pred = diff_piano_render(
            freqs=phys_out["freqs"],
            tau_s=phys_out["tau_s"],
            tau_f=phys_out["tau_f"],
            amps=phys_out["amps"],
            w_curve=phys_out["w_curve"],
            dur_samples=CLIP_LEN_samples,
            reverb_wet=overrides.get("reverb_wet"),
            reverb_decay=overrides.get("reverb_decay")
        )
        
        # Loss
        loss = loss_fn(y_pred, targets)
        
        loss.backward()
        optimizer.step()
        
        history.append({"epoch": epoch, "loss": loss.item()})
        if epoch % 10 == 0:
            pbar.set_description(f"L: {loss.item():.4f}")
        
    print("C4 mf Optimization Complete.")
    
    state = {
        "model_state": model.state_dict(),
        "velocities": torch.sigmoid(vel_logits.detach()),
    }
    torch.save(state, PARAM_OUT)
    
    with open("loss_c4_mf.json", "w") as f:
        json.dump(history, f)

if __name__ == "__main__":
    main()
