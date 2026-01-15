import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import json
import random

from pianosynth.optimization import PianoParam
from pianosynth.spectral import MultiResSTFTLoss, diff_piano_render

PROCESSED_DATA_DIR = Path("data/clean_et")
PARAM_OUT = Path("src/pianosynth/params_spectral.pt")

def load_audio_data(device="cpu", max_files=None):
    """
    Loads all processed audio files into memory.
    Returns: list of (midi, dynamic_str, audio_tensor)
    """
    meta_path = PROCESSED_DATA_DIR / "metadata.pt"
    if not meta_path.exists():
        return []
        
    meta = torch.load(meta_path)
    # meta[midi][dyn] = filename
    
    dataset = []
    
    # We need a fixed length for batching?
    # Or we can slice randomly during training.
    # Let's load the full tensors.
    
    files = list(PROCESSED_DATA_DIR.glob("*.pt"))
    files = [f for f in files if f.name != "metadata.pt" and f.name != "analysis_data.pt"]
    
    if max_files:
        files = files[:max_files]
        
    for f in tqdm(files, desc="Loading Audio"):
        # filename fmt: {midi}_{dyn}.pt
        parts = f.stem.split('_')
        midi = int(parts[0])
        dyn = parts[1]
        
        audio = torch.load(f).float().to(device)
        dataset.append({
            "midi": midi,
            "dyn": dyn,
            "audio": audio
        })
        
    return dataset

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    dataset = load_audio_data(device=device)
    if not dataset:
        print("No data found.")
        return
        
    # Model
    model = PianoParam(device=device)
    loss_fn = MultiResSTFTLoss(device=device)
    
    # Velocities
    # Map (midi, dyn) -> index
    # We optimize a velocity embedding per sample
    n_samples = len(dataset)
    
    # Init guess
    init_vels = []
    for d in dataset:
        if d['dyn'] == 'pp': v=0.2
        elif d['dyn'] == 'mf': v=0.5
        elif d['dyn'] == 'ff': v=0.8
        else: v=0.5
        init_vels.append(v)
        
    vel_logits = torch.logit(torch.tensor(init_vels, device=device))
    vel_logits = torch.nn.Parameter(vel_logits, requires_grad=True)
    
    optimizer = optim.Adam(list(model.parameters()) + [vel_logits], lr=1e-3)
    
    # Training Loop
    # We train on random slices of audio? 
    # Or full 2-second clips?
    # Batch size 8
    BATCH_SIZE = 8
    CLIP_LEN_samples = 44100 * 2 # 2 seconds
    EPOCHS = 2000
    
    pbar = tqdm(range(EPOCHS))
    history = []
    
    for epoch in pbar:
        # Mini-batch
        batch_indices = random.sample(range(n_samples), BATCH_SIZE)
        
        batch_midi = []
        batch_vel_idx = []
        batch_target_audio = []
        
        for idx in batch_indices:
            batch_midi.append(dataset[idx]['midi'])
            batch_vel_idx.append(idx)
            
            # Slice audio
            audio = dataset[idx]['audio']
            if len(audio) > CLIP_LEN_samples:
                # Take start? Or random?
                # Start is most important for attack/decay
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
        # Maps (midi, vel, overrides) -> (freqs, tau_s, tau_f, amps, w_curve)
        from pianosynth.physics import calculate_partials
        
        phys_out = calculate_partials(
            midi=m_t,
            velocity=vs,
            overrides=overrides,
            n_partials=64, # or higher? let's stick to 64 for training speed
            device=device
        )
        
        # 3. Render
        y_pred = diff_piano_render(
            freqs=phys_out["freqs"],
            tau_s=phys_out["tau_s"],
            tau_f=phys_out["tau_f"],
            amps=phys_out["amps"],
            w_curve=phys_out["w_curve"],
            dur_samples=CLIP_LEN_samples
        )
        
        # Loss
        loss = loss_fn(y_pred, targets)
        
        loss.backward()
        optimizer.step()
        
        history.append({"epoch": epoch, "loss": loss.item()})
        pbar.set_description(f"L: {loss.item():.4f}")
        
    print("Spectral Optimization Complete.")
    
    state = {
        "model_state": model.state_dict(),
        "velocities": torch.sigmoid(vel_logits.detach()),
        # we lose mapping to filename if we shuffle? 
        # But we just need latent params for synthesis
    }
    torch.save(state, PARAM_OUT)
    
    with open("loss_spectral.json", "w") as f:
        json.dump(history, f)

if __name__ == "__main__":
    main()
