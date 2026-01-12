import torch
import json
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from pianosynth.optimization import PianoParam
from pianosynth.spectral import MultiResSTFTLoss, diff_piano_render
from pianosynth.physics import calculate_partials

# Re-use dataset logic from train_spectral
class PianoAudioDataset(Dataset):
    def __init__(self, data_root, device="cpu"):
        self.files = sorted(list(Path(data_root).glob("*.wav")))
        self.device = device
        self.cache = []
        print("Loading audio for eval...")
        for f in tqdm(self.files):
            # Parse midi/vel from filename? 
            # Filename format: Piano.ff.B3.wav
            # logic in train_spectral used existing analysis data just for mapping?
            # actually train_spectral loads audio and parses filename.
            
            parts = f.name.split('.')
            # dynamic = parts[1] (pp, mf, ff)
            # note = parts[2] (A0, C4...)
            
            # Map note to midi
            # We need a helper, or just use train_params logic
            pass 
            # Let's trust train_spectral logic.
            # actually I'll just copy the necessary parsing logic here.

def parse_note(note_str):
    # e.g. C4, Bb3, F#5
    # simple parser
    notes = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    notes_alt = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
    octave = int(note_str[-1])
    pitch = note_str[:-1]
    
    try:
        idx = notes.index(pitch)
    except:
        idx = notes_alt.index(pitch)
        
    return idx + (octave + 1) * 12

def parse_dynamic(dyn_str):
    if dyn_str == "pp": return 25
    if dyn_str == "mf": return 70
    if dyn_str == "ff": return 110
    return 70

def load_data(data_root):
    # Load .pt files like train_spectral.py
    files = list(Path(data_root).glob("*.pt"))
    files = [f for f in files if f.name != "metadata.pt" and f.name != "analysis_data.pt"]
    
    data = []
    for f in files:
        # filename fmt: {midi}_{dyn}.pt
        parts = f.stem.split('_')
        if len(parts) < 2: continue
        
        midi = int(parts[0])
        dyn = parts[1]
        
        # Approximate velocity
        if dyn == 'pp': vel = 0.2
        elif dyn == 'mf': vel = 0.5
        elif dyn == 'ff': vel = 0.8
        else: vel = 0.5
        
        try:
            audio = torch.load(f).float()
            # Clip to 2 sec if needed, but train_spectral does this on batch.
            # We'll clip here to save RAM/time
            sr = 44100
            if len(audio) > sr * 2:
                audio = audio[:sr*2]
            
            data.append({
                "midi": midi,
                "vel": vel,
                "audio": audio,
                "sr": sr,
                "name": f.name
            })
        except:
            continue
    return data

    return data

def flatten_defaults(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten_defaults(v, prefix + k + "_"))
        elif isinstance(v, (int, float)):
            out[prefix + k] = torch.tensor(float(v))
        # Skip strings like "description"
    return out

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Data
    data = load_data("data/processed") # Correct path
    print(f"Loaded {len(data)} samples.")
    if len(data) == 0:
        print("No data found.")
        return
        
    # Validation subset (every 10th sample)
    val_data = data[::10]
    print(f"Eval set: {len(val_data)} samples.")
    
    # Load Models
    model = PianoParam(device=device)
    
    # Optimized
    ckpt = torch.load("src/pianosynth/params_spectral.pt", map_location=device)
    if "model_state" in ckpt: model.load_state_dict(ckpt["model_state"])
    else: model.load_state_dict(ckpt)
    
    opt_overrides = model.get_overrides()
    # Detach
    opt_overrides = {k: v.detach() for k, v in opt_overrides.items()}
    
    # Default
    # Correct path
    with open("src/pianosynth/default_params.json", "r") as f:
        defaults_json = json.load(f)
    def_overrides = flatten_defaults(defaults_json)
    # Move to device
    def_overrides = {k: v.to(device) for k, v in def_overrides.items()}
    
    # Loss Fn
    loss_fn = MultiResSTFTLoss(device=device)
    
    # Loop
    losses_def = []
    losses_opt = []
    
    for item in tqdm(val_data):
        midi = torch.tensor([item['midi']], device=device).float()
        vel = torch.tensor([item['vel']], device=device).float()
        target = item['audio'].to(device).unsqueeze(0) # [1, T]
        
        # Render Function
        def render(ov):
            phys = calculate_partials(midi, vel, ov, n_partials=64, device=device)
            return diff_piano_render(
                phys["freqs"], phys["tau_s"], phys["tau_f"], phys["amps"], phys["w_curve"],
                dur_samples=target.shape[-1]
            )
            
        # Defaults
        y_def = render(def_overrides)
        l_def = loss_fn(y_def, target).item()
        losses_def.append(l_def)
        
        # Optimized
        y_opt = render(opt_overrides)
        l_opt = loss_fn(y_opt, target).item()
        losses_opt.append(l_opt)
        
    print(f"\nMean Loss (Default):   {np.mean(losses_def):.4f}")
    print(f"Mean Loss (Optimized): {np.mean(losses_opt):.4f}")
    
    diff = np.mean(losses_opt) - np.mean(losses_def)
    print(f"Diff (Opt - Def): {diff:.4f}")
    if diff < 0:
        print("Optimization successfully reduced loss.")
    else:
        print("Optimization FAIL: Loss increased!?")

if __name__ == "__main__":
    main()
