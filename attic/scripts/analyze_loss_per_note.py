import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from pianosynth.optimization_batch import PianoParamPerKey
from pianosynth.spectral import MultiResSTFTLoss, diff_piano_render
from pianosynth.physics import calculate_partials

PROCESSED_DATA_DIR = Path("data/processed")
CHECKPOINT = Path("src/pianosynth/checkpoints/params_best_ever.pt")
OUT_DIR = Path("results_categorical")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def preprocess_sample_simple(audio, midi, sr=44100):
   # Simplified preprocessing matches training
    f_et = 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
    # ... (Reuse training logic if possible, or just load raw if preprocessed matches)
    # The training script logic includes trimming silence.
    # Let's replicate trimming.
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
    
    # Load Model
    model = PianoParamPerKey(device=device)
    if not CHECKPOINT.exists():
        print("Checkpoint not found.")
        return
    cpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(cpt["model_state"])
    
    loss_fn = MultiResSTFTLoss(device=device)
    
    # Iterate all notes/dynamics
    midis = list(range(21, 109))
    dyns = ['pp', 'mf', 'ff']
    dyn_map = {'pp': 0, 'mf': 1, 'ff': 2}
    
    results = {d: [] for d in dyns}
    
    print("Calculating Loss per Note...")
    
    for m in tqdm(midis):
        for d_name in dyns:
            f_path = PROCESSED_DATA_DIR / f"{m}_{d_name}.pt"
            if not f_path.exists():
                results[d_name].append(np.nan)
                continue
                
            try:
                audio_gt = torch.load(f_path).float().to(device)
            except:
                results[d_name].append(np.nan)
                continue
                
            audio_gt = preprocess_sample_simple(audio_gt, m)
            if audio_gt is None or len(audio_gt) < 1000:
                results[d_name].append(np.nan)
                continue
                
            # Render Length
            CLIP_LEN = 44100 * 2
            if len(audio_gt) > CLIP_LEN:
                audio_gt = audio_gt[:CLIP_LEN]
            else:
                # Pad
                pad = CLIP_LEN - len(audio_gt)
                audio_gt = torch.cat([audio_gt, torch.zeros(pad, device=device)])
                
            # Run Model
            m_t = torch.tensor([m], device=device).float()
            d_idx = dyn_map[d_name]
            d_t = torch.tensor([d_idx], device=device).long()
            
            with torch.no_grad():
                overrides = model(m_t, d_t)
                phys_out = calculate_partials(m_t, overrides, device=device)
                
                y_pred = diff_piano_render(
                    freqs=phys_out["freqs"],
                    tau_s=phys_out["tau_s"],
                    tau_f=phys_out["tau_f"],
                    amps=phys_out["amps"],
                    w_curve=phys_out["w_curve"],
                    dur_samples=CLIP_LEN,
                    reverb_wet=phys_out.get("reverb_wet"),
                    reverb_decay=phys_out.get("reverb_decay")
                )
                # y_pred: [1, T] or [T] ?
                if y_pred.ndim == 1: y_pred = y_pred.unsqueeze(0)
                if audio_gt.ndim == 1: audio_gt = audio_gt.unsqueeze(0)
                
                loss = loss_fn(y_pred, audio_gt)
                results[d_name].append(loss.item())

    # Plotting
    plt.figure(figsize=(12, 6))
    
    x = np.array(midis)
    colors = {'pp': 'blue', 'mf': 'green', 'ff': 'red'}
    
    for d_name in dyns:
        y = np.array(results[d_name])
        # Mask NaNs
        mask = ~np.isnan(y)
        plt.plot(x[mask], y[mask], '.-', color=colors[d_name], label=d_name, alpha=0.7)
        
    plt.xlabel("MIDI Note")
    plt.ylabel("Multi-Res STFT Loss")
    plt.title("Spectral Loss vs MIDI Note (Lower is Better)")
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    
    # Highlight "Bad" zones mentioned by user
    # Bass < G4 (MIDI ~55)
    plt.axvline(x=55, color='k', linestyle='--', alpha=0.5, label='G4')
    # C6 (MIDI 84)
    plt.axvline(x=84, color='k', linestyle='--', alpha=0.5, label='C6')
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "loss_per_note.png")
    print(f"Saved plot to {OUT_DIR / 'loss_per_note.png'}")

if __name__ == "__main__":
    main()
