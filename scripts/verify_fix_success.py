import torch
import pandas as pd
import numpy as np
from pathlib import Path
from pianosynth.optimization_batch import PianoParamPerKey
from pianosynth.spectral import MultiResSTFTLoss, diff_piano_render
from pianosynth.physics import calculate_partials

PROCESSED_DATA_DIR = Path("data/processed")
CHECKPOINT = Path("src/pianosynth/checkpoints/params_categorical_patched.pt")
CSV_PATH = Path("results_categorical/neighbor_improvements.csv")

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if not CHECKPOINT.exists():
        print("Patched checkpoint not found.")
        return
    if not CSV_PATH.exists():
        print("Improvements CSV not found.")
        return
        
    print(f"Loading {CHECKPOINT}...")
    cpt = torch.load(CHECKPOINT, map_location=device)
    model = PianoParamPerKey(device=device)
    model.load_state_dict(cpt["model_state"])
    
    loss_fn = MultiResSTFTLoss(device=device)
    
    df = pd.read_csv(CSV_PATH)
    print(f"Checking {len(df)} patched notes...")
    
    dyn_map = {'pp': 0, 'mf': 1, 'ff': 2}
    
    results = []
    
    CLIP_LEN = 44100 * 2
    
    for idx, row in df.iterrows():
        d_name = row['Dynamic']
        m = int(row['Note'])
        old_loss = row['SelfLoss']
        
        # Load Ground Truth
        f_path = PROCESSED_DATA_DIR / f"{m}_{d_name}.pt"
        if not f_path.exists(): continue
        
        try:
            audio_gt = torch.load(f_path).float().to(device)
        except: continue
        
        audio_gt = preprocess_sample_simple(audio_gt, m)
        if audio_gt is None or len(audio_gt) < 1000: continue
        
        if len(audio_gt) > CLIP_LEN:
            audio_gt = audio_gt[:CLIP_LEN]
        else:
            pad = CLIP_LEN - len(audio_gt)
            audio_gt = torch.cat([audio_gt, torch.zeros(pad, device=device)])
            
        audio_gt = audio_gt.unsqueeze(0)
        
        # Run Model (Current State)
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
            
            if y_pred.ndim == 1: y_pred = y_pred.unsqueeze(0)
            loss = loss_fn(y_pred, audio_gt).item()
            
        # Compare
        delta = old_loss - loss
        results.append({
            "Dynamic": d_name,
            "Note": m,
            "OldLoss": old_loss,
            "NewLoss": loss,
            "Delta": delta
        })
        
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("Delta", ascending=False)
    
    print("\nVerification Results (Top Improvements):")
    print(res_df.head(10).to_string(index=False))
    
    print("\nSummary:")
    improved = res_df[res_df["Delta"] > 0]
    print(f"Improved: {len(improved)} / {len(res_df)}")
    print(f"Average Delta: {res_df['Delta'].mean():.4f}")
    
    worsened = res_df[res_df["Delta"] < 0]
    if not worsened.empty:
        print(f"Worsened: {len(worsened)}")
        print(worsened.head(5).to_string(index=False))
    
    res_df.to_csv("results_categorical/patch_verification.csv", index=False)

if __name__ == "__main__":
    main()
