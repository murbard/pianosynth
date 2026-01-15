import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from pianosynth.optimization_batch import PianoParamPerKey
from pianosynth.optimization_batch import PianoParamPerKey
from pianosynth.spectral import MultiResSTFTLoss, diff_piano_render
from pianosynth.physics import calculate_partials

PROCESSED_DATA_DIR = Path("data/processed")
CHECKPOINT = Path("src/pianosynth/checkpoints/params_best_ever.pt")
OUT_DIR = Path("results_low_note_analysis")
OUT_DIR.mkdir(exist_ok=True)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    m = 102 # Patch Verified
    d_name = 'mf'
    d_idx = 1
    
    # Load GT
    # Load GT with Fallback (Matches comparison script logic)
    f_path = PROCESSED_DATA_DIR / f"{m}_{d_name}.pt"
    if not f_path.exists():
        print(f"GT not found for {m} {d_name}. Trying fallback...")
        found = False
        for alt in ['ff', 'pp', 'mf']:
             f_alt = PROCESSED_DATA_DIR / f"{m}_{alt}.pt"
             if f_alt.exists():
                 f_path = f_alt
                 print(f"Using fallback: {f_alt}")
                 found = True
                 break
        if not found:
             print("No GT found even with fallback.")
             return
    audio_gt = torch.load(f_path).float().to(device)
    
    CLIP_LEN = 44100 * 3 # 3 Seconds
    if len(audio_gt) > CLIP_LEN: audio_gt = audio_gt[:CLIP_LEN]
    else: audio_gt = F.pad(audio_gt, (0, CLIP_LEN - len(audio_gt)))
    
    # Load Model
    model = PianoParamPerKey(device=device)
    if CHECKPOINT.exists():
        cpt = torch.load(CHECKPOINT, map_location=device)
        model.load_state_dict(cpt["model_state"])
        print(f"Loaded {CHECKPOINT}")
    
    # Render
    m_t = torch.tensor([m], device=device).float()
    d_t = torch.tensor([d_idx], device=device).long()
    
    with torch.no_grad():
        seed = 42 + m + d_idx * 1000
        torch.manual_seed(seed)
        
        overrides = model(m_t, d_t)
        phys = calculate_partials(m_t, overrides, device=device)
        y_pred = diff_piano_render(
            phys["freqs"], phys["tau_s"], phys["tau_f"],
            phys["amps"], phys["w_curve"], CLIP_LEN,
            reverb_wet=phys.get("reverb_wet"), reverb_decay=phys.get("reverb_decay")
        )
        if y_pred.ndim == 2: y_pred = y_pred.squeeze(0)
    
    # Loss Analysis
    if audio_gt.ndim == 1: audio_gt_batch = audio_gt.unsqueeze(0)
    if y_pred.ndim == 1: y_pred_batch = y_pred.unsqueeze(0)
    
    loss_fn = MultiResSTFTLoss(device=device)
    total_loss = loss_fn(y_pred_batch, audio_gt_batch).item()
    print(f"Total MultiResSTFT Loss: {total_loss:.4f}")
    
    # Plot Spectrograms using Torch
    win_length = 2048
    hop_length = 512
    n_fft = 2048
    window = torch.hann_window(win_length, device=device)
    
    # GT STFT
    stft_gt = torch.stft(audio_gt, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
    mag_gt = torch.abs(stft_gt).cpu().numpy()
    log_jg_gt = np.log1p(mag_gt)
    
    # Pred STFT
    stft_pred = torch.stft(y_pred, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
    mag_pred = torch.abs(stft_pred).cpu().numpy()
    log_jg_pred = np.log1p(mag_pred)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(log_jg_gt, aspect='auto', origin='lower', cmap='inferno')
    plt.title(f"GT {m} {d_name} (Log Mag)")
    plt.colorbar()
    
    plt.subplot(2, 2, 2)
    plt.imshow(log_jg_pred, aspect='auto', origin='lower', cmap='inferno')
    plt.title(f"Pred {m} {d_name} (Loss {total_loss:.2f})")
    plt.colorbar()
    
    # Zoom Low Frequency (First 100 bins ~ 0-2100Hz)
    plt.subplot(2, 2, 3)
    plt.imshow(log_jg_gt[:100, :], aspect='auto', origin='lower', cmap='inferno')
    plt.title("GT Low Freq (<2kHz)")
    
    plt.subplot(2, 2, 4)
    plt.imshow(log_jg_pred[:100, :], aspect='auto', origin='lower', cmap='inferno')
    plt.title("Pred Low Freq (<2kHz)")
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "spectrogram_compare.png")
    print(f"Saved {OUT_DIR / 'spectrogram_compare.png'}")
    
    # Plot Average Spectrum
    plt.figure(figsize=(12, 6))
    S_gt = np.mean(mag_gt, axis=1)
    S_pred = np.mean(mag_pred, axis=1)
    # Freqs: 0 to Nyquist
    freqs = np.linspace(0, 44100/2, len(S_gt))
    
    plt.plot(freqs, 20*np.log10(S_gt + 1e-6), label='GT', alpha=0.7)
    plt.plot(freqs, 20*np.log10(S_pred + 1e-6), label='Pred', alpha=0.7, linestyle='--')
    plt.xlim(0, 5000)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Log Magnitude (dB)")
    plt.legend()
    plt.title("Average Log-Magnitude Spectrum (0-5kHz)")
    plt.grid(True, alpha=0.3)
    plt.savefig(OUT_DIR / "spectrum_compare.png")
    print(f"Saved {OUT_DIR / 'spectrum_compare.png'}")

if __name__ == "__main__":
    main()
