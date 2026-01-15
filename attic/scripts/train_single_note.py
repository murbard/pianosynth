import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import numpy as np

from pianosynth.optimization import PianoParam
from pianosynth.spectral import MultiResSTFTLoss, diff_piano_render
from pianosynth import synth

PROCESSED_DATA_DIR = Path("data/clean_et")
TARGET_FILE = "60_mf.pt"
OUTPUT_DIR = Path("results_single_note")

def save_wav(fname, audio_tensor, sr=44100):
    # Normalize to -1..1
    audio = audio_tensor.detach().cpu().numpy()
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    wavfile.write(fname, sr, (audio * 32767).astype(np.int16))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 1. Load Data
    target_path = PROCESSED_DATA_DIR / TARGET_FILE
    if not target_path.exists():
        print(f"Error: {target_path} not found.")
        return
        
    print(f"Loading target: {target_path}")
    target_audio_full = torch.load(target_path).float().to(device)
    
    # Clip to 2 seconds for training
    CLIP_LEN_SAMPLES = 44100 * 2
    if len(target_audio_full) > CLIP_LEN_SAMPLES:
        target_audio = target_audio_full[:CLIP_LEN_SAMPLES]
    else:
        target_audio = torch.cat([target_audio_full, torch.zeros(CLIP_LEN_SAMPLES - len(target_audio_full), device=device)])
    
    # 2. Model Setup
    model = PianoParam(device=device)
    loss_fn = MultiResSTFTLoss(device=device)
    
    # Learn velocity for this single sample
    # Init at 0.5 (mf)
    vel_logit = torch.nn.Parameter(torch.logit(torch.tensor(0.5, device=device)), requires_grad=True)
    
    # Learn Additive Noise Params
    # Gain: Init small, e.g. -60dB (approx 1e-3). Logit(1e-3) ~ -6.9
    # Color: 0=White, 1=Brown. Init at 0.5 (Pink-ish). Logit(0.5) = 0.
    noise_gain_logit = torch.nn.Parameter(torch.logit(torch.tensor(0.001, device=device)), requires_grad=True)
    noise_color_logit = torch.nn.Parameter(torch.logit(torch.tensor(0.5, device=device)), requires_grad=True)
    
    # Learn Reverb Params
    # Wet: Init small. Logit(0.05) ~ -2.9
    # Decay: Init 0.5s. Logit(0.5) = 0. (We sigmoid logit and scale if needed? Reverb decay is in seconds 0..inf? 
    # Let's map sigmoid to e.g. 0.1 to 5.0 seconds range?)
    # Simple sigmoid: 0..1s may be too short.
    # Let's use Softplus or Exp for decay time.
    # Init 0.5s.
    reverb_wet_logit = torch.nn.Parameter(torch.logit(torch.tensor(0.01, device=device)), requires_grad=True)
    # Learn log decay
    reverb_decay_log = torch.nn.Parameter(torch.log(torch.tensor(0.5, device=device)), requires_grad=True)
    
    optimizer = optim.Adam(list(model.parameters()) + [vel_logit, noise_gain_logit, noise_color_logit, reverb_wet_logit, reverb_decay_log], lr=1e-3)
    
    # 3. Training Loop
    EPOCHS = 2000
    pbar = tqdm(range(EPOCHS))
    history = []
    
    # Render Initial State for comparison
    with torch.no_grad():
        v_init = torch.sigmoid(vel_logit)
        n_gain_init = torch.sigmoid(noise_gain_logit)
        n_color_init = torch.sigmoid(noise_color_logit)
        rev_wet_init = torch.sigmoid(reverb_wet_logit)
        rev_decay_init = torch.exp(reverb_decay_log)
        
        overrides_init = model.get_overrides()
        from pianosynth.physics import calculate_partials
        phys_init = calculate_partials(
            midi=torch.tensor([60.0], device=device),
            velocity=v_init.unsqueeze(0),
            overrides=overrides_init,
            n_partials=64,
            device=device
        )
        y_init = diff_piano_render(
            freqs=phys_init["freqs"],
            tau_s=phys_init["tau_s"],
            tau_f=phys_init["tau_f"],
            amps=phys_init["amps"],
            w_curve=phys_init["w_curve"],
            dur_samples=CLIP_LEN_SAMPLES,
            noise_params={'gain': n_gain_init, 'color': n_color_init},
            reverb_params={'wet': rev_wet_init, 'decay': rev_decay_init}
        )
        initial_audio = y_init[0]

    for epoch in pbar:
        optimizer.zero_grad()
        
        v = torch.sigmoid(vel_logit)
        n_gain = torch.sigmoid(noise_gain_logit)
        n_color = torch.sigmoid(noise_color_logit)
        rev_wet = torch.sigmoid(reverb_wet_logit)
        rev_decay = torch.exp(reverb_decay_log)
        
        overrides = model.get_overrides()
        
        phys_out = calculate_partials(
            midi=torch.tensor([60.0], device=device),
            velocity=v.unsqueeze(0),
            overrides=overrides,
            n_partials=64,
            device=device
        )
        
        y_pred = diff_piano_render(
            freqs=phys_out["freqs"],
            tau_s=phys_out["tau_s"],
            tau_f=phys_out["tau_f"],
            amps=phys_out["amps"],
            w_curve=phys_out["w_curve"],
            dur_samples=CLIP_LEN_SAMPLES,
            noise_params={'gain': n_gain, 'color': n_color},
            reverb_params={'wet': rev_wet, 'decay': rev_decay}
        )
        
        loss = loss_fn(y_pred, target_audio.unsqueeze(0))
        
        loss.backward()
        optimizer.step()
        
        history.append(loss.item())
        if epoch % 100 == 0:
            pbar.set_description(f"L: {loss.item():.4f}")
            
    final_loss = history[-1]
    print(f"Final Loss: {final_loss}")
    
    # 4. Generate Results
    with torch.no_grad():
        final_audio = y_pred[0]
        
    # Save Loss
    with open(OUTPUT_DIR / "loss_history.json", "w") as f:
        json.dump(history, f)
        
    with open(OUTPUT_DIR / "final_loss.txt", "w") as f:
        f.write(str(final_loss))

    # Save Checkpoint
    save_path = OUTPUT_DIR / "params_c4.pt"
    torch.save({
        'model_state': model.state_dict(),
        'noise_params': {'gain': torch.sigmoid(noise_gain_logit), 'color': torch.sigmoid(noise_color_logit)},
        'reverb_params': {'wet': torch.sigmoid(reverb_wet_logit), 'decay': torch.exp(reverb_decay_log)},
        'velocity': torch.sigmoid(vel_logit)
    }, save_path)
    print(f"Checkpoint saved to {save_path}")

    # Save Audio Comparison
    # True, Default, Optimized
    def normalize_tensor(t):
        peak = t.abs().max()
        if peak > 1e-6:
            return t / peak * 0.95
        return t

    target_norm = normalize_tensor(target_audio)
    initial_norm = normalize_tensor(initial_audio)
    final_norm = normalize_tensor(final_audio)

    silence = torch.zeros(22050, device=device) # 0.5s silence
    full_seq = torch.cat([target_norm, silence, initial_norm, silence, final_norm])
    save_wav(OUTPUT_DIR / "comparison.wav", full_seq)
    
    # Save Images (Spectrograms)
    # Use matplotlib specgram
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    def plot_spec(ax, audio, title):
        audio_np = audio.detach().cpu().numpy()
        Pxx, freqs, bins, im = ax.specgram(audio_np, NFFT=1024, Fs=44100, noverlap=512)
        ax.set_title(title)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (s)")
        
    plot_spec(axes[0], target_norm, "Ground Truth (C4 mf) [Normalized]")
    plot_spec(axes[1], initial_norm, "Default Parameters [Normalized]")
    plot_spec(axes[2], final_norm, "Optimized Parameters (2000 epochs) [Normalized]")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "spectrograms.png")
    print("Results saved to results_single_note/")

if __name__ == "__main__":
    main()
