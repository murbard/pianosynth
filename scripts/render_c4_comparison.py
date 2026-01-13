import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from pianosynth.optimization import PianoParam
from pianosynth.spectral import diff_piano_render
from pianosynth.physics import calculate_partials

PARAM_PATH = Path("src/pianosynth/params_c4_mf.pt")
PROCESSED_DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path(".")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Parameters
    if not PARAM_PATH.exists():
        print(f"Error: {PARAM_PATH} not found.")
        return
        
    print(f"Loading parameters from {PARAM_PATH}...")
    checkpoint = torch.load(PARAM_PATH, map_location=device)
    
    model = PianoParam(device=device)
    model.load_state_dict(checkpoint["model_state"])
    velocities = checkpoint["velocities"] # [0.5] if batch size 1
    
    # 2. Load True Audio
    # Find 60_mf.pt
    true_file = PROCESSED_DATA_DIR / "60_mf.pt"
    if not true_file.exists():
        print("Error: True audio 60_mf.pt not found.")
        return
        
    true_audio = torch.load(true_file).float().to(device)
    
    # 3. Optimize Generation
    # We need to render exactly what was trained.
    # The training loop used specific midi and velocity.
    midi_val = 60
    
    # We only have one velocity in the checkpoint corresponding to our single sample
    # The training script used batch_vel_idx = [0]
    velocity = velocities[0].unsqueeze(0) # Shape [1]
    
    midi_t = torch.tensor([midi_val], device=device, dtype=torch.float32)
    
    overrides = model.get_overrides()
    
    # Length: same as true audio? Or 2s clip?
    # Training was 2s (88200 samples)
    dur_samples = 88200
    
    print("Calculating physics...")
    phys_out = calculate_partials(
        midi=midi_t,
        velocity=velocity,
        overrides=overrides,
        n_partials=64,
        device=device
    )
    
    print(f"Rendering {dur_samples} samples...")
    synth_audio = diff_piano_render(
        freqs=phys_out["freqs"],
        tau_s=phys_out["tau_s"],
        tau_f=phys_out["tau_f"],
        amps=phys_out["amps"],
        w_curve=phys_out["w_curve"],
        dur_samples=dur_samples
    )
    
    synth_audio = synth_audio.squeeze() # [samples]
    
    # 4. Save WAVs
    # True
    true_audio_cpu = true_audio.cpu().numpy()
    sf.write(OUTPUT_DIR / "c4_mf_true.wav", true_audio_cpu, 44100)
    print("Saved c4_mf_true.wav")
    
    # Synth
    # Normalize synth to avoid clipping if necessary, or match true peak?
    # Generally better to check levels.
    synth_audio_cpu = synth_audio.detach().cpu().numpy()
    
    # Simple peak normalization to -1dB if it's too quiet/loud?
    # Or just raw to see energy match?
    # Let's just save raw first.
    sf.write(OUTPUT_DIR / "c4_mf_synth.wav", synth_audio_cpu, 44100)
    print("Saved c4_mf_synth.wav")
    
if __name__ == "__main__":
    main()
