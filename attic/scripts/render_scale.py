import torch
import torch.nn as nn
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import argparse

from pianosynth.optimization import PianoParam
from pianosynth.spectral import diff_piano_render
from pianosynth.physics import calculate_partials

CLIP_LEN_SAMPLES = 88200 # 2 seconds
SAMPLE_RATE = 44100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def render_scale(checkpoint_path, output_wav, output_png):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    model = PianoParam().to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    
    noise_params = checkpoint.get('noise_params', {'gain': 0.0, 'color': 0.5})
    reverb_params = checkpoint.get('reverb_params', {'wet': 0.0, 'decay': 0.5})
    
    overrides = model.get_overrides()
    
    # Generate Chromatic C3 to C6
    pitches = list(range(48, 85)) 
    
    total_samples = len(pitches) * int(0.5 * SAMPLE_RATE) + int(2.0 * SAMPLE_RATE)
    output_buffer = torch.zeros(total_samples, device=DEVICE)
    
    print("Rendering scale...")
    
    current_sample = 0
    STEP_SIZE = int(0.5 * SAMPLE_RATE) # 0.5s per note
    RENDER_LEN = int(2.0 * SAMPLE_RATE) # Allow 2s ring out
    
    with torch.no_grad():
        for midi_note in pitches:
            velocity = 0.5 # mf
            
            phys_out = calculate_partials(
                midi=torch.tensor([float(midi_note)], device=DEVICE),
                velocity=torch.tensor([float(velocity)], device=DEVICE),
                overrides=overrides,
                n_partials=64,
                device=DEVICE
            )
            
            note_audio = diff_piano_render(
                freqs=phys_out["freqs"],
                tau_s=phys_out["tau_s"],
                tau_f=phys_out["tau_f"],
                amps=phys_out["amps"],
                w_curve=phys_out["w_curve"],
                dur_samples=RENDER_LEN,
                noise_params=noise_params,
                reverb_params=reverb_params
            )
            
            # Add to buffer
            end_sample = current_sample + RENDER_LEN
            if end_sample > total_samples:
                end_sample = total_samples
                note_audio = note_audio[..., :end_sample-current_sample]
                
            output_buffer[current_sample:end_sample] += note_audio[0]
            current_sample += STEP_SIZE
            
    # Normalize
    output_buffer = output_buffer / (output_buffer.abs().max() + 1e-6)
    
    # Save Wav
    wav_data = (output_buffer.cpu().numpy() * 32767).astype(np.int16)
    wav.write(output_wav, SAMPLE_RATE, wav_data)
    print(f"Saved audio to {output_wav}")
    
    # Save Spectrogram using Matplotlib
    plt.figure(figsize=(12, 6))
    plt.specgram(output_buffer.cpu().numpy(), NFFT=2048, Fs=SAMPLE_RATE, noverlap=1024, cmap='inferno')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram of Synthesized Scale')
    plt.tight_layout()
    plt.savefig(output_png)
    print(f"Saved spectrogram to {output_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--wav", type=str, default="results/scale_test.wav")
    parser.add_argument("--png", type=str, default="results/scale_spectrogram.png")
    args = parser.parse_args()
    
    render_scale(args.checkpoint, args.wav, args.png)
