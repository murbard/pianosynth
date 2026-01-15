import torch
import torch.nn as nn
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

from pianosynth.optimization import PianoParam
from pianosynth.spectral import diff_piano_render, MultiResSTFTLoss
from pianosynth.physics import calculate_partials

SAMPLE_RATE = 44100
CLIP_LEN_SAMPLES = int(2.0 * SAMPLE_RATE) # 2s evaluation
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def profile_loss(checkpoint_path, data_dir, output_png):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    model = PianoParam().to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    
    noise_params = checkpoint.get('noise_params', {'gain': 0.0, 'color': 0.5})
    reverb_params = checkpoint.get('reverb_params', {'wet': 0.0, 'decay': 0.5})
    
    loss_fn = MultiResSTFTLoss(device=DEVICE)
    
    overrides = model.get_overrides()
    
    files = glob.glob(os.path.join(data_dir, "*.pt"))
    # Exclude metadata
    files = [f for f in files if "metadata" not in f]
    
    results = [] # (midi, velocity, loss)
    
    print(f"Profiling loss across {len(files)} files...")
    
    with torch.no_grad():
        for filepath in tqdm(files):
            # Parse filename: 60_mf.pt
            filename = os.path.basename(filepath)
            try:
                midi_str = filename.split('_')[0]
                dyn_str = filename.split('_')[1].replace('.pt', '')
                midi_val = float(midi_str)
                
                # Approximate velocity for plotting
                vel_map = {'pp': 0.25, 'mf': 0.5, 'ff': 0.75}
                vel_val = vel_map.get(dyn_str, 0.5)
            except:
                continue
                
            target_audio = torch.load(filepath, map_location=DEVICE)
            
            # Crop/Pad
            if target_audio.shape[-1] > CLIP_LEN_SAMPLES:
                target_audio = target_audio[..., :CLIP_LEN_SAMPLES]
            else:
                pad = CLIP_LEN_SAMPLES - target_audio.shape[-1]
                target_audio = torch.nn.functional.pad(target_audio, (0, pad))
            
            # Render with C4 params
            # We must pass the ACTUAL velocity of the note or the approximated one?
            # The model is parameterised by velocity.
            # If we use the velocity from the filename (pp/mf/ff), it's fair.
            
            phys_out = calculate_partials(
                midi=torch.tensor([midi_val], device=DEVICE),
                velocity=torch.tensor([vel_val], device=DEVICE),
                overrides=overrides, # These are Fixed C4 Params!
                n_partials=64,
                device=DEVICE
            )
            
            y_pred = diff_piano_render(
                freqs=phys_out["freqs"],
                tau_s=phys_out["tau_s"],
                tau_f=phys_out["tau_f"],
                amps=phys_out["amps"],
                w_curve=phys_out["w_curve"],
                dur_samples=CLIP_LEN_SAMPLES,
                noise_params=noise_params,
                reverb_params=reverb_params
            )
            
            loss = loss_fn(y_pred, target_audio.unsqueeze(0))
            
            results.append((midi_val, vel_val, loss.item()))
            
    # Plotting
    results = np.array(results)
    # results[:,0] = midi
    # results[:,1] = vel
    # results[:,2] = loss
    
    midis = results[:, 0]
    vels = results[:, 1]
    losses = results[:, 2]
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    # Color by velocity
    scatter = plt.scatter(midis, losses, c=vels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Velocity (approx)')
    
    plt.xlabel('MIDI Pitch')
    plt.ylabel('Loss (STFT + Artifact)')
    plt.title('Generalization Error of C4 Parameters across Keyboard')
    plt.grid(True, alpha=0.3)
    
    # Mark C4
    plt.axvline(x=60, color='r', linestyle='--', label='C4 (Trained Note)')
    plt.legend()
    
    plt.savefig(output_png)
    print(f"Profile saved to {output_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="results_single_note/params_c4.pt")
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--png", type=str, default="results_single_note/loss_profile.png")
    args = parser.parse_args()
    
    profile_loss(args.checkpoint, args.data, args.png)
