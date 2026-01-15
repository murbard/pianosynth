import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import glob
import os
import random

from pianosynth.optimization import PianoParam
from pianosynth.spectral import diff_piano_render, MultiResSTFTLoss
from pianosynth.physics import calculate_partials

# --- Configuration ---
BATCH_SIZE = 8 # Reduced to 8 (Safe)
EPOCHS = 500 # ~16000 updates
CLIP_LEN_SAMPLES = 88200 # 2 seconds
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Dataset ---
class PianoDataset(Dataset):
    def __init__(self, data_dir, device='cpu'):
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        # Exclude metadata.pt
        self.files = [f for f in self.files if "metadata.pt" not in f]
        print(f"Found {len(self.files)} training samples.")
        self.device = device
        
        # Preload data
        print("Preloading dataset...")
        self.data = []
        for filepath in tqdm(self.files):
            filename = os.path.basename(filepath)
            midi_str = filename.split('_')[0]
            dyn_str = filename.split('_')[1].replace('.pt', '')
            
            midi_val = float(midi_str)
            vel_map = {'pp': 0.25, 'mf': 0.5, 'ff': 0.75}
            vel_val = vel_map.get(dyn_str, 0.5)
            
            target_audio = torch.load(filepath, map_location=self.device)
            
            # Crop/Pad
            if target_audio.shape[-1] > CLIP_LEN_SAMPLES:
                target_audio = target_audio[..., :CLIP_LEN_SAMPLES]
            else:
                pad_len = CLIP_LEN_SAMPLES - target_audio.shape[-1]
                target_audio = torch.nn.functional.pad(target_audio, (0, pad_len))
            
            self.data.append((target_audio, torch.tensor(midi_val, device=self.device), torch.tensor(vel_val, device=self.device)))
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def train():
    print(f"Using device: {DEVICE}")
    
    # 1. Model Setup
    model = PianoParam().to(DEVICE)
    
    # Loss Setup (Generalized Flat V5)
    loss_fn = MultiResSTFTLoss(device=DEVICE)
    
    # Learnable Global Params (Noise/Reverb)
    # Gain: Init small, e.g. -60dB
    noise_gain_logit = torch.nn.Parameter(torch.logit(torch.tensor(0.001, device=DEVICE)), requires_grad=True)
    noise_color_logit = torch.nn.Parameter(torch.logit(torch.tensor(0.5, device=DEVICE)), requires_grad=True)
    
    # Reverb Params
    reverb_wet_logit = torch.nn.Parameter(torch.logit(torch.tensor(0.01, device=DEVICE)), requires_grad=True)
    reverb_decay_log = torch.nn.Parameter(torch.log(torch.tensor(0.5, device=DEVICE)), requires_grad=True)
    
    dataset = PianoDataset('data/clean_et', device=DEVICE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = optim.Adam(
        list(model.parameters()) + [noise_gain_logit, noise_color_logit, reverb_wet_logit, reverb_decay_log],
        lr=LEARNING_RATE
    )
    
    # 2. Training Loop
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_audio, batch_midi, batch_vel in pbar:
            batch_audio = batch_audio.to(DEVICE)
            batch_midi = batch_midi.to(DEVICE)
            batch_vel = batch_vel.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Global params
            n_gain = torch.sigmoid(noise_gain_logit)
            n_color = torch.sigmoid(noise_color_logit)
            rev_wet = torch.sigmoid(reverb_wet_logit)
            rev_decay = torch.exp(reverb_decay_log)
            
            overrides = model.get_overrides()
            
            # Physics Calculation (Batched)
            # Need to ensure calculate_partials handles batching correctly
            phys_out = calculate_partials(
                midi=batch_midi,
                velocity=batch_vel,
                overrides=overrides,
                n_partials=64,
                device=DEVICE
            )
            
            # Render
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
            
            loss = loss_fn(y_pred, batch_audio)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} Avg Loss: {total_loss / len(dataloader)}")
        
        # Periodic Save
        if (epoch + 1) % 10 == 0:
            save_path = f"experiments/params_epoch_{epoch+1}.pt"
            torch.save({
                'model_state': model.state_dict(),
                'noise_params': {'gain': n_gain, 'color': n_color},
                'reverb_params': {'wet': rev_wet, 'decay': rev_decay}
            }, save_path)
            print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    train()
