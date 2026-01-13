import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
import numpy as np
import scipy.io.wavfile as wavfile
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import os

from pianosynth.wavernn import WaveRNN

def train_wavernn():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Data Setup
    DATA_DIR = Path("data/processed")
    files = glob.glob(str(DATA_DIR / "*.pt"))
    files = [f for f in files if "metadata" not in f]
    print(f"Found {len(files)} files.")
    
    # Parameters
    CTX_LEN = 2000
    SEQ_LEN = 100 
    BATCH_SIZE = 64 # Increased batch size for stability with more data? Or 32. Let's try 64.
    EPOCHS = 5000 # Long training
    LR = 1e-3
    HIDDEN_SIZE = 80 # ~25k params
    
    # Preload Data
    print("Preloading data...")
    all_data = [] # List of (audio_tensor, midi_val, vel_val)
    
    vel_map = {'pp': 0.25, 'mf': 0.5, 'ff': 0.75}
    
    for fpath in tqdm(files):
        try:
            # Parse filename: {midi}_{dyn}.pt
            fname = os.path.basename(fpath)
            parts = fname.replace('.pt', '').split('_')
            midi_val = float(parts[0]) / 128.0 # Normalize 0-1
            dyn_str = parts[1]
            vel_val = vel_map.get(dyn_str, 0.5)
            
            audio = torch.load(fpath, map_location=device).float()
            
            # Filter shorts
            if len(audio) < CTX_LEN + SEQ_LEN + 1:
                continue
                
            all_data.append({
                'audio': audio,
                'midi': midi_val,
                'vel': vel_val,
                'valid_len': len(audio) - CTX_LEN - SEQ_LEN
            })
        except Exception as e:
            print(f"Skipping {fpath}: {e}")
            
    print(f"Loaded {len(all_data)} valid samples.")

    # Dataset
    class PianoDataset(Dataset):
        def __init__(self, data_list, ctx_len, seq_len, steps_per_epoch=1000):
            self.data = data_list
            self.ctx_len = ctx_len
            self.seq_len = seq_len
            self.steps = steps_per_epoch
            
        def __len__(self):
            return self.steps * BATCH_SIZE # Total samples per epoch
            
        def __getitem__(self, idx):
            # 1. Pick random file
            item = random.choice(self.data)
            audio = item['audio']
            
            # 2. Pick random start
            start_idx = random.randint(0, item['valid_len'])
            
            # 3. Extract window
            full_window = audio[start_idx : start_idx + self.ctx_len + self.seq_len]
            
            # Inputs: Unfold
            inputs_seq = full_window.unfold(0, self.ctx_len, 1)[:-1] # (Seq, 2000)
            
            # Targets
            targets = full_window[self.ctx_len:]
            
            # Time
            t_start = (start_idx + self.ctx_len) / 44100.0
            times = torch.arange(self.seq_len, device=audio.device) / 44100.0 + t_start
            
            # Metadata
            velocity = torch.tensor(item['vel'], device=audio.device)
            pitch = torch.tensor(item['midi'], device=audio.device)
            
            return inputs_seq, targets, times, velocity, pitch

    # Increase steps per epoch since we have much more data
    dataset = PianoDataset(all_data, CTX_LEN, SEQ_LEN, steps_per_epoch=200) 
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0) # Workers=0 for GPU tensors in list
    
    # Model
    model = WaveRNN(hidden_size=HIDDEN_SIZE).to(device)
    print(f"Model Parameters: {model.count_parameters()}")
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    def gaussian_nll(y, mu, sigma):
        log_sigma = torch.log(sigma + 1e-7)
        var = sigma**2
        return torch.mean(log_sigma + 0.5 * (y - mu)**2 / var)
    
    # Train
    history = []
    
    # Output dir for full run
    OUT_DIR = Path("results_full_keyboard")
    OUT_DIR.mkdir(exist_ok=True)
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, leave=False)
        for batch in pbar:
            inputs, targets, times, vel, pitch = batch # Already on device if loaded there? 
            # If we keep data on GPU (device), then batch is on GPU.
            # But dataloader might try to re-collate.
            # Let's ensure.
            
            # Adjust shapes
            targets = targets.unsqueeze(2)
            times = times.unsqueeze(2)
            vel = vel.unsqueeze(1)
            pitch = pitch.unsqueeze(1)
            
            optimizer.zero_grad()
            mus, sigmas = model.forward_sequence(inputs, vel, pitch, times)
            loss = gaussian_nll(targets, mus, sigmas)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_description(f"Loss: {loss.item():.4f}")
            
        avg_loss = total_loss / len(dataloader)
        history.append(avg_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} Loss: {avg_loss:.4f}")
        
        # Save & Generate
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), OUT_DIR / f"wavernn_{epoch+1}.pt")
            
            print("Generating preview (C4 mf)...")
            model.eval()
            with torch.no_grad():
                ctx_len = 2000
                gen_buffer = torch.zeros(ctx_len, device=device)
                hidden = None
                
                # C4 mf
                velocity = torch.tensor([[0.5]], device=device)
                log_pitch = torch.tensor([[60.0/128.0]], device=device)
                
                gen_samples = []
                N_GEN = 22050
                
                for i in range(N_GEN):
                    past = gen_buffer[-ctx_len:].unsqueeze(0)
                    time_val = torch.tensor([[i / 44100.0]], device=device)
                    mu, sigma, hidden = model(past, velocity, log_pitch, time_val, hidden)
                    dist = torch.distributions.Normal(mu, sigma)
                    sample = dist.sample()
                    gen_samples.append(sample.item())
                    gen_buffer = torch.cat([gen_buffer, sample.squeeze(0)])
                    
                gen_arr = np.clip(np.array(gen_samples), -1, 1)
                wavfile.write(OUT_DIR / f"epoch_{epoch+1}_c4.wav", 44100, (gen_arr * 32767).astype(np.int16))
            model.train()
            
    # Final
    torch.save(model.state_dict(), OUT_DIR / "wavernn_final.pt")
    plt.plot(history)
    plt.savefig(OUT_DIR / "loss_history.png")

if __name__ == "__main__":
    train_wavernn()
