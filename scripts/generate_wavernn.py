import torch
import scipy.io.wavfile as wavfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from pianosynth.wavernn import WaveRNN

def generate_wavernn():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    MODEL_PATH = "results_single_note/wavernn_final.pt"
    if not Path(MODEL_PATH).exists():
        print(f"Model not found at {MODEL_PATH}")
        # Try checkpoint?
        MODEL_PATH = "results_single_note/wavernn_100.pt"
        if not Path(MODEL_PATH).exists():
            print("No model found.")
            return

    model = WaveRNN(hidden_size=32).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # Generate 1 second of audio
    SR = 44100
    DURATION = 1.0
    N_SAMPLES = int(SR * DURATION)
    
    # Init buffer
    ctx_len = 2000
    buffer = torch.zeros(ctx_len, device=device)
    
    # Hidden State
    hidden = torch.zeros(1, 32, device=device) # (B, Hidden) ... wait forward expects (B, Hidden)
    hidden = torch.zeros(1, 32, device=device) 
    
    generated = []
    
    # Metadata
    velocity = torch.tensor([0.5], device=device)
    log_pitch = torch.tensor([60.0/128.0], device=device)
    
    print("Generating...")
    with torch.no_grad():
        for i in tqdm(range(N_SAMPLES)):
            # Past samples: buffer[-2000:]
            past = buffer[-ctx_len:].unsqueeze(0) # (1, 2000)
            
            time_val = torch.tensor([i / SR], device=device).unsqueeze(0) # (1, 1)
            
            mu, sigma, hidden = model(past, velocity, log_pitch, time_val, hidden)
            
            # Sample
            dist = torch.distributions.Normal(mu, sigma)
            sample = dist.sample()
            
            # Append
            generated.append(sample.item())
            buffer = torch.cat([buffer, sample.squeeze(0)])
            
            # Keep buffer mostly small? No, we need it to grow for generation but only need last 2000 for input.
            # Torch cat is slow. Use fixed buffer? 
            # For 1 sec (44k), cat is fine-ish.
            
    # Save
    gen_audio = np.array(generated)
    # Clip -1..1
    gen_audio = np.clip(gen_audio, -1, 1)
    
    wavfile.write("results_single_note/wavernn_gen.wav", SR, (gen_audio * 32767).astype(np.int16))
    print("Saved to results_single_note/wavernn_gen.wav")
    
    # Plot 
    plt.figure(figsize=(10, 4))
    plt.plot(gen_audio[:1000])
    plt.title("First 1000 samples")
    plt.savefig("results_single_note/wavernn_gen_plot.png")

if __name__ == "__main__":
    generate_wavernn()
