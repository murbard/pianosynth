import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from pianosynth.optimization import PianoParam

PARAM_PATH = Path("src/pianosynth/params_optimized.pt")
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("comparisons")
SR = 44100

def synthesize_params(freqs, decays, amps, dur_sec=3.0, sr=44100):
    """
    Synthesize audio from lists of partial parameters.
    """
    device = freqs.device
    N = freqs.shape[0]
    T = int(dur_sec * sr)
    t = torch.arange(T, device=device) / sr
    
    y = torch.zeros(T, device=device)
    
    # We can vectorize over N
    # shape: [N, T]
    # f: [N, 1]
    # d: [N, 1]
    # a: [N, 1]
    
    f = freqs.unsqueeze(1)
    d = decays.unsqueeze(1)
    a = amps.unsqueeze(1)
    
    # envelope = exp(-t/d)
    # sin(2pi f t)
    
    # Some d might be 100.0 (dummy).
    # Some f might be 0.0.
    
    # Phase randomization per partial?
    phi = torch.rand(N, 1, device=device) * 2 * torch.pi
    
    signal = a * torch.exp(-t.unsqueeze(0) / d) * torch.sin(2 * torch.pi * f * t.unsqueeze(0) + phi)
    
    y = signal.sum(dim=0)
    
    # Normalize? 
    # Real samples are normalized to -1dB.
    # We should match that for comparison, or normalize both.
    
    return y.cpu()

def main():
    if not PARAM_PATH.exists():
        print("Params not found.")
        return
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading optimized params...")
    checkpoint = torch.load(PARAM_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = PianoParam(device=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # Metadata
    midis = checkpoint['midi_indices']
    dyns = checkpoint['dynamics']
    velocities = checkpoint['velocities'] # saved as cpu tensor
    
    # Select a few notes to verify
    # Low, Mid, High
    # pp, ff
    
    test_cases = [
        (40, 'mf'), (40, 'ff'),  # E2
        (60, 'pp'), (60, 'ff'),  # C4
        (80, 'mf'), (80, 'ff')   # G#5
    ]
    
    data_lookup = {}
    for i, (m, d) in enumerate(zip(midis, dyns)):
        if m not in data_lookup: data_lookup[m] = {}
        data_lookup[m][d] = i
        
    for m_target, d_target in test_cases:
        if m_target in data_lookup and d_target in data_lookup[m_target]:
            idx = data_lookup[m_target][d_target]
            vel = velocities[idx].to(device)
            print(f"Synthesizing MIDI {m_target} {d_target} (Learned v={vel.item():.2f})...")
            
            # Predict
            m_t = torch.tensor([float(m_target)], device=device)
            v_t = vel.unsqueeze(0)
            
            with torch.no_grad():
                pf, pd, pa = model(m_t, v_t)
                
            y_synth = synthesize_params(pf[0], pd[0], pa[0], dur_sec=3.0)
            
            # Normalize synth
            y_synth = y_synth / (y_synth.abs().max() + 1e-6) * 0.9
            
            # Load Original
            # Need filename from metadata?
            # Or just check processed dir
            # metadata.pt has filename
            meta = torch.load(PROCESSED_DIR / "metadata.pt")
            fname = meta[m_target][d_target] # e.g. 60_ff.pt
            
            y_real = torch.load(PROCESSED_DIR / fname)
            # Crop real to 3s if longer
            if len(y_real) > len(y_synth):
                y_real = y_real[:len(y_synth)]
            else:
                # pad real
                pad = torch.zeros(len(y_synth) - len(y_real))
                y_real = torch.cat([y_real, pad])
                
            # Normalize real
            y_real = y_real / (y_real.abs().max() + 1e-6) * 0.9
            
            # Combine Stereo
            stereo = torch.stack([y_synth, y_real], dim=1).numpy()
            
            out_file = OUTPUT_DIR / f"cmp_{m_target}_{d_target}.wav"
            sf.write(str(out_file), stereo, SR)
            print(f"Saved {out_file} (L=Model, R=Real)")
            
    print("Verification complete.")

if __name__ == "__main__":
    main()
