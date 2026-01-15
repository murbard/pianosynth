import torch
import numpy as np
from pathlib import Path
import soundfile as sf
from scipy.signal import find_peaks

PROCESSED_DATA_DIR = Path("data/processed")

def get_pitch(m, d):
    # Try direct then fallback
    f_path = PROCESSED_DATA_DIR / f"{m}_{d}.pt"
    used_fallback = False
    if not f_path.exists():
        for alt in ['ff', 'mf', 'pp']:
            p = PROCESSED_DATA_DIR / f"{m}_{alt}.pt"
            if p.exists():
                f_path = p
                used_fallback = True
                break
    
    if not f_path.exists():
        print(f"MIDI {m} {d}: Not Found")
        return

    audio = torch.load(f_path).float().numpy()
    
    # Simple Autocorrelation or FFT
    # Use FFT
    n = len(audio)
    # Zero pad for resolution
    target_len = 44100 * 5
    if n < target_len:
        audio = np.pad(audio, (0, target_len - n))
        
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1/44100)
    
    peak_idx = np.argmax(spec)
    peak_freq = freqs[peak_idx]
    
    # Convert to MIDI
    # f = 440 * 2^((m-69)/12)
    # m = 69 + 12 * log2(f/440)
    
    est_midi = 69 + 12 * np.log2(peak_freq / 440.0)
    
    print(f"MIDI {m} [{d}] {'(Fallback)' if used_fallback else ''} -> F0: {peak_freq:.2f} Hz -> Est MIDI: {est_midi:.2f}")

def check_raw(name, dyn):
    f_path = PROCESSED_DATA_DIR.parent / "raw" / dyn / f"{name}.aiff"
    if not f_path.exists():
        print(f"RAW {name} {dyn}: Missing")
        return

    data, sr = sf.read(str(f_path))
    if data.ndim > 1: data = data.mean(axis=1) # Mono
    
    # FFT
    # Take 1 sec
    sl = data[:sr]
    spec = np.abs(np.fft.rfft(sl))
    freqs = np.fft.rfftfreq(len(sl), 1/sr)
    
    idx = np.argmax(spec)
    peak = freqs[idx]
    
    # Basic Midi est
    if peak > 0:
        est_m = 69 + 12 * np.log2(peak/440.0)
    else:
        est_m = 0
        
    print(f"RAW {name} {dyn} -> {peak:.1f} Hz (Est MIDI {est_m:.1f})")

def main():
    print("Checking Raw Files Validity...")
    notes = ["C4", "A4"]
    for n in notes:
        check_raw(n, "mf")
        check_raw(n, "ff")

if __name__ == "__main__":
    main()
