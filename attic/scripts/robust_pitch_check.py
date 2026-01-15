import torch
import numpy as np
import soundfile as sf
from pathlib import Path

RAW_DIR = Path("data/raw")

def measure_robust(path, window_size=65536):
    try:
        data, sr = sf.read(str(path))
        if data.ndim > 1: data = data.mean(axis=1)
        
        # Check non-silence
        if np.max(np.abs(data)) < 1e-4:
            return 0.0, "SILENCE"

        # Take a slice from the sustain part (e.g. 0.5s to 3.0s)
        # Avoid attack transient for pitch
        start = int(0.5 * sr)
        end = int(3.0 * sr)
        if len(data) > end:
            sl = data[start:end]
        elif len(data) > start + 2048:
            sl = data[start:]
        else:
            sl = data
            
        n = len(sl)
        # Pad to next power of 2 for FFT speed, or just use large N
        n_fft = 1
        while n_fft < n or n_fft < window_size:
            n_fft *= 2
            
        # Blackman-Harris Window
        # Only window the actual data length
        w = np.blackman(n) # numpy doesn't have blackman-harris directly? 
        # Actually it does: blackman is close. blackmanharris is in scipy.signal
        from scipy.signal.windows import blackmanharris
        w = blackmanharris(n)
        
        # Zero pad
        padded = np.zeros(n_fft)
        padded[:n] = sl * w
        
        # FFT
        spec = np.abs(np.fft.rfft(padded))
        freqs = np.fft.rfftfreq(n_fft, 1/sr)
        
        # HPS (Harmonic Product Spectrum) - 3 harmonics
        # Downsample spectrum
        hps = spec.copy()
        for h in range(2, 4):
            # Decimate
            dec = spec[::h]
            hps[:len(dec)] *= dec
            
        # Find peak in HPS (focus on 20Hz - 4200Hz)
        valid_mask = (freqs > 20) & (freqs < 4200)
        
        # Apply mask to HPS (be careful with indices)
        # Just zero out invalid
        # But HPS length matches spec length (n_fft/2 + 1)
        
        # Only look at valid bins
        min_bin = int(20 * n_fft / sr)
        max_bin = int(4200 * n_fft / sr)
        
        peak_idx = np.argmax(hps[min_bin:max_bin]) + min_bin
        peak_freq = freqs[peak_idx]
        
        # Refine?
        # Parabolic on original spectrum at fundamental?
        # Or just trust HPS. Resolution of 65536 at 44.1k is ~0.67Hz.
        # At 27Hz (A0), 0.67Hz error is 40 cents.
        # 4*65536 = 262k -> 0.16Hz.
        
        return peak_freq, "OK"
        
    except Exception as e:
        return 0.0, f"ERROR {e}"

def main():
    print("Robust Pitch Check (Blackman-Harris + HPS)...")
    
    suspects = [
        ("A0", "ff"), # Expect Silent
        ("B0", "mf"), # Expect ~30Hz (B0)
        ("E1", "ff"), # Expect ~41Hz (E1) or ~20Hz (E0)?
        ("A4", "mf")  # Control 440
    ]
    
    for note, dyn in suspects:
        p = RAW_DIR / dyn / f"{note}.aiff"
        if not p.exists():
            print(f"{note} {dyn}: Missing")
            continue
            
        f, status = measure_robust(p, window_size=131072*2) # High Res
        if f > 0:
            midi = 69 + 12 * np.log2(f/440.0)
            print(f"{note} {dyn} -> {f:.2f} Hz (MIDI {midi:.2f}) [{status}]")
        else:
            print(f"{note} {dyn} -> {f:.2f} Hz [{status}]")

if __name__ == "__main__":
    main()
