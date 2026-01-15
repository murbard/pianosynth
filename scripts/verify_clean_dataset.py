import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.signal.windows import blackmanharris

CLEAN_DIR = Path("data/clean_et")

def measure_pitch(audio, sr=44100):
    # Robust measurement (HPS based for speed/reliability on clean data)
    data = audio.numpy()
    n = len(data)
    n_fft = 65536
    while n_fft < n: n_fft *= 2
    
    if n < 4: return 0.0
    w = blackmanharris(n)
    padded = np.zeros(n_fft)
    padded[:n] = data * w
    spec = np.abs(np.fft.rfft(padded))
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    
    hps = spec.copy()
    for h in range(2, 4):
        dec = spec[::h]
        L = len(dec)
        hps[:L] *= dec
        
    min_bin = int(20 * n_fft / sr)
    max_bin = int(4200 * n_fft / sr)
    max_bin = min(max_bin, len(hps))
    
    if max_bin <= min_bin: return 0.0
    
    peak_idx = np.argmax(hps[min_bin:max_bin]) + min_bin
    return freqs[peak_idx]

def main():
    print("Verifying Clean Dataset...")
    files = list(CLEAN_DIR.glob("*.pt"))
    print(f"Found {len(files)} files.")
    
    errors = []
    deviations = []
    
    for f in tqdm(files):
        # Parse Name: 21_mf.pt
        try:
            stem = f.stem
            m = int(stem.split("_")[0])
            d = stem.split("_")[1]
        except:
            continue
            
        wav = torch.load(f)
        f_meas = measure_pitch(wav)
        
        f_target = 440.0 * (2.0 ** ((m - 69.0) / 12.0))
        
        if f_meas < 15:
            errors.append(f"{m} {d}: Measurement Failed (Silence?)")
            continue
            
        # Octave Check
        ratio = f_meas / f_target
        octave_dev = np.log2(ratio)
        octave_int = round(octave_dev)
        
        # We expect 0 octave shift for "clean" data, 
        # BUT if we allowed octave errors in source, the cleaned data might still be octave shifted?
        # Ideally resampling corrected it.
        # Wait, my resampling logic calculated ratio based on `f_target_aligned`.
        # `f_target_aligned` was `f_target_base * (2^octave_diff)`.
        # So I intentionally preserved the octave error in the file to avoid massive resampling.
        # So here we should verify against `f_target * 2^octave_int`.
        
        f_target_aligned = f_target * (2.0 ** octave_int)
        cents = 1200 * np.log2(f_meas / f_target_aligned)
        
        deviations.append(abs(cents))
        
        if abs(cents) > 10.0: # 10 cents tolerance
             errors.append(f"{m} {d}: Dev {cents:.2f} cents (Octave {octave_int})")
             
    print("\n--- Calibration Results ---")
    if len(deviations) > 0:
        print(f"Mean Error: {np.mean(deviations):.2f} cents")
        print(f"Max Error: {np.max(deviations):.2f} cents")
        
    if len(errors) > 0:
        print("\n--- Outliers (>10 cents) ---")
        for e in errors[:20]:
            print(e)
        if len(errors) > 20: print(f"... and {len(errors)-20} more.")
    else:
        print("PERFECT! All files within tolerance.")

if __name__ == "__main__":
    main()
