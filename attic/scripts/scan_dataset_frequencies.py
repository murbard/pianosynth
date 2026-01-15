import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

RAW_DIR = Path("data/raw")

def measure(path):
    try:
        data, sr = sf.read(str(path))
        if data.ndim > 1: data = data.mean(axis=1)
        
        # 1 sec window?
        # Low notes need long window.
        if len(data) > sr * 3:
             sl = data[int(0.2*sr):int(1.5*sr)]
        else:
             sl = data
             
        spec = np.abs(np.fft.rfft(sl))
        freqs = np.fft.rfftfreq(len(sl), 1/sr)
        
        idx = np.argmax(spec)
        peak = freqs[idx]
        
        # Refined peak?
        # Parabolic
        if 0 < idx < len(spec)-1:
             alpha = spec[idx-1]
             beta = spec[idx]
             gamma = spec[idx+1]
             p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
             peak = (idx + p) * (sr / len(sl))
             
        return peak
    except Exception as e:
        return 0.0

def main():
    files = list(RAW_DIR.rglob("*.aiff"))
    print(f"Scanning {len(files)} files...")
    
    results = []
    
    for f in tqdm(files):
        freq = measure(f)
        if freq > 0:
            midi = 69 + 12 * np.log2(freq/440.0)
        else:
            midi = 0
            
        results.append({
            "path": f,
            "freq": freq,
            "midi": midi,
            "label": f.stem
        })
        
    # Sort by Measured MIDI
    results.sort(key=lambda x: x["midi"])
    
    print("\n--- Low Note Candidates (MIDI < 30) ---")
    found_low = [r for r in results if 20 < r["midi"] < 35]
    for r in found_low:
        print(f"Est: {r['midi']:.1f} ({r['freq']:.1f} Hz) - File: {r['path'].parent.name}/{r['label']}")
        
    print("\n--- Summary ---")
    print(f"Total Files: {len(results)}")
    print(f"Valid Frequency Files: {len([r for r in results if r['freq'] > 20])}")
    print(f"Lowest Detected: {results[0]['midi']:.1f} ({results[0]['path'].name})")
    print(f"Highest Detected: {results[-1]['midi']:.1f}")

    # Check for gaps
    detected_midis = np.array([r["midi"] for r in results if r["freq"] > 20])
    if len(detected_midis) > 0:
        import matplotlib.pyplot as plt
        # Histogram?
        # Just check coverage
        bins = np.arange(21, 109)
        hist, _ = np.histogram(detected_midis, bins=bins)
        missing_bins = bins[:-1][hist == 0]
        print(f"Missing MIDI Bins (No file found near this pitch): {missing_bins}")
        
    # Save Map
    torch.save(results, "raw_file_map.pt")

if __name__ == "__main__":
    main()
