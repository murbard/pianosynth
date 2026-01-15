import soundfile as sf
import numpy as np
from pathlib import Path
import torch

RAW_DIR = Path("data/raw")
OUT_DIR = Path("results_diagnostic")
OUT_DIR.mkdir(exist_ok=True)

def load_clip(path, dur_sec=2.0):
    try:
        data, sr = sf.read(str(path))
        if data.ndim > 1: data = data.mean(axis=1)
        
        target_len = int(sr * dur_sec)
        if len(data) > target_len:
            data = data[:target_len]
        else:
            padding = np.zeros(target_len - len(data))
            data = np.concatenate([data, padding])
            
        # Normalize
        peak = np.abs(data).max()
        if peak > 0:
            data = data / peak * 0.8
            
        return data, sr
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return np.zeros(int(44100*dur_sec)), 44100

def main():
    # Candidates for Corruption
    # A0 (21)
    # B0 (23) 
    # E1 (28) - Detected as 580Hz (D5)
    # Controls: A4 (69)
    
    suspects = [
        ("A0", "ff"),
        ("B0", "mf"),
        ("E1", "ff"),
        ("A4", "mf") # Control
    ]
    
    full_audio = []
    
    sr = 44100
    
    for note, dyn in suspects:
        path = RAW_DIR / dyn / f"{note}.aiff"
        print(f"Loading {path}...")
        clip, _ = load_clip(path)
        
        # Add silence spacer
        spacer = np.zeros(int(sr * 0.5))
        
        full_audio.append(clip)
        full_audio.append(spacer)
        
    full_seq = np.concatenate(full_audio)
    
    out_path = OUT_DIR / "diagnosis_corruption.wav"
    sf.write(str(out_path), full_seq, sr)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
