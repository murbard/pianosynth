import torch
import soundfile as sf
import os
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SR = 44100

def note_str_to_midi(note_str):
    """
    Parses 'Bb0', 'C#4', 'A0', etc.
    """
    # Regex: (Note letter)(optional accidental)(Octave)
    match = re.match(r"^([A-G])(b|#)?([0-8])$", note_str)
    if not match:
        raise ValueError(f"Invalid note string: {note_str}")
    
    letter, acc, octave = match.groups()
    octave = int(octave)
    
    # Base mapping (C=0)
    base_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    pitch = base_map[letter]
    
    if acc == '#':
        pitch += 1
    elif acc == 'b':
        pitch -= 1
        
    midi = (octave + 1) * 12 + pitch
    return midi

def detect_onset(audio: torch.Tensor, threshold_db=-40, backtrack_ms=5):
    """
    Detects the start of the note.
    Simple backtracking algorithm: find peak, go back until threshold.
    """
    # Create envelope (rectified)
    env = audio.abs()
    
    # 1. Find Main Peak (Attack)
    peak_val, peak_idx = env.max(dim=0)
    
    # If silence, return 0
    if peak_val < 1e-4:
        print("Warning: Silence detected.")
        return 0
        
    # 2. Backtrack from peak until energy drops below threshold relative to peak
    # Threshold in linear scale
    thresh_lin = peak_val * (10 ** (threshold_db / 20))
    
    # Scan backwards from peak
    # We look for the first point (going backwards) that is below threshold
    # OR a zero crossing if we want to be super precise, but threshold is usually enough.
    
    # Slicing backwards is tricky in torch, convert to numpy for indexing logic if complex, 
    # but simple loop or searchsorted is fine.
    
    pre_peak = env[:peak_idx]
    
    # Mask of values below threshold
    below_thresh = (pre_peak < thresh_lin)
    
    # Find last index where this is true
    # We want indices where below_thresh is true.
    # We want the *last* one (closest to peak).
    indices = torch.nonzero(below_thresh)
    if len(indices) == 0:
        # Start of file is still loud?
        onset_idx = 0
    else:
        onset_idx = indices[-1].item()
        
    return onset_idx

def detect_offset(audio: torch.Tensor, onset_idx: int, threshold_db=-60, min_dur_sec=0.1):
    """
    Detects where to cut the end (when it fades out).
    """
    if onset_idx >= len(audio):
        return len(audio)
        
    env = audio[onset_idx:].abs()
    peak_val = env.max()
    thresh_lin = peak_val * (10 ** (threshold_db / 20))
    
    # Scan forwards from peak? No, scan backwards from end.
    # Find last point ABOVE threshold.
    
    # Actually, piano notes decay monotonically-ish.
    # Let's smooth the envelope first? 
    # For now, raw envelope is noisy. Let's maximize over windows.
    # But simple approach: find last sample above threshold.
    
    above_thresh = (env > thresh_lin)
    indices = torch.nonzero(above_thresh)
    
    if len(indices) == 0:
        return len(audio)
    
    last_loud_sample = indices[-1].item() + onset_idx
    
    # Add a small buffer (release)
    sr_buffer = int(0.1 * SR)
    offset_idx = min(len(audio), last_loud_sample + sr_buffer)
    
    return offset_idx


def process_file(file_path: Path):
    try:
        # Load audio (sf.read returns numpy)
        data, sr = sf.read(str(file_path))
        
        # Check SR
        if sr != SR:
            # Simple resample if needed, but Iowa is usually 44.1k
            # If not, implement resampling. For now assume 44.1k or fail loudly
            pass
            
        # Convert to torch
        audio = torch.tensor(data, dtype=torch.float32)
        
        # To Mono
        if audio.ndim == 2:
            audio = audio.mean(dim=1)
            
        # 1. Onset
        onset_idx = detect_onset(audio)
        
        # 2. Offset
        offset_idx = detect_offset(audio, onset_idx)
        
        # 3. Trim
        audio_trimmed = audio[onset_idx:offset_idx]
        
        # 4. Normalize (Peak to -1dB = 0.891)
        peak = audio_trimmed.abs().max()
        if peak > 1e-6:
            audio_norm = audio_trimmed / peak * 0.891
        else:
            audio_norm = audio_trimmed
            
        return audio_norm
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Map dynamic strings to rough initial velocity guess (just for metadata/sorting)
    # The actual optimization will LEARN the velocity.
    dyn_map = {'pp': 0.2, 'mf': 0.6, 'ff': 0.9}
    
    files = list(RAW_DIR.rglob("*.aiff"))
    print(f"Found {len(files)} raw files.")
    
    metadata = {}
    
    for f in tqdm(files):
        # f is data/raw/pp/Bb0.aiff
        dynamic = f.parent.name # 'pp'
        note_str = f.stem       # 'Bb0'
        
        try:
            midi = note_str_to_midi(note_str)
        except ValueError:
            print(f"Skipping {f}, invalid note name")
            continue
            
        audio = process_file(f)
        if audio is not None and len(audio) > 1000:
            out_name = f"{midi}_{dynamic}.pt"
            torch.save(audio, PROCESSED_DIR / out_name)
            
            # Store metadata
            if midi not in metadata:
                metadata[midi] = {}
            metadata[midi][dynamic] = out_name
            
    # Save index
    torch.save(metadata, PROCESSED_DIR / "metadata.pt")
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
