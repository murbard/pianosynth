import torch
import numpy as np
import scipy.signal
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt

def load_audio(midi, dyn):
    # Try Raw
    path = Path(f"data/raw/{dyn}")
    # Find file starting with name
    names = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    octave = (midi // 12) - 1
    # note = names[midi % 12] # This was missing
    # Actually name logic:
    # Midi 21 is A0.
    # 21 % 12 = 9 (A). (21//12)-1 = 0.
    name = f"{names[midi % 12]}{octave}"
    raw_file = path / f"{name}.aiff"
    if raw_file.exists():
        print(f"Loading Raw: {raw_file}")
        wav, sr = sf.read(str(raw_file))
        if wav.ndim > 1: wav = wav.mean(axis=1)
        return torch.tensor(wav).float(), sr
    
    proc_file = Path(f"data/processed/{midi}_{dyn}.pt")
    if proc_file.exists():
        print(f"Loading Processed: {proc_file}")
        wav = torch.load(proc_file)
        if wav.dim() > 1: wav = wav.view(-1)
        return wav.float(), 44100

    return None, None

def analyze(midi=21, dyn='pp'):
    audio, sr = load_audio(midi, dyn)
    if audio is None:
        print("File not found")
        return

    n = len(audio)
    n_fft = 65536
    w = torch.hann_window(n_fft)
    
    # Take a chunk
    if len(audio) > n_fft:
        chunk = audio[10000:10000+n_fft]
    else:
        chunk = audio
        n_fft = len(chunk)

    chunk = chunk * torch.hann_window(len(chunk))
    spec = torch.fft.rfft(chunk)
    mag = torch.abs(spec).numpy()
    freqs = torch.fft.rfftfreq(len(chunk), 1/sr).numpy()
    
    f_expected = 440.0 * (2.0 ** ((midi - 69)/12))
    print(f"Expected: {f_expected:.2f} Hz")
    
    # Peak picking
    peaks, _ = scipy.signal.find_peaks(mag, height=mag.max()*0.05, distance=5)
    print("\nTop Peaks:")
    for p in peaks[:20]:
        f = freqs[p]
        if f < 500:
            ratio = f / f_expected
            print(f"  {f:.2f} Hz (Ratio: {ratio:.2f}) Mag: {mag[p]:.2f}")

    # Harmonic Matcher
    print("\nHarmonic Series Score:")
    
    best_score = -1
    best_fwhm = 0
    
    # Scan small range around expected
    scan_range = np.linspace(f_expected*0.9, f_expected*1.1, 100)
    for f_cand in scan_range:
        score = 0
        for k in range(1, 16):
            # Check energy at k * f_cand
            target = k * f_cand
            idx = int(target / (sr/n_fft))
            if idx < len(mag):
                # Sum 3 bins
                val = np.sum(mag[idx-2:idx+3])
                score += val # * (1.0 / k) # Decay?
        
        if score > best_score:
            best_score = score
            best_f = f_cand
            
    print(f"Best Harmonic Fit: {best_f:.2f} Hz")
    semitone_diff = 12 * np.log2(best_f / f_expected)
    print(f"Deviation: {semitone_diff:.2f} semitones")

if __name__ == "__main__":
    analyze(21, 'pp')
    analyze(21, 'mf')
    analyze(22, 'pp')
