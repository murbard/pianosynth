import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import scipy.io.wavfile as wavfile
import soundfile as sf
from scipy.signal.windows import blackmanharris

RAW_DIR = Path("data/raw")
OLD_PROC_DIR = Path("data/processed")
OUT_DIR = Path("data/clean_et")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Map Note Names to MIDI
NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

def midi_to_name(m):
    octave = (m // 12) - 1
    note = NOTE_NAMES[m % 12]
    return f"{note}{octave}"

def load_raw_or_processed(midi, dyn):
    # Try Raw AIFF first (Best Quality)
    name = midi_to_name(midi)
    raw_path = RAW_DIR / dyn / f"{name}.aiff"
    
    if raw_path.exists():
        wav, sr = sf.read(str(raw_path))
        wav = torch.tensor(wav, dtype=torch.float32)
        if wav.ndim > 1: wav = wav.mean(dim=1)
        
        # Resample to 44100
        if sr != 44100:
            wav_5d = wav.view(1, 1, -1)
            scale = 44100 / sr
            new_len = int(wav.shape[0] * scale)
            wav = F.interpolate(wav_5d, size=new_len, mode='linear', align_corners=False).view(-1)
            
        return wav, "raw"
    
    # Fallback to Old Processed
    proc_path = OLD_PROC_DIR / f"{midi}_{dyn}.pt"
    if proc_path.exists():
        wav = torch.load(proc_path)
        if wav.dim() > 1: wav = wav.view(-1)
        return wav.float(), "processed"
        
    return None, None

def measure_robust(audio, sr=44100):
    if torch.max(torch.abs(audio)) < 1e-4: return 0.0
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

def measure_yin(data, sr=44100, f_min=20, f_max=4200):
    n = len(data)
    w_size = n // 2
    if w_size < 100: return 0.0
    
    spectrum = np.fft.rfft(data, n=2*n)
    r = np.fft.irfft(spectrum * np.conj(spectrum))[:w_size]
    
    tau_min = int(sr / f_max)
    tau_max = int(sr / f_min)
    tau_max = min(tau_max, w_size - 1)
    if tau_min < 1: tau_min = 1
    
    diff = 2 * (r[0] - r)
    candidates = diff[tau_min:tau_max]
    if len(candidates) == 0: return 0.0
    
    tau_idx = np.argmin(candidates) + tau_min
    
    if 0 < tau_idx < len(diff)-1:
        y1 = diff[tau_idx-1]; y2 = diff[tau_idx]; y3 = diff[tau_idx+1]
        denom = (y1 - 2*y2 + y3)
        if denom != 0:
            tau_refined = tau_idx - 0.5 * (y1 - y3) / denom
        else:
            tau_refined = tau_idx
    else:
        tau_refined = tau_idx
        
    return sr / tau_refined

def measure_fft(data, sr=44100):
    n = len(data)
    n_fft = 65536
    while n_fft < n: n_fft *= 2
    w = blackmanharris(n)
    padded = np.zeros(n_fft)
    padded[:n] = data * w
    spec = np.abs(np.fft.rfft(padded))
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    idx = np.argmax(spec)
    return freqs[idx]

def measure_consensus(audio, sr=44100):
    if torch.max(torch.abs(audio)) < 1e-4: return 0.0
    data = audio.numpy()
    
    try: f_hps = measure_robust(audio, sr)
    except: f_hps = 0.0
    try: f_yin = measure_yin(data, sr)
    except: f_yin = 0.0
    try: f_fft = measure_fft(data, sr)
    except: f_fft = 0.0
    
    estimates = [f for f in [f_hps, f_yin, f_fft] if f > 15 and f < 5000]
    if not estimates: return 0.0
    
    # Consistency check
    if f_yin > 0 and f_hps > 0:
        ratio = f_yin / f_hps
        if 0.96 < ratio < 1.04: return (f_yin + f_hps) / 2.0
        if 1.9 < ratio < 2.1: return f_hps 
        if 0.45 < ratio < 0.55: return f_yin 
            
    if f_hps > 0: return f_hps
    if f_yin > 0: return f_yin
    return f_fft

def measure_harmonic_series(audio, f_expected, sr=44100):
    n = len(audio)
    n_fft = 65536
    
    # Take a chunk if too long
    if n > n_fft:
        chunk = audio[10000:10000+n_fft]
    else:
        chunk = audio
        n_fft = n
        
    w = torch.hann_window(len(chunk))
    spec = torch.fft.rfft(chunk * w)
    mag = torch.abs(spec).numpy()
    
    # Scan small range around expected
    # +/- 1 semitone range
    f_min = f_expected * (2.0 ** (-1.0/12.0))
    f_max = f_expected * (2.0 ** (1.0/12.0))
    scan_range = np.linspace(f_min, f_max, 50)
    
    best_score = -1
    best_f = 0.0
    
    freq_resolution = sr / n_fft
    
    for f_cand in scan_range:
        score = 0
        valid_harmonics = 0
        for k in range(1, 16):
            target_f = k * f_cand
            idx = int(target_f / freq_resolution)
            
            if idx+2 < len(mag):
                # Sum energy around expected harmonic bin
                val = np.sum(mag[idx-2:idx+3])
                score += val 
                valid_harmonics += 1
        
        if score > best_score:
            best_score = score
            best_f = f_cand
            
    return best_f

def main():
    device = "cpu"
    SR = 44100
    
    valid_data = {}
    print("Loading valid files...")
    for m in tqdm(range(21, 109)):
        for d in ['pp', 'mf', 'ff']:
            wav, src = load_raw_or_processed(m, d)
            if wav is not None and torch.max(torch.abs(wav)) > 1e-3:
                valid_data[(m, d)] = wav
            elif wav is not None:
                print(f"Skipping {m} {d}: Silent")

    print(f"Found {len(valid_data)} / {264} valid files.")

    for m in tqdm(range(21, 109)):
        for d in ['pp', 'mf', 'ff']:
            # Fetch / Patch
            if (m, d) in valid_data:
                wav = valid_data[(m, d)]
                origin = "original"
            else:
                # Neighbor Search
                best_n = None
                min_dist = 999
                for (sm, sd) in valid_data.keys():
                    if sd == d:
                        dist = abs(sm - m)
                        if dist < min_dist:
                            min_dist = dist
                            best_n = (sm, sd)
                
                if best_n is None:
                    print(f"CRITICAL: Missing {d} dynamic entirely.")
                    continue
                
                wav = valid_data[best_n]
                origin = f"patched_{best_n[0]}"

            # Tuning Details
            src_m = m if origin == "original" else int(origin.split("_")[-1])
            f_expected = 440.0 * (2.0 ** ((src_m - 69.0) / 12.0))
            f_target_base = 440.0 * (2.0 ** ((m - 69.0) / 12.0))

            f_meas = measure_consensus(wav, sr=SR)
            
            f_reference = f_expected
            octave_diff = 0
            
            # HUM REJECTION (50Hz / 60Hz)
            if (55.0 < f_meas < 65.0 or 45.0 < f_meas < 55.0) and f_expected > 90.0:
                 f_meas = 0.0

            if f_meas > 15.0:
                ratio_raw = f_meas / f_expected
                octave_diff = round(np.log2(ratio_raw))
                f_exp_harmonic = f_expected * (2.0 ** octave_diff)
                semitone_dev = 12.0 * np.log2(f_meas / f_exp_harmonic)
                
                # DIAGNOSTIC CHECK: "Is it A4?"
                if abs(semitone_dev) > 1.0:
                    # Generic measurement failed. Try Harmonic Series match.
                    f_harmonic = measure_harmonic_series(wav, f_expected, sr=SR)
                    
                    ratio_h = f_harmonic / f_expected
                    semitone_dev_h = 12.0 * np.log2(ratio_h)
                    
                    if abs(semitone_dev_h) < 1.0:
                        # SUCCESS! Harmonic Series found the pitch.
                        # print(f"Recovered {midi_to_name(m)} {d}: Meas {f_meas:.1f}Hz -> Harm {f_harmonic:.1f}Hz (Dev {semitone_dev_h:.2f}st)")
                        f_reference = f_harmonic
                        f_target_aligned = f_target_base * (2.0 ** octave_diff) # Wait octave_diff might be from bad f_meas
                        # If we used harmonic series around f_expected, octave_diff is 0.
                        f_target_aligned = f_target_base
                    else:
                        # Still failed.
                        print(f"WARNING: {m} {d} (Src {src_m}) | Meas {f_meas:.1f}Hz | Harm {f_harmonic:.1f}Hz | Exp {f_expected:.1f}Hz")
                        print(" -> Detection Failed. TRUSTING LABEL.")
                        f_reference = f_expected
                        f_target_aligned = f_target_base
                else:
                    # Good match (Generic)
                    f_reference = f_meas
                    f_target_aligned = f_target_base * (2.0 ** octave_diff)
            else:
                # Measurement failed. Try Harmonic Series.
                f_harmonic = measure_harmonic_series(wav, f_expected, sr=SR)
                ratio_h = f_harmonic / f_expected
                semitone_dev_h = 12.0 * np.log2(ratio_h)
                
                if abs(semitone_dev_h) < 1.0:
                    f_reference = f_harmonic
                    f_target_aligned = f_target_base
                else:
                    f_reference = f_expected
                    f_target_aligned = f_target_base


            # Ratio
            if f_reference > 0:
                ratio = f_target_aligned / f_reference
            else:
                ratio = 1.0
            
            # Safeguard
            if ratio < 0.25 or ratio > 4.0:
                ratio = 1.0

            # Resample
            if abs(ratio - 1.0) > 0.0005:
                wav_5d = wav.view(1, 1, -1)
                new_len = int(wav.shape[0] / ratio)
                wav_tuned = F.interpolate(wav_5d, size=new_len, mode='linear', align_corners=False).view(-1)
            else:
                wav_tuned = wav

            # Trim/Pad (6s)
            TARGET_LEN = 44100 * 6
            if len(wav_tuned) > TARGET_LEN:
                wav_tuned = wav_tuned[:TARGET_LEN]
            else:
                wav_tuned = F.pad(wav_tuned, (0, TARGET_LEN - len(wav_tuned)))

            # Save
            torch.save(wav_tuned, OUT_DIR / f"{m}_{d}.pt")

    print("Reprocessing Complete.")

if __name__ == "__main__":
    main()
