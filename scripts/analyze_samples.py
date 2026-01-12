import torch
import torch.fft
from pathlib import Path
from tqdm import tqdm
import math

PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "analysis_data.pt"
SR = 44100
N_PARTIALS = 64
FFT_SIZE = 4096  # High freq resolution
HOP_SIZE = 256

def analyze_note(audio: torch.Tensor, midi: int):
    """
    Extracts partials from a single note recording.
    Returns:
        freqs: [N]
        decays: [N] (slope of log envelop)
        amps: [N] (initial amplitude)
    """
    # 1. Expected F0
    f0_nominal = 440.0 * (2 ** ((midi - 69) / 12))
    
    # 2. STFT
    # shape: [Freq, Time]
    window = torch.hann_window(FFT_SIZE).to(audio.device)
    stft = torch.stft(audio, n_fft=FFT_SIZE, hop_length=HOP_SIZE, window=window, return_complex=True)
    mag = stft.abs()
    
    freq_bins = torch.fft.rfftfreq(FFT_SIZE, d=1/SR).to(audio.device)
    bin_hz = SR / FFT_SIZE
    
    found_freqs = []
    found_decays = []
    found_amps = []
    
    # Analyze first N partials
    for n in range(1, N_PARTIALS + 1):
        # Target freq
        # Initial guess with slight stiffness B=0.0001?
        # f_target = n * f0_nominal * sqrt(1 + B * n^2)
        # Just search widely around harmonic
        center_f = n * f0_nominal
        width_f = max(50.0, 0.05 * center_f) # window to search peak
        
        # Convert to bins
        idx_min = max(0, int((center_f - width_f) / bin_hz))
        idx_max = int((center_f + width_f) / bin_hz)
        
        if idx_max >= mag.shape[0]:
            break

        if idx_min >= idx_max:
             found_freqs.append(0.0)
             found_decays.append(100.0)
             found_amps.append(0.0)
             continue
            
        # Extract band
        band = mag[idx_min:idx_max, :] # [F_band, Time]
        
        # Find peak frequency (average over early time to catch attack)
        # Sum over first 0.5 sec?
        t_frames = int(0.5 * SR / HOP_SIZE)
        if t_frames > band.shape[1]: t_frames = band.shape[1]
        
        avg_spec = band[:, :t_frames].mean(dim=1)
        peak_idx_local = avg_spec.argmax()
        peak_val = avg_spec[peak_idx_local]
        
        # Parabolic interpolation for precise bin
        # (omitted simple version: take bin center)
        true_bin = idx_min + peak_idx_local.item()
        true_freq = true_bin * bin_hz
        
        # Extract Envelope (magnitude over time at that bin)
        # track peak bin just in case it wobbles? No, fixed bin usually fine for piano.
        # Maybe summing 3 bins?
        env = mag[true_bin-1:true_bin+2, :].sum(dim=0)
        
        # Fit Decay
        # Find active region: start (peak) to end (noise floor)
        peak_t = env.argmax()
        max_val = env[peak_t]
        
        if max_val < 1e-4: # Too quiet to analyze
            found_freqs.append(0.0)
            found_decays.append(100.0) # Fast decay dummy
            found_amps.append(0.0)
            continue
            
        # Fit line to log(env)
        # from peak_t to where it drops by 30dB?
        thresh = max_val * 0.03 # -30dB
        
        mask = (env[peak_t:] > thresh)
        # end_t is where mask becomes false
        # simple end finding
        indices = torch.nonzero(~mask)
        if len(indices) > 0:
            end_t = indices[0].item() + peak_t
        else:
            end_t = len(env)
            
        if end_t - peak_t < 5: # Too short
            found_freqs.append(true_freq)
            found_decays.append(0.1)
            found_amps.append(max_val.item())
            continue
            
        # Linear regression on log(y) = -t/tau + C
        # y = A * exp(-t/tau) -> ln(y) = ln(A) - (1/tau)*t
        
        region = env[peak_t:end_t]
        log_y = torch.log(region + 1e-9)
        t = torch.arange(len(region)).to(audio.device) * (HOP_SIZE / SR)
        
        # slope m = (N sum(xy) - sum(x)sum(y)) / (N sum(x^2) - (sum(x))^2)
        # or use lstsq
        A_mat = torch.stack([t, torch.ones_like(t)], dim=1)
        # X = [m, c]
        sol = torch.linalg.lstsq(A_mat, log_y).solution
        slope = sol[0].item() # -1/tau
        intercept = sol[1].item() # ln(A)
        
        tau = -1.0 / (slope - 1e-6)
        amp = math.exp(intercept)
        
        found_freqs.append(true_freq)
        found_decays.append(tau)
        found_amps.append(amp)
        
    # Pad to N_PARTIALS
    while len(found_freqs) < N_PARTIALS:
        found_freqs.append(0.0)
        found_decays.append(100.0)
        found_amps.append(0.0)
        
    return (torch.tensor(found_freqs), 
            torch.tensor(found_decays), 
            torch.tensor(found_amps))

def main():
    if not PROCESSED_DIR.exists():
        print("Processed data not found.")
        return
        
    # Load index
    try:
        metadata = torch.load(PROCESSED_DIR / "metadata.pt")
    except:
        print("No metadata.pt found.")
        return
        
    results = {} # params[midi][dynamic] = {freqs, decays, amps}
    
    all_files = []
    for midi, dynamics in metadata.items():
        for dyn, fname in dynamics.items():
            all_files.append((midi, dyn, fname))
            
    print(f"Analyzing {len(all_files)} files...")
    
    for midi, dyn, fname in tqdm(all_files):
        path = PROCESSED_DIR / fname
        if not path.exists(): continue
        
        audio = torch.load(path)
        
        # Analyze
        freqs, decays, amps = analyze_note(audio, midi)
        
        if midi not in results: results[midi] = {}
        results[midi][dyn] = {
            "freqs": freqs,
            "decays": decays,
            "amps": amps
        }
        
    torch.save(results, OUTPUT_PATH)
    print(f"Analysis complete. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
