
import soundfile as sf
import torch
import numpy as np
import scipy.io.wavfile

def resynthesize_c4():
    print("Loading Iowa Piano C4 sample...")
    # Load audio
    try:
        audio, sr = sf.read("Piano.ff.C4.aiff")
    except Exception as e:
        print(f"Error loading AIFF: {e}")
        return

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    print(f"Loaded audio: {len(audio)} samples, SR: {sr}")

    # Convert to torch tensor
    y = torch.tensor(audio, dtype=torch.float32)

    # STFT parameters
    n_fft = 2048
    hop_length = 512
    window = torch.hann_window(n_fft)
    
    # Compute STFT
    print("Computing STFT...")
    stft = torch.stft(y, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    mag = stft.abs()
    
    # Frequencies
    freqs = torch.fft.rfftfreq(n_fft, 1/sr)
    
    # Base frequency for C4
    f0 = 261.63
    
    # Reconstruct
    resynth = torch.zeros_like(y)
    t = torch.arange(len(y), dtype=torch.float32) / sr
    
    print("Resynthesizing with 128 partials...")
    
    # Scale factor for STFT to time-domain amplitude (approx 1/(N_FFT/2))
    # Window scaling compensation is also needed
    scale_factor = 1.0 / (n_fft / 2 * window.mean()) 
    # Actually standard window sum rule... 
    # Simple approx: divide by sum of window / 2 for sinusoidal amplitude extraction?
    # Inverse of STFT magnitude for a sinusoid: Mag = A * Sum(Window) / 2
    # So A = Mag * 2 / Sum(Window)
    amp_scale = 2.0 / window.sum()
    
    for n in range(1, 129):
        # Theoretical harmonic freq
        target_f_harmonic = n * f0
        
        # Search window: +/- 50% of f0 initially, but we know it stretches up.
        # Let's search in [n*f0 - f0/4, n*f0 + n*f0*0.2] or something broader?
        # Safe bet: search +/- f0/2 centered on the expected inharmonic freq?
        # We don't know B. Let's just search +/- f0/2 around n*f0 for low n, 
        # but widen/shift for high n.
        
        # Simple robust approach: Search neighborhood of target_f
        # Since spacing is ~f0, searching +/- f0/2 is the max unique range.
        
        # Let's center on n*f0, search +/- f0/2. 
        # Inharmonicity shifts it UP. So for large n, n*f0 is too low.
        # But for n=64, shift is ~20%. 
        # So n*f0 might be 0.8 * actual_f. 
        # We need a smarter tracker or wide search.
        
        # Since we just want to grab the ENERGY, let's just find the max peak 
        # in the band [(n-0.5)*f0, (n+0.5)*f0]. 
        # Wait, if n=64 is shifted by 18%, that's 64*1.18 = 75.5 harmonics!
        # So "bin n" (around n*f0) corresponds to harmonic ~54?
        # We will heavily misalign if we stick to integer multiples of f0 boundaries.
        
        # Better: use a large B estimate to center the search window?
        # B ~ 0.0001
        est_freq = n * f0 * np.sqrt(1 + 0.0001 * n**2)
        
        search_width = f0 * 0.4
        f_min = est_freq - search_width
        f_max = est_freq + search_width
        
        # Find bins in range
        freq_indices = torch.where((freqs >= f_min) & (freqs <= f_max))[0]
        
        if len(freq_indices) == 0:
            continue
            
        # Find peak in this band (average over time? or max over time?)
        # Let's pick the bin with max TOTAL energy in the spectrogram to define the partial
        bin_energies = mag[freq_indices, :].sum(dim=1)
        best_local_idx = bin_energies.argmax()
        bin_idx = freq_indices[best_local_idx]
        
        actual_f = freqs[bin_idx]
        
        # Extract envelope
        env = mag[bin_idx, :] * amp_scale # Normalize amplitude
        env = torch.nn.functional.interpolate(env[None, None, :], size=len(y), mode='linear', align_corners=False).squeeze()
        
        # Synthesize
        partial = env * torch.sin(2 * torch.pi * actual_f * t)
        resynth += partial
        
    # Normalize
    # resynth = resynth / (resynth.abs().max() + 1e-12) # Don't normalize yet, want to compare energy

    # Calculate Energy
    energy_orig = (y**2).sum()
    energy_resynth = (resynth**2).sum()
    fraction = energy_resynth / energy_orig
    print(f"Original Energy: {energy_orig:.2f}")
    print(f"Resynth Energy: {energy_resynth:.2f}")
    print(f"Energy Fraction (Resynth/Orig): {fraction:.4f}")
    
    # Normalize for saving
    resynth_norm = resynth / (resynth.abs().max() + 1e-12)
    
    # Save
    print("Saving resynth_c4.wav...")
    # Convert to int16
    audio_int16 = (resynth_norm.numpy() * 32767).astype(np.int16)
    scipy.io.wavfile.write("resynth_c4.wav", sr, audio_int16)
    print("Done!")

if __name__ == "__main__":
    resynthesize_c4()
