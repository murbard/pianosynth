
import soundfile as sf
import scipy.io.wavfile
import numpy as np

def analyze_decay():
    print("Loading files...")
    # Load Original
    orig, sr_orig = sf.read("Piano.ff.C4.aiff")
    if len(orig.shape) > 1:
        orig = orig.mean(axis=1)
        
    # Load Resynth
    sr_resynth, resynth = scipy.io.wavfile.read("resynth_c4.wav")
    # Convert int16 to float
    resynth = resynth.astype(np.float32) / 32767.0
    
    # Ensure same length for comparison (truncate to shorter)
    min_len = min(len(orig), len(resynth))
    orig = orig[:min_len]
    resynth = resynth[:min_len]
    
    # Window size for RMS (e.g., 100ms)
    window_sec = 0.1
    window_samples = int(sr_orig * window_sec)
    
    print(f"Comparing RMS over {window_sec}s windows:")
    print(f"{'Time(s)':<10} | {'Orig dB':<10} | {'Resynth dB':<10} | {'Diff dB':<10}")
    print("-" * 46)
    
    # Analyze in 0.5s steps
    duration = len(orig) / sr_orig
    for t in np.arange(0, duration, 0.5):
        start_idx = int(t * sr_orig)
        end_idx = start_idx + window_samples
        
        if end_idx > len(orig):
            break
            
        chunk_orig = orig[start_idx:end_idx]
        chunk_resynth = resynth[start_idx:end_idx]
        
        rms_orig = np.sqrt(np.mean(chunk_orig**2) + 1e-12)
        rms_resynth = np.sqrt(np.mean(chunk_resynth**2) + 1e-12)
        
        db_orig = 20 * np.log10(rms_orig)
        db_resynth = 20 * np.log10(rms_resynth)
        diff = db_resynth - db_orig
        
        print(f"{t:<10.2f} | {db_orig:<10.2f} | {db_resynth:<10.2f} | {diff:<10.2f}")

if __name__ == "__main__":
    analyze_decay()
