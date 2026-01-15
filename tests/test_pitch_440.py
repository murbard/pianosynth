import numpy as np
from pianosynth.pitch import estimate_pitch

def test_sine_440():
    sr = 44100
    freq = 440.0
    t = np.arange(int(sr*0.5))/sr
    audio = np.sin(2*np.pi*freq*t)
    
    # Test constrained
    f0, conf = estimate_pitch(audio, sr, freq, bounds_semitones=2)
    print(f"Sine 440Hz (Bounds 2) -> Est: {f0:.2f}, Conf: {conf:.2f}")
    
    # Test global-ish
    f0g, confg = estimate_pitch(audio, sr, freq, bounds_semitones=12)
    print(f"Sine 440Hz (Bounds 12) -> Est: {f0g:.2f}, Conf: {confg:.2f}")

    if conf < 0.9:
        print("FAIL: Low confidence on sine wave!")
    
if __name__ == "__main__":
    test_sine_440()
