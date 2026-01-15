import numpy as np
from pianosynth.pitch import estimate_pitch

def test_sine():
    sr = 44100
    freq = 110.0
    t = np.arange(int(sr*0.5))/sr
    audio = np.sin(2*np.pi*freq*t)
    
    f0, conf = estimate_pitch(audio, sr, freq)
    print(f"Sine 110Hz -> Est: {f0:.2f}, Conf: {conf:.2f}")
    assert conf > 0.9
    assert abs(f0 - freq) < 1.0

if __name__ == "__main__":
    test_sine()
