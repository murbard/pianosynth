import unittest
import numpy as np
from pianosynth.pitch import estimate_pitch

class TestPitchEstimation(unittest.TestCase):
    def test_sine_wave_exact(self):
        sr = 44100
        freq = 440.0
        duration = 0.5
        t = np.arange(int(sr * duration)) / sr
        audio = np.sin(2 * np.pi * freq * t)
        
        # Test with expected freq close to actual
        est_freq, conf = estimate_pitch(audio, sr, expected_freq=440.0)
        
        self.assertAlmostEqual(est_freq, 440.0, delta=1.0)
        self.assertGreater(conf, 0.95)

    def test_sine_wave_shifted(self):
        # Actual is 442, Expected is 440 (within 2 semis)
        sr = 44100
        freq = 442.0
        t = np.arange(int(sr * 0.5)) / sr
        audio = np.sin(2 * np.pi * freq * t)
        
        est_freq, conf = estimate_pitch(audio, sr, expected_freq=440.0)
        self.assertAlmostEqual(est_freq, 442.0, delta=1.0)

    def test_out_of_bounds_constraints(self):
        # Actual is 880 (octave up), Expected is 440
        # Should NOT find 880, might find nothing or 440 (if subharmonic strong) 
        # but sine wave doesn't have subharmonics.
        # This checks that we obey the window limits.
        sr = 44100
        freq = 880.0
        t = np.arange(int(sr * 0.5)) / sr
        audio = np.sin(2 * np.pi * freq * t)
        
        # Search around 440 (+/- 2 semitones = 391 to 493)
        # 880 Hz period is outside the search window [min_period, max_period] corresponding to freq range.
        # It should likely pick the edge or fail.
        
        # The logic:
        # min_freq ~ 391 -> max_period ~ 112
        # max_freq ~ 493 -> min_period ~ 89
        
        # 880 Hz period is ~ 50. This is < min_period (89).
        # So it won't check lag 50.
        
        est_freq, conf = estimate_pitch(audio, sr, expected_freq=440.0)
        
        # Should not be 880
        self.assertNotAlmostEqual(est_freq, 880.0, delta=5.0)

    def test_silence(self):
        sr = 44100
        audio = np.zeros(4096)
        est_freq, conf = estimate_pitch(audio, sr, expected_freq=440.0)
        self.assertLess(conf, 0.1)

    def test_noisy_sine(self):
        # Sine wave with white noise (SNR check)
        sr = 44100
        freq = 440.0
        t = np.arange(int(sr * 0.5)) / sr
        signal = np.sin(2 * np.pi * freq * t)
        noise = np.random.normal(0, 0.2, len(signal)) # 20% noise amplitude
        audio = signal + noise
        
        est_freq, conf = estimate_pitch(audio, sr, expected_freq=440.0)
        self.assertAlmostEqual(est_freq, 440.0, delta=2.0)
        self.assertGreater(conf, 0.8) # Should still be fairly confident

    def test_missing_fundamental(self):
        # Harmonic series: 2f, 3f, 4f... missing f
        sr = 44100
        freq = 220.0 # Fundamental
        t = np.arange(int(sr * 0.5)) / sr
        
        # Build signal with only harmonics 2, 3, 4
        # 440, 660, 880
        audio = (0.5 * np.sin(2 * np.pi * (2*freq) * t) + 
                 0.3 * np.sin(2 * np.pi * (3*freq) * t) + 
                 0.2 * np.sin(2 * np.pi * (4*freq) * t))
                 
        # Yin-based methods are usually good at finding the "period" even if f0 energy is absent
        # because the pattern repeats at 1/f0.
        est_freq, conf = estimate_pitch(audio, sr, expected_freq=220.0)
        
        self.assertAlmostEqual(est_freq, 220.0, delta=2.0)
        self.assertGreater(conf, 0.8)

