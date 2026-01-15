import unittest
import numpy as np
from pianosynth.process import shift_pitch, process_track
from pianosynth.pitch import estimate_pitch

class TestProcessing(unittest.TestCase):
    def test_shift_pitch_resample_accuracy(self):
        """
        Verify that shifting a tone by X semitones and estimating its pitch
        results in the expected target frequency.
        """
        sr = 44100
        freq = 440.0 
        t = np.arange(int(sr * 1.0)) / sr
        audio = np.sin(2 * np.pi * freq * t)
        
        # Shift by +50 cents (+0.5 semitones)
        # New freq should be 440 * 2^(0.5/12)
        target_shift = 0.5
        target_freq = freq * (2 ** (target_shift / 12.0))
        
        shifted_audio = shift_pitch(audio, sr, semitones=target_shift)
        
        # Verify length changed
        expected_len_ratio = 1 / (2 ** (target_shift / 12.0))
        actual_len_ratio = len(shifted_audio) / len(audio)
        self.assertAlmostEqual(actual_len_ratio, expected_len_ratio, places=3)
        
        # Verify Pitch
        est_freq, conf = estimate_pitch(shifted_audio, sr, expected_freq=target_freq)
        self.assertAlmostEqual(est_freq, target_freq, delta=1.0)
        self.assertGreater(conf, 0.95)

    def test_process_track_silence(self):
        sr = 44100
        audio = np.zeros(int(sr * 0.5))
        result, meta = process_track(audio, sr, midi_note=69)
        self.assertIsNone(result)
        self.assertEqual(meta['status'], 'silent')

    def test_process_track_octave_error(self):
        """
        If we give it a 440Hz tone but tell it it's MIDI 81 (880Hz),
        constrained search should fail to find 880Hz pitch in the signal.
        Result should be rejection (None).
        """
        sr = 44100
        freq = 440.0 
        t = np.arange(int(sr * 0.5)) / sr
        audio = np.sin(2 * np.pi * freq * t)
        
        result, meta = process_track(audio, sr, midi_note=81)
        
        self.assertIsNone(result)
        self.assertEqual(meta['status'], 'low_confidence_pitch')

    def test_process_dataset_integration(self):
        """
        Integration test for process_dataset.
        Sets up a dummy raw directory with one valid note and one missing note.
        Verifies that:
        1. Valid note is processed and tuned.
        2. Missing note is interpolated from the valid one.
        3. Log file is created and correct.
        """
        import tempfile
        import shutil
        import os
        from pathlib import Path
        import soundfile as sf
        import pandas as pd
        import torch
        from pianosynth.process import process_dataset
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir) / "raw"
            out_dir = Path(tmp_dir) / "processed"
            
            # Setup Raw: C4 (midi 60) 'mf'
            # Create a sine wave at 261.6 Hz (C4)
            # Actually, let's detune it slightly to check tuning
            # C4 is 261.63. Let's make it 262.5 (+6 cents)
            sr = 44100
            freq = 262.5
            t = np.arange(int(sr * 0.5)) / sr
            audio = np.sin(2 * np.pi * freq * t)
            
            # Save as C4.aiff in mf folder
            mf_dir = raw_dir / "mf"
            mf_dir.mkdir(parents=True)
            sf.write(mf_dir / "C4.aiff", audio, sr)
            
            # Run Process
            # This should process C4, and try to find everything else.
            # It should interpolate C#4 (61) from C4.
            
            process_dataset(raw_dir, out_dir)
            
            # Verify Output Files
            p1 = torch.load(out_dir / "60_mf.pt")
            p2 = torch.load(out_dir / "61_mf.pt")
            
            # Check existence and length equality
            self.assertTrue((out_dir / "60_mf.pt").exists())
            self.assertTrue((out_dir / "61_mf.pt").exists())
            self.assertEqual(len(p1), len(p2), "Processed files should have uniform length")
            
            # Verify Log
            log_path = out_dir / "process_log.csv"
            self.assertTrue(log_path.exists())
            
            df = pd.read_csv(log_path)
            
            # Check C4 Record
            c4_rec = df[(df['midi'] == 60) & (df['dynamic'] == 'mf')].iloc[0]
            self.assertEqual(c4_rec['status'], 'ok')
            self.assertEqual(c4_rec['source'], 'raw')
            # approx 6 cent shift
            # We shifted FROM 262.5 TO 261.6
            # Tuning down -> negative shift
            # 1200 * log2(261.63/262.5) ~= -5.7 cents
            self.assertLess(c4_rec['pitch_shift_cents'], 0)
            
            # Check C#4 Record (Interpolated)
            cs4_rec = df[(df['midi'] == 61) & (df['dynamic'] == 'mf')].iloc[0]
            self.assertEqual(cs4_rec['status'], 'ok')
            self.assertTrue(cs4_rec['source'].startswith('interpolated_from_60'))
            self.assertEqual(cs4_rec['interpolation_shift'], 1)

    def test_harmonic_lock_handling(self):
        """
        Simulate a signal where pitch estimator locks onto 2nd harmonic (octave higher).
        Pipeline should recognize this as "close enough to harmonic" and NOT reject it,
        and NOT shift it down an octave.
        It should effectively treat it as the correct note.
        """
        sr = 44100
        target_f0 = 110.0 # A2
        detected_f0 = 220.0 # A3 (2nd harmonic)
        
        # Synthesize 220Hz
        t = np.arange(int(sr * 0.5)) / sr
        audio = np.sin(2 * np.pi * detected_f0 * t)
        
        # Tell process_track it is A2 (midi 45)
        # It will see 220Hz.
        result, meta = process_track(audio, sr, midi_note=45)
        
        # Should be OK, not rejected
        self.assertIsNotNone(result)
        self.assertEqual(meta['status'], 'ok')
        # Shift should be small (tuning correction), NOT -1200 cents
        self.assertTrue(abs(meta['pitch_shift_cents']) < 50, f"Shift was {meta['pitch_shift_cents']}, expected near 0")

    def test_interpolation_limit(self):
        """
        Verify that interpolation is strictly limited to +/- 1 semitone.
        """
        import tempfile
        import pandas as pd
        from pianosynth.process import process_dataset
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir) / "raw"
            out_dir = Path(tmp_dir) / "processed"
            
            # Setup: C4 (60) exists. D4 (62) is missing.
            # Gap is 2 semitones. Should NOT interpolate.
            
            mf_dir = raw_dir / "mf"
            mf_dir.mkdir(parents=True)
            
            # Create valid C4
            sr = 44100
            t = np.arange(int(sr * 0.5)) / sr
            audio = np.sin(2 * np.pi * 261.6 * t)
            import soundfile as sf
            sf.write(mf_dir / "C4.aiff", audio, sr)
            
            process_dataset(raw_dir, out_dir)
            
            log_path = out_dir / "process_log.csv"
            df = pd.read_csv(log_path)
            
            # Check 62 (D4)
            d4_rec = df[(df['midi'] == 62) & (df['dynamic'] == 'mf')].iloc[0]
            # Should be missing, NOT interpolated
            self.assertNotEqual(d4_rec['status'], 'ok')
            self.assertNotEqual(d4_rec['source'], 'interpolated_from_60')

    def test_log_consistency(self):
        """
        Verify that the process log accurately reflects the state of processed files.
        Ensures that if a file is processed, its log entry has:
        - status='ok' (or fallback status)
        - final_samples > 0
        - source='raw'
        """
        import tempfile
        import pandas as pd
        import soundfile as sf
        from pianosynth.process import process_dataset
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir) / "raw"
            (raw_dir / "mf").mkdir(parents=True)
            out_dir = Path(tmp_dir) / "processed"
            
            # Create a valid file
            sr = 44100
            t = np.arange(int(sr * 1.0)) / sr
            # Use sin wave A4 (midi 69)
            audio = np.sin(2 * np.pi * 440.0 * t)
            # Add some silence at start to trigger trimming logic
            audio = np.concatenate([np.zeros(1000), audio])
            
            sf.write(raw_dir / "mf" / "A4.aiff", audio, sr) # A4 is midi 69
            
            process_dataset(raw_dir, out_dir)
            
            # Check Log
            log_path = out_dir / "process_log.csv"
            df = pd.read_csv(log_path)
            
            # Get record for A4 mf (midi 69)
            rec = df[(df['midi'] == 69) & (df['dynamic'] == 'mf')].iloc[0]
            
            # Check consistency
            self.assertIn(rec['status'], ['ok', 'low_confidence_pitch_fallback', 'high_drift_fallback'])
            self.assertTrue(rec['final_samples'] > 0)
            self.assertEqual(rec['source'], 'raw')
            
            # Check file exists
            pt_path = out_dir / "69_mf.pt"
            self.assertTrue(pt_path.exists())
            
            # Load file and check length matches log
            t_loaded = torch.load(pt_path)
            self.assertEqual(t_loaded.shape[0], rec['final_samples'])

