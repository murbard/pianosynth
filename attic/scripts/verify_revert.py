
import torch
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pianosynth import synth, physics, default_params

def test_revert():
    print("Testing Revert of String Parameters...")
    
    # 1. Determinism
    print("\n--- Testing Determinism ---")
    y1 = synth.piano_additive(60, velocity=0.7, dur=1.0)
    y2 = synth.piano_additive(60, velocity=0.7, dur=1.0)
    
    diff = (y1 - y2).abs().max()
    print(f"Difference between two runs: {diff}")
    assert diff < 1e-6, "Output is not deterministic!"
    print("PASS: Output is deterministic.")
    
    # 2. String Wiggle
    print("\n--- Testing String Wiggle ---")
    # We need to inspect physics internals or use physics directly.
    # checking physics.calculate_partials
    
    overrides = synth.EPS_DEFAULTS.copy()
    # Need to flatten
    flat_overrides = {}
    for cat, sub in overrides.items():
        if isinstance(sub, dict):
            for k, v in sub.items():
                flat_overrides[k] = v
        else:
             print(f"Skipping {cat}")
             
    # Physics uses specific keys. We need to make sure we didn't break "get_overrides" usage.
    # Actually physics.calculate_partials takes a dict that P() accesses. 
    # In physics.py: def P(k): return overrides[k]
    # So we just pass the flat dict.
    
    res = physics.calculate_partials(60, 0.7, flat_overrides, device="cpu")
    freqs = res["freqs"] # Reference
    # Wait, res returns "freqs" as Center freq? physics.py line 296 says "freqs": fn
    # fn is reference [1, N, 1] ... wait.
    
    # Let's check calculate_partials implementation again.
    # It returns "freqs" as fn (ref). The 3-string logic is inside f0_s and used for rendering?
    # No, physics.py returns "freqs": fn.
    # BUT, we want to know if the underlying F0s are different.
    
    # The current physics.py implementation calculates f0_s [Batch, 1, 3]
    # But currently it DOES NOT return f0_s in the dict!
    # It returns "freqs": fn.
    # And "amps": amps_3.
    
    # Wait, if `freqs` in output is just `fn`, then `diff_piano_render` will render 3 strings at SAME FREQUENCY?
    # physics.py:296 "freqs": fn.
    # physics.py:183 fn = n_exp * f0_s * sqrt(...) 
    # Ah, fn IS calculated from f0_s!
    # Let's check line 183 in physics.py
    # fn = n_exp * f0_s * ...
    # f0_s is [Batch, 1, 3].
    # So fn is [Batch, N, 3].
    
    # Let's verify shape of freqs
    print(f"Freqs shape: {res['freqs'].shape}")
    assert res['freqs'].ndim == 3, "Freqs should be [Batch, N, 3]"
    assert res['freqs'].shape[2] == 3, "Freqs should have 3 strings"
    
    f_center = res['freqs'][0, 0, 1]
    f_left = res['freqs'][0, 0, 0]
    f_right = res['freqs'][0, 0, 2]
    
    print(f"F_left: {f_left}, F_center: {f_center}, F_right: {f_right}")
    
    assert f_left != f_center, "Left string should differ from center"
    assert f_right != f_center, "Right string should differ from center"
    print("PASS: Strings have variations.")
    
    # 3. Fixed Seed check
    print("\n--- Testing Fixed Seed ---")
    res2 = physics.calculate_partials(60, 0.7, flat_overrides, device="cpu")
    f_left_2 = res2['freqs'][0, 0, 0]
    
    print(f"Run 1 Left: {f_left}")
    print(f"Run 2 Left: {f_left_2}")
    
    assert (f_left - f_left_2).abs() < 1e-9, "Physics Randomness should be fixed seed!"
    print("PASS: Physics is deterministic.")

if __name__ == "__main__":
    test_revert()
