import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm

from pianosynth.optimization_batch import PianoParamPerKey
from pianosynth.spectral import diff_piano_render
from pianosynth.physics import calculate_partials

CHECKPOINT_DIR = Path("src/pianosynth/checkpoints")
PROCESSED_DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path(".")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    full_audio_segments = []
    # 0.2s gap
    gap_silence = torch.zeros(int(44100 * 0.2), device=device)
    
    # Load Master Checkpoint
    checkpoint_path = CHECKPOINT_DIR / "params_all_keys.pt"
    if not checkpoint_path.exists():
        print("Master params_all_keys.pt not found.")
        return
        
    print(f"Loading {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PianoParamPerKey(device=device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    
    # Velocities?
    # We can use a standard velocity for synthesis (e.g. 0.6) or 
    # try to match the training data.
    # Let's use a fixed reasonable velocity for comparison consistency.
    velocity_val = 0.6
    
    # Iterate all notes (21-108)
    notes = list(range(21, 109))
    
    # Sparse training only updated 27 notes? 
    # If we render all 88, the untrained ones will be default curves.
    # That's fine.
    
    dur_samples = 44100 * 2 # 2 seconds
    
    with torch.no_grad():
        for midi in tqdm(notes, desc="Rendering"):
            # Prepare Inputs
            midi_t = torch.tensor([midi], device=device).float()
            velocity = torch.tensor([velocity_val], device=device).float()
            
            # Get Params
            overrides = model(midi_t)
            
            # Physics
            phys_out = calculate_partials(
                midi=midi_t,
                velocity=velocity,
                overrides=overrides,
                n_partials=64,
                device=device
            )
            
            # Synth
            synth_audio = diff_piano_render(
                freqs=phys_out["freqs"],
                tau_s=phys_out["tau_s"],
                tau_f=phys_out["tau_f"],
                amps=phys_out["amps"],
                w_curve=phys_out["w_curve"],
                dur_samples=dur_samples,
                reverb_wet=overrides.get("reverb_wet"),
                reverb_decay=overrides.get("reverb_decay")
            ).squeeze()
            
            # Norm
            synth_audio = synth_audio / (synth_audio.abs().max() + 1e-6) * 0.8
            
            # Try to load Real Audio for comparison
            # Look for MF
            real_audio = None
            f_real = PROCESSED_DATA_DIR / f"{midi}_mf.pt"
            if f_real.exists():
                real_audio = torch.load(f_real).float().to(device)
            else:
                 # Try finding any
                 raws = list(PROCESSED_DATA_DIR.glob(f"{midi}_*.pt"))
                 if raws: real_audio = torch.load(raws[0]).float().to(device)
            
            segment_parts = []
            
            if real_audio is not None:
                # Trim/Pad Real
                if len(real_audio) > dur_samples: real_audio = real_audio[:dur_samples]
                else: real_audio = torch.cat([real_audio, torch.zeros(dur_samples - len(real_audio), device=device)])
                
                real_audio = real_audio / (real_audio.abs().max() + 1e-6) * 0.8
                segment_parts.append(real_audio)
                segment_parts.append(gap_silence)
                
            segment_parts.append(synth_audio)
            segment_parts.append(gap_silence)
            
            full_audio_segments.append(torch.cat(segment_parts))
            
    if not full_audio_segments:
        print("No audio generated.")
        return
        
    full_audio = torch.cat(full_audio_segments)
    
    out_path = OUTPUT_DIR / "full_comparison.wav"
    sf.write(out_path, full_audio.detach().cpu().numpy(), 44100)
    print(f"Saved comparison to {out_path}")

if __name__ == "__main__":
    main()
