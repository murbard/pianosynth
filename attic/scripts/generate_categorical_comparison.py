import torch
import torch.nn.functional as F
import scipy.io.wavfile as wavfile
import numpy as np
from pathlib import Path
from tqdm import tqdm

from pianosynth.optimization_batch import PianoParamPerKey
from pianosynth.spectral import diff_piano_render
from pianosynth.physics import calculate_partials

PROCESSED_DATA = Path("data/processed")
CHECKPOINT = Path("src/pianosynth/checkpoints/params_best_ever.pt")
OUT_DIR = Path("results_categorical")
OUT_DIR.mkdir(exist_ok=True)

def load_audio(midi, dyn, device='cpu'):
    path = PROCESSED_DATA / f"{midi}_{dyn}.pt"
    if not path.exists():
        # Try fallback
        for alt in ['mf', 'ff', 'pp']:
            p = PROCESSED_DATA / f"{midi}_{alt}.pt"
            if p.exists():
                # print(f"Fallback {midi} {dyn} -> {alt}")
                return torch.load(p).float().to(device)
        return None
    return torch.load(path).float().to(device)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Model
    model = PianoParamPerKey(device=device)
    if CHECKPOINT.exists():
        print(f"Loading {CHECKPOINT}")
        cpt = torch.load(CHECKPOINT, map_location=device)
        model.load_state_dict(cpt["model_state"])
    else:
        print("Checkpoint not found! Using initialized values.")

    dyn_map = {'pp': 0, 'mf': 1, 'ff': 2}
    
    # Common Render Params
    SR = 44100
    CLIP_DUR = 2.0 # Seconds per note snippet
    CLIP_SAMPLES = int(SR * CLIP_DUR)
    
    for dyn_str in ['mf']:
        print(f"Generating artifact for {dyn_str}...")
        full_buffer = []
        dyn_idx = dyn_map[dyn_str]
        
        for m in tqdm(range(21, 109)):
            # 1. Get Ground Truth
            gt_audio = load_audio(m, dyn_str, device=device)
            
            if gt_audio is not None:
                # Trim/Pad to Clip
                if len(gt_audio) > CLIP_SAMPLES:
                    gt_audio = gt_audio[:CLIP_SAMPLES]
                else:
                    gt_audio = F.pad(gt_audio, (0, CLIP_SAMPLES - len(gt_audio)))
            else:
                # Silence if missing
                gt_audio = torch.zeros(CLIP_SAMPLES, device=device)
            
            # 2. Synthesize
            with torch.no_grad():
                midi_t = torch.tensor([m], device=device).float()
                dyn_t = torch.tensor([dyn_idx], device=device).long()
                
                overrides = model(midi_t, dyn_t)
                
                # Expand implicit scalar overrides to [1]
                # (PianoParamPerKey handles batching, returning [Batch] tensors)
                
                phys = calculate_partials(midi_t, overrides, device=device)
                
                synth_audio = diff_piano_render(
                    freqs=phys["freqs"],
                    tau_s=phys["tau_s"],
                    tau_f=phys["tau_f"],
                    amps=phys["amps"],
                    w_curve=phys["w_curve"],
                    dur_samples=CLIP_SAMPLES,
                    reverb_wet=overrides["reverb_wet"],
                    reverb_decay=overrides["reverb_decay"]
                )
                
                # Normalize Synth to avoid clipping? 
                # Or trust it matches GT level implicitly via loss?
                # Trust model. But safety clip.
                synth_audio = synth_audio.clamp(-1.0, 1.0)
                
                # Ensure 1D for concatenation
                if synth_audio.ndim == 2:
                    synth_audio = synth_audio.squeeze(0)
                
            # 3. Concatenate: GT -> Synth
            # Gap of silence?
            gap = torch.zeros(int(SR * 0.1), device=device)
            
            pair = torch.cat([gt_audio, gap, synth_audio, gap])
            full_buffer.append(pair)
            
        print("Concatenating full sequence...")
        full_seq = torch.cat(full_buffer).cpu().numpy()
        
        # Save
        out_path = OUT_DIR / f"comparison_{dyn_str}.wav"
        wavfile.write(out_path, SR, (full_seq * 32767).astype(np.int16))
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
