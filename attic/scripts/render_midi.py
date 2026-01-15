import torch
import torch.nn as nn
import pretty_midi
import numpy as np
import scipy.io.wavfile as wav
import argparse
import glob
from tqdm import tqdm

from pianosynth.optimization import PianoParam
from pianosynth.spectral import diff_piano_render
from pianosynth.physics import calculate_partials

CLIP_LEN_SAMPLES = 88200 # Max duration per note (2s)
SAMPLE_RATE = 44100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch.nn.functional as F
import os

def render_sampler(midi_path, output_path):
    print(f"Loading MIDI: {midi_path}")
    pm = pretty_midi.PrettyMIDI(midi_path)
    
    # Load all samples into memory (simple sampler)
    # Map (midi) -> (audio_tensor) for 'mf'
    # For now, just load 'mf' samples to keep it simple or pick closest dynamic?
    # User asked for ground truth. 
    # Let's load all {midi}_mf.pt files.
    
    print("Loading samples...")
    sample_map = {}
    data_dir = "data/processed"
    files = glob.glob(os.path.join(data_dir, "*_mf.pt"))
    for f in tqdm(files):
        # f: 60_mf.pt
        try:
            name = os.path.basename(f)
            midi = int(name.split('_')[0])
            audio = torch.load(f, map_location=DEVICE)
            sample_map[midi] = audio
        except:
            continue
            
    print(f"Loaded {len(sample_map)} samples.")
    
    total_time = pm.get_end_time()
    total_samples = int((total_time + 5.0) * SAMPLE_RATE)
    output_buffer = torch.zeros(total_samples, device=DEVICE)
    
    print("Rendering with sampler...")
    with torch.no_grad():
        for instrument in pm.instruments:
            if instrument.is_drum: continue
            
            for note in tqdm(instrument.notes):
                start_time = note.start
                end_time = note.end
                duration = end_time - start_time
                midi_note = note.pitch
                
                # Find nearest sample
                available_midis = np.array(list(sample_map.keys()))
                idx = (np.abs(available_midis - midi_note)).argmin()
                nearest_midi = available_midis[idx]
                sample_audio = sample_map[nearest_midi]
                
                # Pitch shift?
                # Simple resampling: speed = target_freq / source_freq
                # 2 ** ((target - source) / 12)
                semitone_diff = midi_note - nearest_midi
                
                # If diff is 0, use as is.
                # If not, resample.
                if semitone_diff == 0:
                    note_audio = sample_audio
                else:
                    # Resample
                    # Speed factor > 1 means higher pitch (shorter)
                    # new_len = old_len / speed
                    speed = 2 ** (semitone_diff / 12.0)
                    
                    # interpolate
                    # view as [1, 1, L]
                    src = sample_audio.view(1, 1, -1)
                    new_len = int(src.shape[-1] / speed)
                    note_audio = F.interpolate(src, size=new_len, mode='linear', align_corners=False).view(-1)
                
                # Apply envelope for duration
                # Sampler note is natural decay.
                # But we must cut it off if duration is short?
                # For consistency with synth:
                # Let ring unless duration < sample len?
                # Piano keys dampen when released.
                
                note_off_sample = int(duration * SAMPLE_RATE)
                RELEASE_TIME = 0.2
                release_samples = int(RELEASE_TIME * SAMPLE_RATE)
                
                target_len = note_off_sample + release_samples
                
                # Pad/Cut note_audio
                if note_audio.shape[0] < target_len:
                    note_audio = F.pad(note_audio, (0, target_len - note_audio.shape[0]))
                else:
                    note_audio = note_audio[:target_len]
                    
                # Apply release fade
                if target_len > note_off_sample:
                   fade_start = note_off_sample
                   len_fade = target_len - fade_start
                   fade_curve = torch.linspace(1.0, 0.0, len_fade, device=DEVICE)
                   # Apply
                   # We need to construct full envelope
                   env = torch.ones(target_len, device=DEVICE)
                   env[fade_start:] = fade_curve
                   note_audio = note_audio * env
                
                # Add to buffer
                start_sample = int(start_time * SAMPLE_RATE)
                end_sample = start_sample + target_len
                
                if end_sample > total_samples:
                    # truncate
                    valid_len = total_samples - start_sample
                    note_audio = note_audio[:valid_len]
                    end_sample = total_samples
                    
                output_buffer[start_sample:end_sample] += note_audio
                
    # Normalize
    output_buffer = output_buffer / (output_buffer.abs().max() + 1e-6)
    wav_data = (output_buffer.cpu().numpy() * 32767).astype(np.int16)
    wav.write(output_path, SAMPLE_RATE, wav_data)
    print(f"Saved sampler render to {output_path}")

def render_midi(midi_path, checkpoint_path, output_path, use_sampler=False):
    if use_sampler:
        render_sampler(midi_path, output_path)
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Init Model
    model = PianoParam().to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    
    # Global Params
    noise_params = checkpoint.get('noise_params', {'gain': 0.0, 'color': 0.5})
    reverb_params = checkpoint.get('reverb_params', {'wet': 0.0, 'decay': 0.5})
    
    # Ensure they are on device (if tensors) or floats
    # ...logic to handle dict values...
    
    print(f"Loading MIDI: {midi_path}")
    pm = pretty_midi.PrettyMIDI(midi_path)
    
    # Render buffer
    # Estimate total duration
    total_time = pm.get_end_time()
    total_samples = int((total_time + 5.0) * SAMPLE_RATE) # +5s tail
    output_buffer = torch.zeros(total_samples, device=DEVICE)
    
    overrides = model.get_overrides()
    
    print("Rendering notes from all tracks...")
    
    # Preload samples if sampler mode
    # For now, let's keep it simple: Synth rendering with damper handling.
    # Sampler rendering needs a separate path or function as it requires loading 260 wavs.
    # We will implement damper logic here first.
    
    with torch.no_grad():
        for instrument in pm.instruments:
            if instrument.is_drum:
                continue
                
            print(f"Track: {instrument.name}, Notes: {len(instrument.notes)}")
            
            for note in tqdm(instrument.notes):
                start_time = note.start
                end_time = note.end
                duration = end_time - start_time
                velocity = note.velocity / 127.0
                midi_note = note.pitch
                
                # Damper Logic:
                # Render for duration + release tail
                RELEASE_TIME = 0.2 # 200ms release
                render_dur_sec = duration + RELEASE_TIME
                RENDER_LEN = int(render_dur_sec * SAMPLE_RATE)
                
                phys_out = calculate_partials(
                    midi=torch.tensor([float(midi_note)], device=DEVICE),
                    velocity=torch.tensor([float(velocity)], device=DEVICE),
                    overrides=overrides,
                    n_partials=64,
                    device=DEVICE
                )
                
                note_audio = diff_piano_render(
                    freqs=phys_out["freqs"],
                    tau_s=phys_out["tau_s"],
                    tau_f=phys_out["tau_f"],
                    amps=phys_out["amps"],
                    w_curve=phys_out["w_curve"],
                    dur_samples=RENDER_LEN,
                    noise_params=noise_params,
                    reverb_params=reverb_params
                )
                
                # Apply Release Envelope
                # Simple linear fade out after note-off
                note_off_sample = int(duration * SAMPLE_RATE)
                release_samples = int(RELEASE_TIME * SAMPLE_RATE)
                
                # Envelope: 1.0 until note_off, then linear to 0
                env = torch.ones(RENDER_LEN, device=DEVICE)
                if RENDER_LEN > note_off_sample:
                   fade_len = min(release_samples, RENDER_LEN - note_off_sample)
                   fade = torch.linspace(1.0, 0.0, fade_len, device=DEVICE)
                   env[note_off_sample:note_off_sample+fade_len] = fade
                   # Zero out rest if any
                   env[note_off_sample+fade_len:] = 0.0
                
                note_audio = note_audio * env
                
                # Add to buffer
                start_sample = int(start_time * SAMPLE_RATE)
                end_sample = start_sample + RENDER_LEN
                
                if end_sample > total_samples:
                    # truncate
                    valid_len = total_samples - start_sample
                    note_audio = note_audio[..., :valid_len]
                    end_sample = total_samples
                
                # Simple summing for polyphony
                output_buffer[start_sample:end_sample] += note_audio[0]
        
    # Normalize
    output_buffer = output_buffer / (output_buffer.abs().max() + 1e-6)
    
    # Save
    wav_data = (output_buffer.cpu().numpy() * 32767).astype(np.int16)
    wav.write(output_path, SAMPLE_RATE, wav_data)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi", type=str, default="bwv772.mid")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/bach_rendered.wav")
    parser.add_argument("--sampler", action="store_true", help="Use recorded samples instead of synth")
    args = parser.parse_args()
    
    if args.sampler:
        # Checkpoint not needed for sampler
        pass
    elif args.checkpoint is None:
        parser.error("--checkpoint required unless --sampler is set")
    
    render_midi(args.midi, args.checkpoint, args.output, use_sampler=args.sampler)
