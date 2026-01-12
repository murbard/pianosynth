import torch
import json
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from pianosynth.optimization import PianoParam
from pianosynth.spectral import MultiResSTFTLoss, diff_piano_render
from pianosynth.physics import calculate_partials

# Re-use dataset logic from train_spectral
class PianoAudioDataset(Dataset):
    def __init__(self, data_root, device="cpu"):
        self.files = sorted(list(Path(data_root).glob("*.wav")))
        self.device = device
        self.cache = []
        print("Loading audio for eval...")
        for f in tqdm(self.files):
            # Parse midi/vel from filename? 
            # Filename format: Piano.ff.B3.wav
            # logic in train_spectral used existing analysis data just for mapping?
            # actually train_spectral loads audio and parses filename.
            
            parts = f.name.split('.')
            # dynamic = parts[1] (pp, mf, ff)
            # note = parts[2] (A0, C4...)
            
            # Map note to midi
            # We need a helper, or just use train_params logic
            pass 
            # Let's trust train_spectral logic.
            # actually I'll just copy the necessary parsing logic here.

def parse_note(note_str):
    # e.g. C4, Bb3, F#5
    # simple parser
    notes = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    notes_alt = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
    octave = int(note_str[-1])
    pitch = note_str[:-1]
    
    try:
        idx = notes.index(pitch)
    except:
        idx = notes_alt.index(pitch)
        
    return idx + (octave + 1) * 12

def parse_dynamic(dyn_str):
    if dyn_str == "pp": return 25
    if dyn_str == "mf": return 70
    if dyn_str == "ff": return 110
    return 70

def load_data(data_root):
    # Load .pt files like train_spectral.py
    files = list(Path(data_root).glob("*.pt"))
    files = [f for f in files if f.name != "metadata.pt" and f.name != "analysis_data.pt"]
    
    data = []
    for f in files:
        # filename fmt: {midi}_{dyn}.pt
        parts = f.stem.split('_')
        if len(parts) < 2: continue
        
        midi = int(parts[0])
        dyn = parts[1]
        
        # Approximate velocity
        if dyn == 'pp': vel = 0.2
        elif dyn == 'mf': vel = 0.5
        elif dyn == 'ff': vel = 0.8
        else: vel = 0.5
        
        try:
            audio = torch.load(f).float()
            # Clip to 2 sec if needed, but train_spectral does this on batch.
            # We'll clip here to save RAM/time
            sr = 44100
            if len(audio) > sr * 2:
                audio = audio[:sr*2]
            
            data.append({
                "midi": midi,
                "vel": vel,
                "audio": audio,
                "sr": sr,
                "name": f.name
            })
        except:
            continue
    return data

    return data

def flatten_defaults(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten_defaults(v, prefix + k + "_"))
        elif isinstance(v, (int, float)):
            out[prefix + k] = torch.tensor(float(v))
        # Skip strings like "description"
    return out

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Data
    data = load_data("data/processed") # Correct path
    print(f"Loaded {len(data)} samples.")
    if len(data) == 0:
        print("No data found.")
        return
        
    # Validation subset (every 10th sample)
    val_data = data[::10]
    print(f"Eval set: {len(val_data)} samples.")
    
    # Load Models
    model = PianoParam(device=device)
    
    # Optimized
    ckpt = torch.load("src/pianosynth/params_spectral.pt", map_location=device)
    if "model_state" in ckpt: model.load_state_dict(ckpt["model_state"])
    else: model.load_state_dict(ckpt)
    
    opt_overrides = model.get_overrides()
    # Detach
    opt_overrides = {k: v.detach() for k, v in opt_overrides.items()}
    
    # Default
    with open("src/pianosynth/default_params.json", "r") as f:
        defaults_json = json.load(f)
    flat = flatten_defaults(defaults_json)
    
    # Manual map to physics.py expected keys
    def_overrides = {}
    
    # helper
    def T(k): return flat[k].to(device)
    
    # Tuning
    def_overrides["railsback_gamma"] = T("tuning_railsback_gamma")
    def_overrides["stretch_bass_amount"] = T("tuning_stretch_bass_amount")
    def_overrides["stretch_bass_range"] = T("tuning_stretch_bass_range")
    def_overrides["stretch_treble_amount"] = T("tuning_stretch_treble_amount")
    def_overrides["stretch_treble_range"] = T("tuning_stretch_treble_range")
    
    # Inharmonicity
    def_overrides["B_knots"] = torch.tensor(defaults_json["inharmonicity"]["B_knots"], device=device)
    def_overrides["B_scale"] = T("inharmonicity_B_scale")
    def_overrides["string_variation_std"] = T("inharmonicity_string_variation_std")

    # Decay
    def_overrides["A"] = T("decay_A")
    def_overrides["p"] = T("decay_p")
    def_overrides["k0"] = T("decay_k0")
    def_overrides["k1"] = T("decay_k1")
    def_overrides["tau_fast_divisor"] = T("decay_tau_fast_divisor")
    
    # Strike Point
    def_overrides["xh_bass"] = T("strike_point_xh_bass")
    def_overrides["xh_treble"] = T("strike_point_xh_treble")
    
    # Unison (Note: physics.py doesn't strictly use these if optimization.py handles detuning logic applied to freq? 
    # physics.py calculates f0 with stretch. It does NOT apply unison detuning?
    # optimization.py applies unison detuning?
    # Wait. physics.py calculates 'freqs' = single frequency per partial (fn).
    # WHERE is the unison beating handled?
    # diff_piano_render takes 'freqs' [Batch, N].
    # If physics.py returns 1 freq per partial, there is NO beating.
    # The 'piano_additive' in synth.py rendered 3 strings per note!
    # diff_piano_render renders 1 string per partial (or sum of partials).
    # DOES THE OPTIMIZED MODEL SUPPORT UNISON BEATING?
    # 'physics.py' calculates 'freqs'.
    # default synth has 'unison_detuning' parameter.
    # optimization.py has 'unison_detuning'.
    # But `calculate_partials` in `physics.py` does NOT use `dc_base` etc to return multiple frequencies!
    # It returns `fn`.
    # And `diff_piano_render` takes `freqs`.
    # Does `diff_piano_render` support expanding to 3 strings?
    # No. `diff_piano_render` is a simple partial summer.
    # THIS IS THE ISSUE. 
    # The optimized model is MONO-STRING (per partial).
    # The default model is TRI-STRING (unison).
    # The "worse" sound might be the lack of rich chorusing!
    # And the "lower loss" might be because STFT loss is insensitive to fine beating interference patterns or averages them out?
    # Wait, check `spectral.py`.
    
    # Unison params
    def_overrides["dc_base"] = T("unison_detuning_dc_base")
    def_overrides["dc_slope"] = T("unison_detuning_dc_slope")
    def_overrides["random_detune_base"] = T("unison_detuning_random_detune_base")
    def_overrides["random_detune_slope"] = T("unison_detuning_random_detune_slope")

    # Hammer
    def_overrides["comb_mix"] = T("hammer_comb_mix")
    def_overrides["comb_base"] = T("hammer_comb_base")
    def_overrides["tilt_base"] = T("hammer_tilt_base")
    def_overrides["tilt_slope"] = T("hammer_tilt_slope")
    def_overrides["fc_min"] = T("hammer_fc_min")
    def_overrides["fc_max"] = T("hammer_fc_max")
    def_overrides["fc_v_power"] = T("hammer_fc_v_power")
    def_overrides["fc_f_power"] = T("hammer_fc_f_power")
    def_overrides["nw_base"] = T("hammer_nw_base")
    def_overrides["nw_slope"] = T("hammer_nw_slope")

    # Body
    def_overrides["highpass_freq"] = T("body_filter_highpass_freq")
    def_overrides["highpass_power"] = T("body_filter_highpass_power")
    def_overrides["lowpass_freq"] = T("body_filter_lowpass_freq")
    def_overrides["lowpass_power"] = T("body_filter_lowpass_power")
    
    # Prompt (W Mix)
    def_overrides["n_wmix_base"] = T("prompt_sound_n_wmix_base")
    def_overrides["n_wmix_slope"] = T("prompt_sound_n_wmix_slope")
    def_overrides["w_min"] = T("prompt_sound_w_min")
    def_overrides["w_max"] = T("prompt_sound_w_max")
    
    # Loss Fn
    loss_fn = MultiResSTFTLoss(device=device)
    
    # Loop
    losses_def = []
    losses_opt = []
    
    for item in tqdm(val_data):
        midi = torch.tensor([item['midi']], device=device).float()
        vel = torch.tensor([item['vel']], device=device).float()
        target = item['audio'].to(device).unsqueeze(0) # [1, T]
        
        # Render Function
        def render(ov):
            phys = calculate_partials(midi, vel, ov, n_partials=64, device=device)
            return diff_piano_render(
                phys["freqs"], phys["tau_s"], phys["tau_f"], phys["amps"], phys["w_curve"],
                dur_samples=target.shape[-1]
            )
            
        # Defaults
        y_def = render(def_overrides)
        l_def = loss_fn(y_def, target).item()
        losses_def.append(l_def)
        
        # Optimized
        y_opt = render(opt_overrides)
        l_opt = loss_fn(y_opt, target).item()
        losses_opt.append(l_opt)
        
    print(f"\nMean Loss (Default):   {np.mean(losses_def):.4f}")
    print(f"Mean Loss (Optimized): {np.mean(losses_opt):.4f}")
    
    diff = np.mean(losses_opt) - np.mean(losses_def)
    print(f"Diff (Opt - Def): {diff:.4f}")
    if diff < 0:
        print("Optimization successfully reduced loss.")
    else:
        print("Optimization FAIL: Loss increased!?")

if __name__ == "__main__":
    main()
