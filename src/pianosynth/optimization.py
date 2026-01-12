import torch
import torch.nn as nn
import json
from pathlib import Path

# Load defaults for initialization
DEFAULT_PARAMS_PATH = Path(__file__).parent / "default_params.json"
with open(DEFAULT_PARAMS_PATH, "r") as f:
    DEFAULTS = json.load(f)

class PianoParam(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.params = nn.ParameterDict()
        
        # Load all scalar params from JSON defaults
        self._init_scalars(DEFAULTS)
        
        # Special case: B_knots is a list, needs specific tensor handling
        # Remove it from scalars if added, or handle explicitly
        if "inharmonicity.B_knots" in self.params:
             del self.params["inharmonicity.B_knots"]
             
        # Initialize B_knots explicitly
        b_knots_def = DEFAULTS["inharmonicity"]["B_knots"]
        # We learn log10(B)
        self.logB_knots = nn.Parameter(torch.log10(torch.tensor(b_knots_def, device=device)))
        self.midi_knots = torch.tensor([21, 36, 60, 84, 108], device=device, dtype=torch.float32)

    def _init_scalars(self, d, prefix=""):
        """ Recursively find floats and create nn.Parameters """
        for k, v in d.items():
            if isinstance(v, dict):
                self._init_scalars(v, prefix + k + ".")
            elif isinstance(v, (float, int)):
                # Flattened key: e.g. "tuning.railsback_gamma"
                # Some params must be positive, others unconstrained.
                # We'll apply constraints in forward/get_params, here just store latent.
                # Initialize latent such that softplus(latent) ~= default if positive required.
                # For simplicity, optimize strictly *latent* values initialized to inverse_softplus(default) 
                # or just raw default if using direct value.
                # Let's assume we optimize multipliers or offsets? 
                # Easier: Just optimize the raw value, apply softplus constraints where needed used in forward.
                
                # Rule: params > 0 should use softplus.
                # Params that can be negative (like stretch_bass_amount) use raw.
                
                name = prefix + k
                val_tensor = torch.tensor(float(v), device=self.device)
                
                # Heuristic: if default > 0, assume positive constraint wanted -> Softplus
                # except stretches which are explicitly negative.
                # Let's create a parameter for each.
                # We'll store the RAW value here.
                # Ideally we want init value to equal default.
                # If we apply softplus later, init = inverse_softplus(default).
                # inverse_softplus(y) = log(exp(y) - 1).
                
                # Groups to constrain positive:
                # decay.*, hammer.* (except slopes?), strike_point.*, body_filter.*, aftersound.fc*, tuning.gamma
                
                # Groups signed:
                # tuning.stretch*, unison.dc* (maybe?), hammer.tilt_slope?
                
                # Let's keep it simple: Optimize everything as raw, constrain explicitly in get_flat_params
                self.params[name.replace(".", "_")] = nn.Parameter(val_tensor)

    def get_B(self, midi_vals):
        """ Interpolate B(midi) """
        logB = torch.zeros_like(midi_vals)
        for i in range(len(self.midi_knots) - 1):
            m0 = self.midi_knots[i]
            m1 = self.midi_knots[i+1]
            b0 = self.logB_knots[i]
            b1 = self.logB_knots[i+1]
            mask = (midi_vals >= m0) & (midi_vals <= m1)
            if mask.any():
                t = (midi_vals[mask] - m0) / (m1 - m0)
                logB[mask] = b0 + (b1 - b0) * t
        return torch.pow(10.0, logB)

    def get_overrides(self):
        """ Returns dictionary of constrained parameters for synth.py """
        overrides = {}
        
        # Helper to get param
        def P(name):
            key = name.replace(".", "_")
            return self.params[key]
            
        # Helper for Softplus
        def SP(name, min_val=0.0):
            val = P(name)
            # Softplus shift to keep init close to default
            # Actually, standard softplus shifts everything > 0.
            # If default is 8.0, raw 8.0 -> softplus(8) ~ 8.
            # If default is 0.03, softplus(0.03) ~ 0.7 (bad).
            # We skip inverse logic for now and just rely on gradient descent to find range.
            # But we must ensure crucial params don't start near 0.
            return torch.nn.functional.softplus(val) + min_val

        # --- Populate overrides ---
        # Match keys expected by synth.py's get_param
        
        # Tuning
        overrides["railsback_gamma"] = SP("tuning.railsback_gamma")
        overrides["stretch_bass_amount"] = P("tuning.stretch_bass_amount") # Negative allowed
        overrides["stretch_bass_range"] = SP("tuning.stretch_bass_range", 1.0)
        overrides["stretch_treble_amount"] = P("tuning.stretch_treble_amount")
        overrides["stretch_treble_range"] = SP("tuning.stretch_treble_range", 1.0)
        
        # Inharmonicity
        overrides["B_knots"] = torch.pow(10.0, self.logB_knots)
        overrides["B_scale"] = SP("inharmonicity.B_scale")
        # Constrain variation to max 5%
        overrides["string_variation_std"] = 0.05 * torch.sigmoid(P("inharmonicity.string_variation_std"))
        
        # Strike Point
        overrides["xh_bass"] = P("strike_point.xh_bass").clamp(0.01, 0.5) 
        overrides["xh_treble"] = P("strike_point.xh_treble").clamp(0.01, 0.5)
        
        # Unison
        overrides["dc_base"] = SP("unison_detuning.dc_base")
        overrides["dc_slope"] = P("unison_detuning.dc_slope")
        overrides["random_detune_base"] = SP("unison_detuning.random_detune_base")
        overrides["random_detune_slope"] = SP("unison_detuning.random_detune_slope")
        
        # Hammer
        overrides["comb_mix"] = P("hammer.comb_mix")
        overrides["comb_base"] = P("hammer.comb_base")
        overrides["tilt_base"] = P("hammer.tilt_base")
        overrides["tilt_slope"] = P("hammer.tilt_slope")
        overrides["fc_min"] = SP("hammer.fc_min", 100.0)
        overrides["fc_max"] = SP("hammer.fc_max", 1000.0)
        overrides["fc_v_power"] = SP("hammer.fc_v_power")
        overrides["fc_f_power"] = SP("hammer.fc_f_power")
        overrides["nw_base"] = SP("hammer.nw_base")
        overrides["nw_slope"] = SP("hammer.nw_slope")
        
        # Body
        overrides["highpass_freq"] = SP("body_filter.highpass_freq", 20.0)
        overrides["highpass_power"] = SP("body_filter.highpass_power")
        overrides["lowpass_freq"] = SP("body_filter.lowpass_freq", 100.0)
        overrides["lowpass_power"] = SP("body_filter.lowpass_power")
        
        # Decay
        overrides["A"] = SP("decay.A", 0.1)
        overrides["p"] = P("decay.p") # Can be negative (as seen!)
        overrides["tau_fast_divisor"] = SP("decay.tau_fast_divisor", 1.0)
        overrides["k0"] = SP("decay.k0")
        overrides["k1"] = SP("decay.k1")
        
        # Prompt
        overrides["n_wmix_base"] = SP("prompt_sound.n_wmix_base")
        overrides["n_wmix_slope"] = SP("prompt_sound.n_wmix_slope")
        overrides["w_min"] = P("prompt_sound.w_min").clamp(0., 1.)
        overrides["w_max"] = P("prompt_sound.w_max").clamp(0., 1.)
        
        # Aftersound
        overrides["fc_base"] = SP("aftersound.fc_base", 100.0)
        overrides["fc_f_power"] = P("aftersound.fc_f_power")
        overrides["scaler_base"] = SP("aftersound.scaler_base")
        overrides["scaler_v"] = SP("aftersound.scaler_v")
        overrides["v_power"] = SP("aftersound.v_power")
        
        # Attack
        overrides["rise_time_const"] = SP("attack.rise_time_const", 1e-4)

        return overrides

    # Keeping forward for compatibility but ideally we use overrides directly into synth
    # But train_spectral.py calls model(m_t, vs) -> freqs, decays, amps
    # We need to reimplement the logic that maps params -> partials HERE
    # OR we make diff_piano_render take the overrides dict and do the physics inside.
    # The latter is much cleaner and avoids duplicating "physics code".
    
    # Wait, train_spectral.py calls:
    # freqs, decays, amps = model(m_t, vs)
    # y_pred = diff_piano_render(freqs, decays, amps, ...)
    
    # We should pivot: train_spectral.py calls model.get_overrides()
    # then passes that to a diff_piano_render_v2 that takes (midi, vel, overrides).
    # THIS ELIMINATES DUPLICATION.
    
    def forward(self):
        # Just return overrides
        return self.get_overrides()
