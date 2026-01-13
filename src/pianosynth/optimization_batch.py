import torch
import torch.nn as nn
import json
from pathlib import Path

# Load defaults
DEFAULT_PARAMS_PATH = Path(__file__).parent / "default_params.json"
with open(DEFAULT_PARAMS_PATH, "r") as f:
    DEFAULTS = json.load(f)

class PianoParamPerKey(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.params = nn.ParameterDict()
        self.start_note = 21
        self.end_note = 108
        self.n_keys = self.end_note - self.start_note + 1 # 88
        
        # We need to initialize our FLAT parameters using the original functional defaults
        # so training starts from a good state.
        
        # 1. Helpers for init calculation
        m_all = torch.arange(self.start_note, self.end_note + 1, device=device, dtype=torch.float32)
        m_norm = (m_all - 21.0) / 87.0
        two = torch.tensor(2.0, device=device)
        f_et = 440.0 * torch.pow(two, (m_all - 69.0) / 12.0)
        
        def D(keys):
            # Access nested keys in DEFAULTS
            d = DEFAULTS
            for k in keys.split("."):
                d = d[k]
            return float(d)

        # --- TUNING ---
        # Removed: we assume strict ET.
        
        # Calc f0 for other inits
        f0 = f_et

        # --- UNISON (Detuning Cents) ---
        dc_base = D("unison_detuning.dc_base")
        dc_slope = D("unison_detuning.dc_slope")
        dc = dc_base + dc_slope * m_norm
        self.register_param_sp("unison_detune_cents", dc)

        rnd_base = D("unison_detuning.random_detune_base")
        rnd_slope = D("unison_detuning.random_detune_slope")
        rnd = rnd_base + rnd_slope * m_norm
        self.register_param_sp("unison_random_cents", rnd)

        # --- INHARMONICITY (B_val) ---
        # Re-using the knot interpolation logic
        knots_m = torch.tensor([21, 36, 60, 84, 108], device=device, dtype=torch.float32)
        b_knots_vals = torch.tensor(DEFAULTS["inharmonicity"]["B_knots"], device=device)
        log_b = torch.log10(b_knots_vals + 1e-12)
        out_log_b = torch.zeros_like(m_all)
        for i in range(4):
            m0, m1 = knots_m[i], knots_m[i+1]
            lb0, lb1 = log_b[i], log_b[i+1]
            mask = (m_all >= m0) & (m_all <= m1)
            if mask.any():
                t = (m_all[mask] - m0) / (m1 - m0)
                out_log_b[mask] = lb0 + (lb1 - lb0) * t
        
        B_val = torch.pow(10.0, out_log_b) * D("inharmonicity.B_scale")
        self.register_param_sp("B_val", B_val)
        
        sv = D("inharmonicity.string_variation_std")
        self.register_param_sigmoid("string_variation_std", torch.full_like(m_all, sv))

        # --- DECAY ---
        # Tau (Total decay at fundamental)
        dA = D("decay.A")
        dp = D("decay.p")
        tau = dA * torch.pow(55.0 / f0, dp)
        self.register_param_sp("decay_tau", tau)
        
        # k (stiffness)
        dk0 = D("decay.k0")
        dk1 = D("decay.k1")
        k = dk0 + dk1 * m_norm
        self.register_param_sp("decay_k", k)
        
        # Tau Fast Div
        div = D("decay.tau_fast_divisor")
        self.register_param_sp("tau_fast_divisor", torch.full_like(m_all, div), min_val=1.0)
        
        # --- HAMMER ---
        # Xh
        xh_b = D("strike_point.xh_bass")
        xh_t = D("strike_point.xh_treble")
        xh = xh_b - (xh_b - xh_t) * m_norm
        self.register_param("hammer_xh", xh) # 0-1 constrained inside physics? Physics uses raw xh.
        # Ideally constrained 0-1. Let's use Sigmoid? 
        # Range is small e.g. 0.1, so Sigmoid(x) -> 0.5 default. 
        # Physics clamps it anyway. Let's learn raw.
        
        # Comb
        self.register_param("comb_mix", torch.full_like(m_all, D("hammer.comb_mix")))
        self.register_param("comb_base", torch.full_like(m_all, D("hammer.comb_base")))

        # Tilt (Vel Dependent)
        self.register_param("tilt_base", torch.full_like(m_all, D("hammer.tilt_base")))
        self.register_param("tilt_slope", torch.full_like(m_all, D("hammer.tilt_slope")))
        
        # Lowpass FC (Vel and F0 dependent, but handled per key)
        # fc = fc_min + ...
        # For a single key, f0 is fixed.
        # We simplify to: fc = low + (high-low)*vel^curve
        # Initial fit?
        # fc_low (at vel=0) = fc_min
        # fc_high (at vel=1) = fc_min + (fc_max-fc_min)*1*...
        
        fc_min = D("hammer.fc_min")
        fc_max = D("hammer.fc_max")
        fc_v = D("hammer.fc_v_power")
        fc_f = D("hammer.fc_f_power")
        
        # Calculated Low/High for this key
        # Term T_f = (f0/261)^f
        T_f = torch.pow(f0 / 261.63, fc_f)
        
        fc_low_init = torch.full_like(m_all, fc_min)
        fc_high_init = fc_min + (fc_max - fc_min) * 1.0 * T_f
        
        self.register_param_sp("hammer_fc_low", fc_low_init, min_val=20.0)
        self.register_param_sp("hammer_fc_high", fc_high_init, min_val=100.0)
        self.register_param_sp("hammer_fc_v_curve", torch.full_like(m_all, fc_v))

        # Nw
        self.register_param_sp("nw_base", torch.full_like(m_all, D("hammer.nw_base")))
        self.register_param_sp("nw_slope", torch.full_like(m_all, D("hammer.nw_slope")))
        
        # --- BODY ---
        self.register_param_sp("highpass_freq", torch.full_like(m_all, D("body_filter.highpass_freq")), min_val=20.0)
        self.register_param_sp("highpass_power", torch.full_like(m_all, D("body_filter.highpass_power")))
        self.register_param_sp("lowpass_freq", torch.full_like(m_all, D("body_filter.lowpass_freq")), min_val=100.0)
        self.register_param_sp("lowpass_power", torch.full_like(m_all, D("body_filter.lowpass_power")))
        
        # --- PROMPT ---
        nmix_b = D("prompt_sound.n_wmix_base")
        nmix_s = D("prompt_sound.n_wmix_slope")
        nmix = nmix_b + nmix_s * m_norm
        self.register_param_sp("prompt_n_mix", nmix)
        
        self.register_param("w_min", torch.full_like(m_all, D("prompt_sound.w_min")))
        self.register_param("w_max", torch.full_like(m_all, D("prompt_sound.w_max")))
        
        # --- AFTERSOUND --- 
        # Sticking to defaults passed through
        # ... (skipped for brevity/irrelevance for main tone?) 
        # Let's just create them.
        for k in ["fc_base", "fc_f_power", "scaler_base", "scaler_v", "v_power"]:
             self.register_param("aftersound_" + k, torch.full_like(m_all, D("aftersound."+k)))
             
        # --- REVERB ---
        # Reverb is global usually? Or per key? 
        # User wants per-key EVERYTHING.
        self.register_param_sigmoid("reverb_wet", torch.full_like(m_all, D("reverb.wet")))
        self.register_param_sp("reverb_decay", torch.full_like(m_all, D("reverb.decay")), min_val=0.05)


    def register_param(self, name, tensor_val):
        self.params[name] = nn.Parameter(tensor_val)
        
    def register_param_sp(self, name, tensor_val, min_val=0.0):
        # Inverse Stable
        target = torch.maximum(tensor_val - min_val, torch.tensor(1e-9, device=self.device))
        
        init = torch.where(
            target > 20.0,
            target,
            torch.log(torch.exp(target) - 1.0)
        )
        self.params[name] = nn.Parameter(init)
        
    def register_param_sigmoid(self, name, tensor_val):
        target = torch.clamp(tensor_val, 0.001, 0.999)
        self.params[name] = nn.Parameter(torch.logit(target))

    def forward(self, midi):
        indices = (midi - self.start_note).long().clamp(0, self.n_keys - 1)
        
        overrides = {}
        def P(n): return self.params[n][indices]
        
        # Special logic for overrides construction
        ov = overrides
        # Tuning
        # ov["tuning_offset_cents"] = P("tuning_offset_cents") # Removed
        ov["unison_detune_cents"] = torch.nn.functional.softplus(P("unison_detune_cents")) # sp constraint
        ov["unison_random_cents"] = torch.nn.functional.softplus(P("unison_random_cents"))
        
        ov["B_val"] = torch.nn.functional.softplus(P("B_val"))
        ov["string_variation_std"] = 0.05 * torch.sigmoid(P("string_variation_std"))
        
        ov["decay_tau"] = torch.nn.functional.softplus(P("decay_tau"))
        ov["decay_k"] = torch.nn.functional.softplus(P("decay_k"))
        ov["tau_fast_divisor"] = torch.nn.functional.softplus(P("tau_fast_divisor")) + 1.0
        
        ov["hammer_xh"] = P("hammer_xh").clamp(0.01, 0.5)
        ov["comb_mix"] = P("comb_mix")
        ov["comb_base"] = P("comb_base")
        ov["tilt_base"] = P("tilt_base")
        ov["tilt_slope"] = P("tilt_slope")
        
        ov["hammer_fc_low"] = torch.nn.functional.softplus(P("hammer_fc_low")) + 20.0
        ov["hammer_fc_high"] = torch.nn.functional.softplus(P("hammer_fc_high")) + 100.0
        ov["hammer_fc_v_curve"] = torch.nn.functional.softplus(P("hammer_fc_v_curve"))
        
        ov["nw_base"] = torch.nn.functional.softplus(P("nw_base"))
        ov["nw_slope"] = torch.nn.functional.softplus(P("nw_slope"))
        
        ov["highpass_freq"] = torch.nn.functional.softplus(P("highpass_freq")) + 20.0
        ov["highpass_power"] = torch.nn.functional.softplus(P("highpass_power"))
        ov["lowpass_freq"] = torch.nn.functional.softplus(P("lowpass_freq")) + 100.0
        ov["lowpass_power"] = torch.nn.functional.softplus(P("lowpass_power"))
        
        ov["prompt_n_mix"] = torch.nn.functional.softplus(P("prompt_n_mix"))
        ov["w_min"] = P("w_min").clamp(0,1)
        ov["w_max"] = P("w_max").clamp(0,1)
        
        # Aftersound (mapped flat)
        ov["aftersound.fc_base"] = P("aftersound_fc_base") # Raw?
        # ... logic if needed, but we mostly ignore aftersound optimization for now
        
        ov["reverb_wet"] = torch.sigmoid(P("reverb_wet"))
        ov["reverb_decay"] = torch.nn.functional.softplus(P("reverb_decay")) + 0.05
        
        return overrides
