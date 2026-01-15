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
        self.n_dyns = 3 # pp, mf, ff
        
        # Helper for defaults
        m_all = torch.arange(self.start_note, self.end_note + 1, device=device, dtype=torch.float32)
        m_norm = (m_all - 21.0) / 87.0
        two = torch.tensor(2.0, device=device)
        f_et = 440.0 * torch.pow(two, (m_all - 69.0) / 12.0)
        
        def D(keys):
            d = DEFAULTS
            for k in keys.split("."):
                d = d[k]
            return float(d)

        # Standard Velocities for Initialization
        # pp=0.25, mf=0.5, ff=0.75
        vels = torch.tensor([0.25, 0.5, 0.75], device=device).view(1, 3) # Broadcastable
        
        # Helper to expand scalar/1D to (88, 3)
        def expand_init(val_ft):
             # val_ft: FloatTensor of shape (88) or scalar
             if val_ft.ndim == 0:
                 val_ft = val_ft.unsqueeze(0).expand(self.n_keys)
             # Must clone to ensure distinct memory (no stride 0 overlap) for optimizer
             return val_ft.unsqueeze(1).expand(self.n_keys, self.n_dyns).clone()

        # --- AMPLITUDE ---
        # amp = v^1.7
        amp_init = torch.pow(vels, 1.7) # (1, 3)
        # expand(self.n_keys, 3) would imply shared memory across keys. 
        # We must clone.
        self.register_param_sp("amplitude", amp_init.expand(self.n_keys, 3).clone())

        # --- TUNING (Unison) ---
        dc_base = D("unison_detuning.dc_base")
        dc_slope = D("unison_detuning.dc_slope")
        dc = dc_base + dc_slope * m_norm
        self.register_param_sp("unison_detune_cents", expand_init(dc))

        rnd_base = D("unison_detuning.random_detune_base")
        rnd_slope = D("unison_detuning.random_detune_slope")
        rnd = rnd_base + rnd_slope * m_norm
        self.register_param_sp("unison_random_cents", expand_init(rnd))

        # --- INHARMONICITY ---
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
        self.register_param_sp("B_val", expand_init(B_val))
        
        sv = D("inharmonicity.string_variation_std")
        self.register_param_sigmoid("string_variation_std", expand_init(torch.full_like(m_all, sv)))

        # --- DECAY ---
        dA = D("decay.A")
        dp = D("decay.p")
        tau = dA * torch.pow(55.0 / f_et, dp)
        self.register_param_sp("decay_tau", expand_init(tau))
        
        dk0 = D("decay.k0")
        dk1 = D("decay.k1")
        k = dk0 + dk1 * m_norm
        self.register_param_sp("decay_k", expand_init(k))
        
        div = D("decay.tau_fast_divisor")
        self.register_param_sp("tau_fast_divisor", expand_init(torch.full_like(m_all, div)), min_val=1.0)
        
        # --- HAMMER ---
        # Xh
        xh_b = D("strike_point.xh_bass")
        xh_t = D("strike_point.xh_treble")
        xh = xh_b - (xh_b - xh_t) * m_norm
        self.register_param("hammer_xh", expand_init(xh)) 

        # Comb
        self.register_param("comb_mix", expand_init(torch.full_like(m_all, D("hammer.comb_mix"))))
        self.register_param("comb_base", expand_init(torch.full_like(m_all, D("hammer.comb_base"))))

        # Tilt (Vel Dependent)
        # p = base - slope * v
        t_base = torch.full_like(m_all, D("hammer.tilt_base")).unsqueeze(1)
        t_slope = torch.full_like(m_all, D("hammer.tilt_slope")).unsqueeze(1)
        p_tilt = t_base - t_slope * vels
        self.register_param_sp("hammer_p_tilt", p_tilt.expand(self.n_keys, 3).clone()) # Note name change: tilt_base/slope -> hammer_p_tilt
        
        # Hammer FC (Vel Dependent)
        fc_min = D("hammer.fc_min")
        fc_max = D("hammer.fc_max")
        fc_v = D("hammer.fc_v_power")
        fc_f = D("hammer.fc_f_power")
        
        # T_f = (f0/261)^f
        T_f = torch.pow(f_et / 261.63, fc_f).unsqueeze(1)
        
        fc_low = fc_min
        fc_high = fc_min + (fc_max - fc_min) * 1.0 * T_f # approx init high
        # fc = low + (high-low)*v^curve
        # We simplify init: fc = fc_min + (fc_max-fc_min)*v^fc_v * (f0/261)^fc_f
        # This matches the legacy "global" formula used in physics breakdown:
        # fc = min + (max-min) * v^v_pow * f_factor
        
        # Re-calculating global init per note/vel
        delta = (fc_max - fc_min)
        fc_init = fc_min + delta * torch.pow(vels, fc_v) * T_f
        self.register_param_sp("hammer_fc", fc_init.expand(self.n_keys, 3).clone(), min_val=20.0)

        # Nw (Vel Dependent)
        nw_b = D("hammer.nw_base")
        nw_s = D("hammer.nw_slope")
        nw = nw_b + nw_s * vels
        self.register_param_sp("hammer_nw", nw.expand(self.n_keys, 3).clone()) # Name change: -> hammer_nw

        # --- BODY ---
        self.register_param_sp("highpass_freq", expand_init(torch.full_like(m_all, D("body_filter.highpass_freq"))), min_val=20.0)
        self.register_param_sp("highpass_power", expand_init(torch.full_like(m_all, D("body_filter.highpass_power"))))
        self.register_param_sp("lowpass_freq", expand_init(torch.full_like(m_all, D("body_filter.lowpass_freq"))), min_val=100.0)
        self.register_param_sp("lowpass_power", expand_init(torch.full_like(m_all, D("body_filter.lowpass_power"))))
        
        # --- PROMPT ---
        nmix_b = D("prompt_sound.n_wmix_base")
        nmix_s = D("prompt_sound.n_wmix_slope")
        nmix = nmix_b + nmix_s * m_norm
        self.register_param_sp("prompt_n_mix", expand_init(nmix))
        
        self.register_param("w_min", expand_init(torch.full_like(m_all, D("prompt_sound.w_min"))))
        self.register_param("w_max", expand_init(torch.full_like(m_all, D("prompt_sound.w_max"))))
        
        # --- AFTERSOUND --- 
        for k in ["fc_base", "fc_f_power", "scaler_base", "scaler_v", "v_power"]:
             self.register_param("aftersound_" + k, expand_init(torch.full_like(m_all, D("aftersound."+k))))
             
        # --- REVERB ---
        self.register_param_sigmoid("reverb_wet", expand_init(torch.full_like(m_all, D("reverb.wet"))))
        self.register_param_sp("reverb_decay", expand_init(torch.full_like(m_all, D("reverb.decay"))), min_val=0.05)


    def register_param(self, name, tensor_val):
        # tensor_val should be (88, 3)
        self.params[name] = nn.Parameter(tensor_val)
        
    def register_param_sp(self, name, tensor_val, min_val=0.0):
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

    def forward(self, midi, dyn_indices):
        """
        midi: [Batch] int tensor of midi numbers (21-108)
        dyn_indices: [Batch] int tensor of indices (0=pp, 1=mf, 2=ff)
        """
        indices = (midi - self.start_note).long().clamp(0, self.n_keys - 1)
        dyns = dyn_indices.long().clamp(0, self.n_dyns - 1)
        
        # Select: self.params[name][indices, dyns]
        def P(n): return self.params[n][indices, dyns]
        
        ov = {}
        
        ov["amplitude"] = torch.nn.functional.softplus(P("amplitude"))
        ov["unison_detune_cents"] = torch.nn.functional.softplus(P("unison_detune_cents"))
        ov["unison_random_cents"] = torch.nn.functional.softplus(P("unison_random_cents"))
        
        ov["B_val"] = torch.nn.functional.softplus(P("B_val"))
        ov["string_variation_std"] = 0.05 * torch.sigmoid(P("string_variation_std"))
        
        ov["decay_tau"] = torch.nn.functional.softplus(P("decay_tau"))
        ov["decay_k"] = torch.nn.functional.softplus(P("decay_k"))
        ov["tau_fast_divisor"] = torch.nn.functional.softplus(P("tau_fast_divisor")) + 1.0
        
        ov["hammer_xh"] = P("hammer_xh").clamp(0.01, 0.5)
        ov["comb_mix"] = P("comb_mix")
        ov["comb_base"] = P("comb_base")
        
        # New explicit params
        ov["hammer_p_tilt"] = torch.nn.functional.softplus(P("hammer_p_tilt"))
        ov["hammer_fc"] = torch.nn.functional.softplus(P("hammer_fc")) + 20.0
        ov["hammer_nw"] = torch.nn.functional.softplus(P("hammer_nw"))
        
        ov["highpass_freq"] = torch.nn.functional.softplus(P("highpass_freq")) + 20.0
        ov["highpass_power"] = torch.nn.functional.softplus(P("highpass_power"))
        ov["lowpass_freq"] = torch.nn.functional.softplus(P("lowpass_freq")) + 100.0
        ov["lowpass_power"] = torch.nn.functional.softplus(P("lowpass_power"))
        
        ov["prompt_n_mix"] = torch.nn.functional.softplus(P("prompt_n_mix"))
        ov["w_min"] = P("w_min").clamp(0,1)
        ov["w_max"] = P("w_max").clamp(0,1)
        
        # Aftersound 
        ov["aftersound.fc_base"] = P("aftersound_fc_base")
        
        ov["reverb_wet"] = torch.sigmoid(P("reverb_wet"))
        ov["reverb_decay"] = torch.nn.functional.softplus(P("reverb_decay")) + 0.05
        
        return ov
