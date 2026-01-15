import torch
import json
import os
from pathlib import Path

# Load defaults once
DEFAULT_PARAMS_PATH = Path(__file__).parent / "default_params.json"
with open(DEFAULT_PARAMS_PATH, "r") as f:
    EPS_DEFAULTS = json.load(f)

def get_param(name, category, override, default_struct):
    """
    Helper to get param from override dict (flat) or default struct (nested).
    Flattened keys in override: e.g. "tuning.railsback_gamma" or just unique names if unique.
    We'll stick to unique names for now based on previous optimization code, 
    but for full config we might need dot notation or just flatness.
    
    Let's assume overrides are FLATTENED keys that match leaf names in JSON.
    """
    if override and name in override:
        return override[name]
    return default_struct[category][name]

def piano_additive(
    midi: int,
    velocity: float = 0.7,
    dur: float = 2.5,
    sr: int = 44100,
    max_partials: int = 96,
    seed: int | None = 0,
    device=None,
    dtype=torch.float32,
    normalize: str | None = "peak",
    **kwargs
) -> torch.Tensor:
    
    override = kwargs.get("params_override", {})
    # Ensure override values are tensors on device if they aren't already
    # Actually, optimization loop passes tensors. 
    # But if passing manual floats, we might need conversion.
    # We'll assume caller handles tensor-ness if gradients required.
    
    def P(name, category):
        val = get_param(name, category, override, EPS_DEFAULTS)
        if isinstance(val, (float, int, list)):
             return torch.tensor(val, device=device, dtype=dtype)
        return val # Assume already tensor

    device = device or "cpu"
    midi_i = int(max(21, min(108, int(midi))))
    m = torch.tensor(float(midi_i), device=device, dtype=dtype)
    v = torch.tensor(float(velocity), device=device, dtype=dtype).clamp(0.0, 1.0)
    two = torch.tensor(2.0, device=device, dtype=dtype)

    # --- Tuning ---
    gamma = P("railsback_gamma", "tuning")
    stretch_bass_amt = P("stretch_bass_amount", "tuning")
    stretch_bass_rng = P("stretch_bass_range", "tuning")
    stretch_treble_amt = P("stretch_treble_amount", "tuning")
    stretch_treble_rng = P("stretch_treble_range", "tuning")
    
    f_et = 440.0 * torch.pow(two, (m - 69.0) / 12.0)
    
    d_cents = torch.where(
        m <= 69.0,
        stretch_bass_amt * torch.pow((69.0 - m) / stretch_bass_rng, gamma),
        stretch_treble_amt * torch.pow((m - 69.0) / stretch_treble_rng, gamma),
    )
    f0 = f_et * torch.pow(two, d_cents / 1200.0)

    # --- Inharmonicity ---
    B_knots = P("B_knots", "inharmonicity")
    
    # Check if knots are log10 or linear in override
    # If it came from optimization, it might be log10 if we didn't unwrap it.
    # But let's assume valid 'B' values are passed (linear) for consistency with JSON.
    # If the user passes optimized logB, they should convert before calling this.
    
    m_knots = [21, 36, 60, 84, 108]
    idx = 0 if midi_i < 36 else (1 if midi_i < 60 else (2 if midi_i < 84 else 3))
    m0 = torch.tensor(float(m_knots[idx]), device=device, dtype=dtype)
    m1 = torch.tensor(float(m_knots[idx + 1]), device=device, dtype=dtype)
    
    # Interpolate in log domain
    logB0 = torch.log10(B_knots[idx] + 1e-12)
    logB1 = torch.log10(B_knots[idx + 1] + 1e-12)
       
    logB = logB0 + (logB1 - logB0) * (m - m0) / (m1 - m0)
    B = torch.pow(torch.tensor(10.0, device=device, dtype=dtype), logB)

    # --- Strike Point ---
    xh_bass = P("xh_bass", "strike_point")
    xh_treble = P("xh_treble", "strike_point")
    xh = xh_bass - (xh_bass - xh_treble) * (m - 21.0) / 87.0

    # --- Unison ---
    dc_base = P("dc_base", "unison_detuning")
    dc_slope = P("dc_slope", "unison_detuning")
    dc = dc_base + dc_slope * (m - 21.0) / 87.0
    
    if midi_i <= 28:
        base_c = torch.tensor([0.0], device=device, dtype=dtype)
    elif midi_i <= 40:
        base_c = torch.stack([-0.5 * dc, 0.5 * dc])
    else:
        base_c = torch.stack([-dc, torch.tensor(0.0, device=device, dtype=dtype), dc])
    S = int(base_c.numel())

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(seed))

    rnd_base = P("random_detune_base", "unison_detuning")
    rnd_slope = P("random_detune_slope", "unison_detuning")

    if gen is not None and S > 1:
        c = base_c + (rnd_base + rnd_slope * (m - 21.0) / 87.0) * torch.randn((S,), generator=gen, device=device, dtype=dtype)
    else:
        c = base_c

    # --- Per-string Inharmonicity ---
    B_scale = P("B_scale", "inharmonicity")
    B_var = P("string_variation_std", "inharmonicity")
    
    if gen is not None and S > 1:
        Bj = 1.0 + B_var * torch.randn((S,), generator=gen, device=device, dtype=dtype)
    else:
        Bj = torch.ones((S,), device=device, dtype=dtype)
    B_s = torch.clamp(B * B_scale * Bj, min=1e-7)

    # --- Partial Count ---
    nyq = 0.5 * sr
    f0_max = f0 * torch.pow(two, (c.abs().max() / 1200.0))
    B_max = B_s.max()
    R = nyq / f0_max
    # Simple limit
    N = int(max_partials)
    # Check Nyquist with inharmonicity
    # f_n approx n f0 sqrt(1 + B n^2) < nyq
    # Just iterate/solve? Or use loose bound.
    # B n^2 < (nyq/nf0)^2 - 1
    # n f0 sqrt(B) n ~ nyq -> B n^3 f0 ~ nyq -> n ~ (nyq/f0 B^0.5)^(1/3) ??
    # For speed, just clamp freqs later.
    
    n = torch.arange(1, N + 1, device=device, dtype=dtype)
    n2 = n * n

    # String Freqs
    f0_s = f0 * torch.pow(two, c[:, None] / 1200.0)
    fn_s = n[None, :] * f0_s * torch.sqrt(1.0 + B_s[:, None] * n2[None, :])
    alias = (fn_s < nyq).to(dtype)

    # Ref Freqs
    fn = n * f0 * torch.sqrt(1.0 + (B * B_scale) * n2)

    # --- Excitation ---
    comb_mix = P("comb_mix", "hammer")
    comb_base = P("comb_base", "hammer")
    comb = comb_base + comb_mix * torch.sin(torch.pi * n * xh).pow(2)

    # Tilt
    tilt_base = P("tilt_base", "hammer")
    tilt_slope = P("tilt_slope", "hammer")
    p = tilt_base - tilt_slope * v
    # Enforce positivity? The optimization logic handles softplus. 
    # Here we just take value.
    tilt = torch.pow(n, -p)

    fc_min = P("fc_min", "hammer")
    fc_max = P("fc_max", "hammer")
    fc_v = P("fc_v_power", "hammer")
    fc_f = P("fc_f_power", "hammer")
    
    fc = fc_min + (fc_max - fc_min) * torch.pow(v, fc_v) * torch.pow(f0 / 261.63, fc_f)
    fc = torch.clamp(fc, 700.0, 14000.0) # These limits could be params too, but kept safe.
    H_hammer = torch.rsqrt(1.0 + torch.pow(fn / fc, 4.0))

    nw_base = P("nw_base", "hammer")
    nw_slope = P("nw_slope", "hammer")
    n_w = nw_base + nw_slope * v
    W = torch.exp(-torch.pow(n / n_w, 2.0))

    # --- Body ---
    hp_f = P("highpass_freq", "body_filter")
    hp_p = P("highpass_power", "body_filter")
    lp_f = P("lowpass_freq", "body_filter")
    lp_p = P("lowpass_power", "body_filter")
    
    H_body = torch.pow(fn / hp_f, hp_p) * torch.rsqrt(1.0 + torch.pow(fn / lp_f, lp_p))

    A0 = torch.pow(v, 1.7) * comb * tilt * H_hammer * W * H_body

    # --- Decays ---
    decay_A = P("A", "decay")
    decay_p = P("p", "decay")
    tau_s0 = decay_A * torch.pow(torch.tensor(55.0, device=device, dtype=dtype) / f0, decay_p)
    
    div = P("tau_fast_divisor", "decay")
    tau_f0 = tau_s0 / div
    
    k0 = P("k0", "decay")
    k1 = P("k1", "decay")
    k = k0 + k1 * (m - 21.0) / 87.0
    
    tau_s = tau_s0 / (1.0 + k * n2)
    tau_f = tau_f0 / (1.0 + 4.0 * k * n2)

    # --- Prompt Mix ---
    nmix_base = P("n_wmix_base", "prompt_sound")
    nmix_slope = P("n_wmix_slope", "prompt_sound")
    n_wmix = nmix_base + nmix_slope * (m - 21.0) / 87.0
    
    w_min = P("w_min", "prompt_sound")
    w_max = P("w_max", "prompt_sound")
    w = w_min + (w_max - w_min) * (1.0 - torch.exp(-torch.pow(n / n_wmix, 2.0)))

    # --- Aftersound ---
    fc_base_sus = P("fc_base", "aftersound")
    fc_f_pow_sus = P("fc_f_power", "aftersound")
    sc_base_sus = P("scaler_base", "aftersound")
    sc_v_sus = P("scaler_v", "aftersound")
    v_pow_sus = P("v_power", "aftersound")
    
    fc_sus = fc_base_sus * torch.pow(f0 / 261.63, fc_f_pow_sus) * (sc_base_sus + sc_v_sus * torch.pow(v, v_pow_sus))
    fc_sus = torch.clamp(fc_sus, 650.0, 3200.0)
    H_sus = torch.rsqrt(1.0 + torch.pow(fn / fc_sus, 4.0))

    # --- Time Envelope ---
    T = int(sr * dur)
    t = torch.arange(T, device=device, dtype=dtype) / sr
    
    rise_tc = P("rise_time_const", "attack")
    Rrise = 1.0 - torch.exp(-t / (rise_tc * (0.8 + 0.8 * (1.0 - v))))

    E_fast = torch.exp(-t[None, :] / tau_f[:, None])
    E_slow = torch.exp(-t[None, :] / tau_s[:, None])

    A_fast = A0 * w
    A_slow = A0 * (1.0 - w) * H_sus

    if gen is None:
        phi = torch.zeros((S, N), device=device, dtype=dtype)
    else:
        phi = 2.0 * torch.pi * torch.rand((S, N), generator=gen, device=device, dtype=dtype) * 0.15

    y = ((A_fast[None, :, None] * E_fast[None, :, :] + A_slow[None, :, None] * E_slow[None, :, :]) *
         torch.sin(2.0 * torch.pi * fn_s[:, :, None] * t[None, None, :] + phi[:, :, None]) *
         alias[:, :, None]
        ).sum(dim=(0, 1))

    y = Rrise * y

    if normalize == "peak":
        y = y / (y.abs().max() + 1e-12)
    return y

def synthesize_from_params(freqs, decays, amps, dur_samples=None, dur_sec=2.5, sr=44100, normalize="peak"):
    # Re-implement using diff_piano_render logic or just keep as simple ref
    # Ideally reuse diff_piano_render from spectral?
    # Keeping simple for now as it's just for quick checks.
    pass
