import torch

def get_b_vectorized(midi, b_knots, device):
    """ Vectorized B interpolation """
    # b_knots: tensor of length 5 (linear values) -> we interpolate log10
    # midi: [Batch]
    knots_m = torch.tensor([21, 36, 60, 84, 108], device=device, dtype=torch.float32)
    log_b = torch.log10(b_knots + 1e-12)
    
    out_log_b = torch.zeros_like(midi)
    
    # Segmented interpolation
    for i in range(4):
        m0, m1 = knots_m[i], knots_m[i+1]
        lb0, lb1 = log_b[i], log_b[i+1]
        
        mask = (midi >= m0) & (midi <= m1)
        if mask.any():
            t = (midi[mask] - m0) / (m1 - m0)
            out_log_b[mask] = lb0 + (lb1 - lb0) * t
            
    # Handle out of bounds (clamp to ends)
    out_log_b[midi < 21] = log_b[0]
    out_log_b[midi > 108] = log_b[-1]
    
    return torch.pow(10.0, out_log_b)

def calculate_partials(midi, velocity, overrides, n_partials=64, device="cpu"):
    """
    Vectorized physics calculator.
    midi: [Batch] or scalar
    velocity: [Batch] or scalar
    overrides: dict from PianoParam.get_overrides()
    
    Returns:
        freqs: [Batch, N]
        decays: [Batch, N]
        amps: [Batch, N]
    """
    # Ensure inputs are tensors [Batch]
    if not torch.is_tensor(midi):
        midi = torch.tensor([midi], device=device).float()
    if midi.ndim == 0: midi = midi.unsqueeze(0)
    
    if not torch.is_tensor(velocity):
        velocity = torch.tensor([velocity], device=device).float()
    if velocity.ndim == 0: velocity = velocity.unsqueeze(0)
    
    B_size = midi.shape[0]
    
    # Helpers
    two = torch.tensor(2.0, device=device)
    n = torch.arange(1, n_partials + 1, device=device).float() # [N]
    
    # Resize n for broadcasting: [1, N]
    n = n.unsqueeze(0)
    
    # Unwrap Overrides
    def P(k): return overrides[k]
    
    # --- Tuning ---
    # f_et
    # stretch curve
    gamma = P("railsback_gamma")
    s_bass_amt = P("stretch_bass_amount")
    s_bass_rng = P("stretch_bass_range")
    s_treb_amt = P("stretch_treble_amount")
    s_treb_rng = P("stretch_treble_range")
    
    f_et = 440.0 * torch.pow(two, (midi - 69.0) / 12.0)
    
    d_cents = torch.where(
        midi <= 69.0,
        s_bass_amt * torch.pow(((69.0 - midi) / s_bass_rng).clamp(min=0), gamma),
        s_treb_amt * torch.pow(((midi - 69.0) / s_treb_rng).clamp(min=0), gamma)
    )
    f0 = f_et * torch.pow(two, d_cents / 1200.0) # [Batch]
    
    # --- Inharmonicity ---
    B_vals = get_b_vectorized(midi, P("B_knots"), device)
    B_scale = P("B_scale")
    B_eff = B_vals * B_scale # [Batch]
    
    # Freqs
    # fn = n * f0 * sqrt(1 + B * n^2)
    # [1, N] * [B, 1] * sqrt(1 + [B,1]*[1,N]^2)
    f0_exp = f0.unsqueeze(1)
    B_exp = B_eff.unsqueeze(1)
    
    fn = n * f0_exp * torch.sqrt(1.0 + B_exp * (n**2))
    
    # --- Decays ---
    # tau_s0 = A * (55/f0)^p
    dA = P("A")
    dp = P("p")
    tau_s0 = dA * torch.pow(55.0 / f0_exp, dp) # [Batch, 1]
    
    # k = k0 + k1 * m_norm
    dk0 = P("k0")
    dk1 = P("k1")
    m_norm = (midi - 21.0) / 87.0
    k = dk0 + dk1 * m_norm # [Batch]
    
    tau_s = tau_s0 / (1.0 + k.unsqueeze(1) * (n**2))
    
    # tau_f = tau_s / div / (1 + 4kn^2) ? 
    # Defaults logic: tau_f0 = tau_s0 / div. tau_f = tau_f0 / (1 + 4k n^2)
    div = P("tau_fast_divisor")
    tau_f = (tau_s0 / div) / (1.0 + 4.0 * k.unsqueeze(1) * (n**2))
    
    # Return JUST tau_s for now? Or effective decay?
    # diff_piano_render took one decay.
    # We should probably return both or blend them into one for the simple renderer.
    # Or update renderer to support double decay.
    # Let's update renderer to support double decay.
    
    # --- Amplitudes ---
    # Comb
    xh_b = P("xh_bass")
    xh_t = P("xh_treble")
    xh = xh_b - (xh_b - xh_t) * m_norm # [Batch]
    
    c_mix = P("comb_mix")
    c_base = P("comb_base")
    # sin(pi * n * xh)
    comb = c_base + c_mix * torch.sin(torch.pi * n * xh.unsqueeze(1)).pow(2)
    
    # Tilt
    t_base = P("tilt_base")
    t_slope = P("tilt_slope")
    p_tilt = t_base - t_slope * velocity # [Batch]
    # Assuming optimization handles constraints, but here we might get negative p_tilt if velocity high.
    # Apply softplus logic? No, optimization.py applied softplus to output of getter.
    # So P("tilt_base") is already softplussed/constrained logic if we used SP helper.
    # Wait, optimization.py: tilt_base is parameter. SP applied in get_overrides.
    # So here p_tilt is safe-ish.
    
    tilt = torch.pow(n, -p_tilt.unsqueeze(1))
    
    # Hammer Lowpass
    fc_min = P("fc_min")
    fc_max = P("fc_max")
    fc_v = P("fc_v_power")
    fc_f = P("fc_f_power")
    
    fc = fc_min + (fc_max - fc_min) * torch.pow(velocity.unsqueeze(1), fc_v) * torch.pow(f0_exp / 261.63, fc_f)
    fc = fc.clamp(700.0, 14000.0)
    
    H_hammer = torch.rsqrt(1.0 + torch.pow(fn / fc, 4.0))
    
    # N_w
    nw_b = P("nw_base")
    nw_s = P("nw_slope")
    n_w = nw_b + nw_s * velocity.unsqueeze(1)
    W = torch.exp(-torch.pow(n / n_w, 2.0))
    
    # Body
    hp_f = P("highpass_freq")
    hp_p = P("highpass_power")
    lp_f = P("lowpass_freq")
    lp_p = P("lowpass_power")
    
    H_body = torch.pow(fn / hp_f, hp_p) * torch.rsqrt(1.0 + torch.pow(fn / lp_f, lp_p))
    
    # Final A0
    # A0 = v^1.7 * ...
    A0 = torch.pow(velocity.unsqueeze(1), 1.7) * comb * tilt * H_hammer * W * H_body
    
    # Prompt W mix (needed for renderer to mix fast/slow)
    nmix_b = P("n_wmix_base")
    nmix_s = P("n_wmix_slope")
    n_wmix = nmix_b + nmix_s * m_norm.unsqueeze(1)
    
    w_min = P("w_min")
    w_max = P("w_max")
    w_curve = w_min + (w_max - w_min) * (1.0 - torch.exp(-torch.pow(n / n_wmix, 2.0)))
    
    return {
        "freqs": fn,
        "tau_s": tau_s,
        "tau_f": tau_f,
        "amps": A0,
        "w_curve": w_curve
    }
