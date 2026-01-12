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
    B_eff = B_vals * B_scale # [Batch] but B_vals is [Batch] if vectorized works?
    # get_b_vectorized returns tensor same shape as midi [Batch]
    
    # Freqs
    # In synth.py: 3 strings per note with detuning.
    # We need to implement this here for the optimizer to learn detuning params.
    # Strings: 1 (Bass), 2 (Mid), 3 (Treble). Simple 3 always for training?
    # Or match synth logic based on MIDI.
    
    # Detuning Params
    dc_base = P("dc_base")
    dc_slope = P("dc_slope")
    dc = dc_base + dc_slope * ((midi - 21.0) / 87.0)
    
    # Random Detuning Params
    rnd_base = P("random_detune_base")
    rnd_slope = P("random_detune_slope")
    rnd_std = rnd_base + rnd_slope * ((midi - 21.0) / 87.0)
    
    # String Variation Params
    B_var_std = P("string_variation_std")

    # Expand to 3 strings: [Batch, 1, 3]
    # Logic from synth.py:
    # <=28: 1 string (0.0)
    # <=40: 2 strings (-0.5dc, 0.5dc)
    # >40: 3 strings (-dc, 0, dc)
    # Vectorized implementation:
    
    c_list = []
    # Fixed detuning offsets [Batch, 3]
    # We'll use 3 dimensions for all, and mask amp later if needed? 
    # Or just replicate synth logic.
    
    # 0 = Center/Single
    # 1 = Left
    # 2 = Right
    
    # Let's produce [Batch, 3] cents deviation
    c_offsets = torch.zeros(B_size, 3, device=device)
    
    # Treble (>40)
    mask_3 = (midi.view(-1) > 40)
    if mask_3.any():
        c_offsets[mask_3, 0] = -dc.view(-1)[mask_3]
        c_offsets[mask_3, 1] = 0.0
        c_offsets[mask_3, 2] = dc.view(-1)[mask_3]
        
    # Mid (<=40 & >28)
    mask_2 = (midi.view(-1) <= 40) & (midi.view(-1) > 28)
    if mask_2.any():
        c_offsets[mask_2, 0] = -0.5 * dc.view(-1)[mask_2]
        c_offsets[mask_2, 1] = 0.5 * dc.view(-1)[mask_2]
        c_offsets[mask_2, 2] = 0.0 # Unused effectively, or 0? 
        # Wait, if we use 3 slots, 2 strings means only 2 are active.
        # Ideally we modulate amplitude to turn off string 3?
        # But for TRAINING simplicty, let's just model 3 strings everywhere 
        # effectively or careful mask.
        # Actually synth.py returns "S" strings.
        # We need constant tensor size for batching -> [Batch, N, 3].
        
    # Bass (<=28)
    # 0,0,0
    
    # Random detuning (Training: Sample random noise? Or learn noise mean? Noise is noise.)
    # If we optimize stochastic params, we use Reparam Trick or just noise injection.
    # Noise injection allows learning variance if loss sees variance.
    # For now, let's inject noise so optimizer learns to minimize it (or set to correct level).
    # [Batch, 3]
    if rnd_std.ndim == 0:
        rnd_std = rnd_std.unsqueeze(0).unsqueeze(1) # [1, 1] to broadcast
    elif rnd_std.ndim == 1:
        rnd_std = rnd_std.unsqueeze(1) # [Batch, 1]
    
    rand_c = torch.randn(B_size, 3, device=device) * rnd_std
    
    c_total = c_offsets + rand_c
    
    # Apply to F0: [Batch, 1, 3]
    # f0 is [Batch], so unsqueeze(1).unsqueeze(2) to get [Batch, 1, 1]
    f0_s = f0.unsqueeze(1).unsqueeze(2) * torch.pow(2.0, c_total.unsqueeze(1) / 1200.0)
    
    # Inharmonicity Variation
    # [Batch, 3]
    # B_var_std is scalar or [1] from P()
    if B_var_std.ndim == 0:
        B_var_std = B_var_std.unsqueeze(0)
    
    B_rand = torch.randn(B_size, 3, device=device) * B_var_std
    B_j = 1.0 + B_rand
    
    B_s = (B_eff.unsqueeze(1) * B_j).clamp(min=1e-7) # [Batch, 3]
    
    # Final Freqs [Batch, N, 3]
    # f0_s: [B, 1, 3]
    # n: [1, N, 1]
    # B_s: [B, 1, 3]
    
    n_exp = n.unsqueeze(2) # [1, N, 1]
    n2 = n_exp.pow(2)
    
    fn = n_exp * f0_s * torch.sqrt(1.0 + B_s.unsqueeze(1) * n2)
    
    # We must handle 1/2 string masking in Amps?
    # Or just let them play (physics approx).
    # synth.py logic: "S" depends on midi.
    # If we output 3 always, bass will phase cancel or double volume?
    # Bass: 0, 0, 0 detune. 3 identical strings = +9.5dB. 
    # Real bass is 1 string.
    # We MUST mask amplitudes.
    
    string_mask = torch.ones(B_size, 3, device=device)
    if mask_2.any():
        string_mask[mask_2, 2] = 0.0 # Only 2 strings
    mask_1 = (midi.view(-1) <= 28)
    if mask_1.any():
        string_mask[mask_1, 1] = 0.0
        string_mask[mask_1, 2] = 0.0
        
    # --- Decays ---
    # tau_s0 = A * (55/f0)^p
    dA = P("A")
    dp = P("p")
    # Base f0 is fine for decay calc? Or per string?
    # synth.py uses f0 (center) for tau calculation mostly?
    # synth.py: tau_s0 = decay_A * (55/f0)**p. f0 is center.
    tau_s0 = dA * torch.pow(55.0 / f0.unsqueeze(1), dp) # [Batch, 1]
    
    # k = k0 + k1 * m_norm
    dk0 = P("k0")
    dk1 = P("k1")
    m_norm = (midi - 21.0) / 87.0
    k = dk0 + dk1 * m_norm # [Batch]
    
    tau_s = tau_s0 / (1.0 + k.unsqueeze(1) * (n**2))
    
    # tau_f ...
    div = P("tau_fast_divisor")
    tau_f = (tau_s0 / div) / (1.0 + 4.0 * k.unsqueeze(1) * (n**2))
    
    # --- Amplitudes ---
    # ... (existing calculation for A0 center) ...
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
    
    tilt = torch.pow(n, -p_tilt.unsqueeze(1))
    
    # Hammer Lowpass
    fc_min = P("fc_min")
    fc_max = P("fc_max")
    fc_v = P("fc_v_power")
    fc_f = P("fc_f_power")
    
    fc = fc_min + (fc_max - fc_min) * torch.pow(velocity.unsqueeze(1), fc_v) * torch.pow(f0.unsqueeze(1) / 261.63, fc_f)
    fc = fc.clamp(700.0, 14000.0)
    
    # Use freq center for partial
    freq_center = n * f0.unsqueeze(1) * torch.sqrt(1.0 + B_eff.unsqueeze(1) * n.pow(2))
    
    H_hammer = torch.rsqrt(1.0 + torch.pow(freq_center / fc, 4.0))
    
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
    
    H_body = torch.pow(freq_center / hp_f, hp_p) * torch.rsqrt(1.0 + torch.pow(freq_center / lp_f, lp_p))
    
    # Final A0 (Center)
    A0 = torch.pow(velocity.unsqueeze(1), 1.7) * comb * tilt * H_hammer * W * H_body
    
    # Split A0 into 3 strings
    # We normalized A0 in synth.py? No it sums directly.
    # But if we have 3 strings, we have 3x energy IF coherent.
    # In synth.py, it simply does `y = sum(strings)`.
    # So for correct loss, we must output 3 strings masked.
    
    # Amps: [Batch, N, 3]
    amps_3 = A0.unsqueeze(2) * string_mask.unsqueeze(1)
    # We should divide single strings by S_count? 
    # No, physically a 3-string piano key hits 3 strings, so it IS louder.
    # The A0 calculation has implicit gain. 
    # If we optimize A0 params (Decay A, etc) against real audio, it will learn total loudness.
    # So we just output 3 strings.
    
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
        "amps": amps_3,
        "w_curve": w_curve
    }
