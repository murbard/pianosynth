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

def calculate_partials(midi, overrides, n_partials=64, device="cpu"):
    """
    Vectorized physics calculator.
    midi: [Batch] or scalar
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
    
    B_size = midi.shape[0]
    
    # Helpers
    two = torch.tensor(2.0, device=device)
    n = torch.arange(1, n_partials + 1, device=device).float() # [N]
    
    # Resize n for broadcasting: [1, N]
    n = n.unsqueeze(0)
    
    # Unwrap Overrides
    def P(k): return overrides[k]
    
    # --- Tuning ---
    f_et = 440.0 * torch.pow(two, (midi - 69.0) / 12.0)
    f0 = f_et # [Batch]
    
    # --- Inharmonicity ---
    if "B_val" in overrides:
        B_eff = P("B_val")
    else:
        # Legacy fallback if overrides incomplete (should not happen in new training)
        B_vals = get_b_vectorized(midi, torch.tensor([5e-4]*5, device=device), device)
        B_scale = 1.0
        B_eff = B_vals * B_scale 
    
    # Detuning
    if "unison_detune_cents" in overrides:
        dc = P("unison_detune_cents")
        rnd_std = P("unison_random_cents")
    else:
        dc = torch.zeros_like(midi)
        rnd_std = torch.zeros_like(midi)
    
    # String Variation Params
    if "string_variation_std" in overrides:
        B_var_std = P("string_variation_std")
    else:
        B_var_std = torch.zeros_like(midi)

    # Expand to 3 strings: [Batch, 1, 3]
    c_list = []
    # Fixed detuning offsets [Batch, 3]
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
        c_offsets[mask_2, 2] = 0.0
        
    # Random detuning
    if rnd_std.ndim == 0:
        rnd_std = rnd_std.unsqueeze(0).unsqueeze(1)
    elif rnd_std.ndim == 1:
        rnd_std = rnd_std.unsqueeze(1)
    
    rand_c = torch.randn(B_size, 3, device=device) * rnd_std
    
    c_total = c_offsets + rand_c
    
    # Apply to F0
    f0_s = f0.unsqueeze(1).unsqueeze(2) * torch.pow(2.0, c_total.unsqueeze(1) / 1200.0)
    
    # Inharmonicity Variation
    if B_var_std.ndim == 0:
        B_var_std = B_var_std.unsqueeze(0).unsqueeze(1)
    elif B_var_std.ndim == 1:
        B_var_std = B_var_std.unsqueeze(1)
    
    B_rand = torch.randn(B_size, 3, device=device) * B_var_std
    B_j = 1.0 + B_rand
    
    B_s = (B_eff.unsqueeze(1) * B_j).clamp(min=1e-7) # [Batch, 3]
    
    # Final Freqs [Batch, N, 3]
    n_exp = n.unsqueeze(2) # [1, N, 1]
    n2 = n_exp.pow(2)
    
    fn = n_exp * f0_s * torch.sqrt(1.0 + B_s.unsqueeze(1) * n2)
    
    # Masking
    string_mask = torch.ones(B_size, 3, device=device)
    if mask_2.any():
        string_mask[mask_2, 2] = 0.0 # Only 2 strings
    mask_1 = (midi.view(-1) <= 28)
    if mask_1.any():
        string_mask[mask_1, 1] = 0.0
        string_mask[mask_1, 2] = 0.0
        
    # --- Decays ---
    if "decay_tau" in overrides:
        tau_s0 = P("decay_tau").unsqueeze(1) # [Batch, 1]
    else:
        tau_s0 = torch.ones_like(f0).unsqueeze(1)
        
    if "decay_k" in overrides:
        k = P("decay_k")
    else:
        k = torch.zeros_like(f0)
    
    tau_s = tau_s0 / (1.0 + k.unsqueeze(1) * (n**2))
    
    # tau_f
    if "tau_fast_divisor" in overrides:
        div = P("tau_fast_divisor")
    else:
        div = torch.ones_like(f0) * 8.0
        
    tau_f = (tau_s0 / div.unsqueeze(1)) / (1.0 + 4.0 * k.unsqueeze(1) * (n**2))
    
    # --- Amplitudes ---
    # Strike Position xh
    if "hammer_xh" in overrides:
        xh = P("hammer_xh")
    else:
        xh = torch.ones_like(f0) * 0.12
        
    if "comb_mix" in overrides:
        c_mix = P("comb_mix")
        c_base = P("comb_base")
    else:
        c_mix = 0.5; c_base = 0.5
        
    comb = c_base.unsqueeze(1) + c_mix.unsqueeze(1) * torch.sin(torch.pi * n * xh.unsqueeze(1)).pow(2)
    
    # Tilt (Categorical)
    if "hammer_p_tilt" in overrides:
        p_tilt = P("hammer_p_tilt")
    else:
        p_tilt = torch.ones_like(f0) * 2.0
        
    tilt = torch.pow(n, -p_tilt.unsqueeze(1))
    
    # Hammer Lowpass fc (Categorical)
    if "hammer_fc" in overrides:
         fc = P("hammer_fc").unsqueeze(1)
    else:
         fc = torch.ones_like(f0).unsqueeze(1) * 2000.0
    
    fc = fc.clamp(20.0, 20000.0)
    
    # Use freq center for partial
    freq_center = n * f0.unsqueeze(1) * torch.sqrt(1.0 + B_eff.unsqueeze(1) * n.pow(2))
    
    H_hammer = torch.rsqrt(1.0 + torch.pow(freq_center / fc, 4.0))
    
    # N_w (Categorical)
    if "hammer_nw" in overrides:
        n_w = P("hammer_nw").unsqueeze(1)
    else:
        n_w = torch.ones_like(f0).unsqueeze(1) * 30.0
        
    W = torch.exp(-torch.pow(n / n_w, 2.0))
    
    # Body
    if "highpass_freq" in overrides:
        hp_f = P("highpass_freq")
        hp_p = P("highpass_power")
        lp_f = P("lowpass_freq")
        lp_p = P("lowpass_power")
        
        H_body = torch.pow(freq_center / hp_f.unsqueeze(1), hp_p.unsqueeze(1)) * torch.rsqrt(1.0 + torch.pow(freq_center / lp_f.unsqueeze(1), lp_p.unsqueeze(1)))
    else:
        H_body = 1.0
    
    # Final A0 (Center)
    # Amplitude (Categorical)
    if "amplitude" in overrides:
        base_amp = P("amplitude").unsqueeze(1)
    else:
        base_amp = 0.5
        
    A0 = base_amp * comb * tilt * H_hammer * W * H_body
    
    # Amps: [Batch, N, 3]
    amps_3 = A0.unsqueeze(2) * string_mask.unsqueeze(1)
    amps_3 = amps_3.clamp(max=1e5)
    
    # Prompt W mix
    if "prompt_n_mix" in overrides:
        n_wmix = P("prompt_n_mix").unsqueeze(1)
    else:
        n_wmix = torch.ones_like(f0).unsqueeze(1) * 20.0
    
    if "w_min" in overrides:
        w_min = P("w_min")
        w_max = P("w_max")
    else:
        w_min = 0.0; w_max = 1.0
        
    w_curve = w_min.unsqueeze(1) + (w_max.unsqueeze(1) - w_min.unsqueeze(1)) * (1.0 - torch.exp(-torch.pow(n / n_wmix, 2.0)))
    
    return {
        "freqs": fn,
        "tau_s": tau_s,
        "tau_f": tau_f,
        "amps": amps_3,
        "w_curve": w_curve,
        "reverb_wet": overrides.get("reverb_wet", None),
        "reverb_decay": overrides.get("reverb_decay", None)
    }
