import torch
from pianosynth.optimization_batch import PianoParamPerKey
from pianosynth.physics import calculate_partials

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    INPUT = "src/pianosynth/checkpoints/params_best_ever.pt"
    
    # Load
    model = PianoParamPerKey(device=device)
    cpt = torch.load(INPUT, map_location=device)
    model.load_state_dict(cpt["model_state"])
    
    # Query Note 21 MF (Lowest note)
    m = 21
    d = 1 # mf
    
    m_t = torch.tensor([m], device=device).float()
    d_t = torch.tensor([d], device=device).long()
    
    with torch.no_grad():
        overrides = model(m_t, d_t)
        phys = calculate_partials(m_t, overrides, device=device)
        
        print(f"--- MIDI {m} Dynamics {d} (mf) ---")
        print(f"Amplitudes Shape: {phys['amps'].shape}")
        print(f"Decay Shape: {phys['tau_s'].shape}")
        
        # Correct Keys
        print(f"Inharmonicity (B): {overrides['B_val'].item():.2e}")
        print(f"Hammer p (Tilt): {overrides['hammer_p_tilt'].item():.4f}")
        print(f"Hammer fc (Cutoff): {overrides['hammer_fc'].item():.4f}")
        print(f"Hammer nw: {overrides['hammer_nw'].item():.4f}")
        
        print(f"Strike Position (xh): {overrides['hammer_xh'].item():.4f}")
        
        print(f"Body LP Freq: {overrides['lowpass_freq'].item():.4f}")
        print(f"Body HP Freq: {overrides['highpass_freq'].item():.4f}")
        
        # Check Neighbors
        print(f"--- Neighbors ---")
        for n_m in [22, 23, 24]:
             m_n = torch.tensor([n_m], device=device).float()
             ov_n = model(m_n, d_t)
             print(f"MIDI {n_m} Amp: {ov_n['amplitude'].item():.4f}, p: {ov_n['hammer_p_tilt'].item():.4f}, HP: {ov_n['highpass_freq'].item():.1f}")

    

if __name__ == "__main__":
    main()
