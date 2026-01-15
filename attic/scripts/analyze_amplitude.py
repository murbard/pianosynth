import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pianosynth.optimization_batch import PianoParamPerKey

CHECKPOINT = Path("src/pianosynth/checkpoints/params_categorical.pt")

def main():
    device = "cpu"
    model = PianoParamPerKey(device=device)
    if not CHECKPOINT.exists():
        print("No checkpoint found.")
        return
        
    print(f"Loading {CHECKPOINT}...")
    cpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(cpt["model_state"])
    
    # Extract Amplitude
    midis = torch.arange(21, 109, device=device).float()
    
    # Assume representative velocities (used in init)
    vels = [0.25, 0.50, 0.75] # pp, mf, ff
    dyns = [0, 1, 2]
    
    amp_vals = []
    with torch.no_grad():
        for d in dyns:
            dyn_t = torch.full_like(midis, d).long()
            ov = model(midis, dyn_t)
            amp_vals.append(ov["amplitude"].numpy())
            
    amp_vals = np.stack(amp_vals) # [3, 88]
    x = midis.numpy()
    
    # --- Analysis 1: Raw Curves ---
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(x, amp_vals[0], 'b-', label='pp (v=0.25)')
    plt.plot(x, amp_vals[1], 'g-', label='mf (v=0.50)')
    plt.plot(x, amp_vals[2], 'r-', label='ff (v=0.75)')
    plt.xlabel('MIDI')
    plt.ylabel('Amplitude')
    plt.title('Learned Amplitude vs Pitch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # --- Analysis 2: Velocity Scaling ---
    # Model: Amp = Gain(m) * Velocity ^ Exponent(m)
    # log(Amp) = log(Gain) + Exponent * log(Vel)
    
    log_v = np.log(np.array(vels))
    log_a = np.log(amp_vals + 1e-9) # [3, 88]
    
    # For each key, fit line to 3 points
    # y = mx + c  => log_a = Exp * log_v + log_Gain
    
    exponents = []
    gains = []
    r2_scores = []
    
    for k in range(88):
        y = log_a[:, k]
        p = np.polyfit(log_v, y, 1) # [slope, intercept]
        
        exponents.append(p[0])
        gains.append(np.exp(p[1]))
        
        # R2
        y_pred = p[0] * log_v + p[1]
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2_scores.append(1 - (ss_res / (ss_tot + 1e-9)))
        
    exponents = np.array(exponents)
    gains = np.array(gains)
    r2_scores = np.array(r2_scores)
    
    print(f"Mean R^2 for Power Law fit: {np.mean(r2_scores):.4f}")
    print(f"Mean Exponent: {np.mean(exponents):.4f}")

    # --- Analysis 3: Spline Modeling of Gain and Exponent ---
    from scipy.interpolate import BSpline

    # Knots setup (Same as B_val for consistency)
    k = 3
    internal_knots = [28, 40, 70]
    t = np.concatenate(([21]*(k+1), internal_knots, [108]*(k+1)))
    n_c = len(t) - (k + 1)
    
    # Basis Matrix
    x_clamped = np.clip(x, 21, 108)
    B_mat = np.zeros((len(x), n_c))
    for i in range(n_c):
        c_dummy = np.zeros(n_c)
        c_dummy[i] = 1.0
        spl = BSpline(t, c_dummy, k)
        B_mat[:, i] = spl(x_clamped)
        
    # Fit Exponent Curve
    fs_exp = np.linalg.lstsq(B_mat, exponents, rcond=None)
    c_exp = fs_exp[0]
    spl_exp = BSpline(t, c_exp, k)
    exponent_fit = spl_exp(x_clamped)
    
    # Fit Gain Curve (Log domain for safety?)
    # Let's fit gain directly or log gain. Gain is linear scale.
    # Log gain is safer for positivity.
    log_gains = np.log(gains + 1e-9)
    fs_gain = np.linalg.lstsq(B_mat, log_gains, rcond=None)
    c_gain = fs_gain[0]
    spl_gain = BSpline(t, c_gain, k)
    log_gain_fit = spl_gain(x_clamped)
    gain_fit = np.exp(log_gain_fit) # * Bias correction?
    
    # Eval Total Model R2
    # Pred = G_fit * v^P_fit
    # We evaluate on the flattened dataset of all (3 * 88) points
    # Raw: amp_vals [3, 88].
    # Pred: [3, 88].
    # vels: [0.25, 0.5, 0.75]
    
    amp_flat = amp_vals.flatten()
    pred_flat = []
    
    for i, v in enumerate(vels): # 0, 1, 2
        # P_fit, G_fit are size 88
        row_pred = gain_fit * (v ** exponent_fit)
        pred_flat.append(row_pred)
        
    pred_flat = np.concatenate(pred_flat)
    
    res_total = amp_flat - pred_flat
    ss_res_tot = np.sum(res_total**2)
    ss_tot_val = np.sum((amp_flat - np.mean(amp_flat))**2)
    r2_global = 1 - (ss_res_tot / ss_tot_val)
    
    print(f"Global Model R^2 (Spline Gain + Spline Exp): {r2_global:.4f}")

    
    # Plot Learned Exponents
    plt.subplot(2, 2, 2)
    plt.plot(x, exponents, 'k.', alpha=0.3, label='Raw p')
    plt.plot(x, exponent_fit, 'c-', linewidth=2.5, label='Spline p')
    plt.axhline(1.7, color='r', linestyle='--', label='Legacy (1.7)')
    plt.xlabel('MIDI')
    plt.ylabel('Velocity Exponent')
    plt.title('Velocity Exponent (p)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot Learned Gain
    plt.subplot(2, 2, 3)
    plt.plot(x, gains, 'm.', alpha=0.3, label='Raw G')
    plt.plot(x, gain_fit, 'b-', linewidth=2.5, label='Spline G')
    plt.xlabel('MIDI')
    plt.ylabel('Base Gain')
    plt.title('Gain Curve (G)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Reconstruct and compare
    # Pred = G * v^p
    # Let's plot reconstruction for mf
    pred_mf_fit = gain_fit * (0.5 ** exponent_fit)
    
    plt.subplot(2, 2, 4)
    plt.plot(x, amp_vals[1], 'g-', alpha=0.5, label='True mf')
    plt.plot(x, pred_mf_fit, 'k--', linewidth=2.0, label='Spline Model mf')
    plt.xlabel('MIDI')
    plt.title(f'Reconstruction (mf) R2={r2_global:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results_categorical/amplitude_model_fit.png")
    print("Saved plot to results_categorical/amplitude_model_fit.png")

if __name__ == "__main__":
    main()
