import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pianosynth.optimization_batch import PianoParamPerKey
from scipy.interpolate import BSpline

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
    
    # Extract B_val
    midis = torch.arange(21, 109, device=device).float()
    
    # Get values for all dynamics
    # 0=pp, 1=mf, 2=ff
    dyns = [0, 1, 2]
    b_vals = []
    
    with torch.no_grad():
        for d in dyns:
            dyn_t = torch.full_like(midis, d).long()
            ov = model(midis, dyn_t)
            b_vals.append(ov["B_val"].numpy())
            
    b_vals = np.stack(b_vals) # [3, 88]
    
    # 1. Analyze Speed Dependency
    means = np.mean(b_vals, axis=0)
    stds = np.std(b_vals, axis=0)
    cv = stds / (means + 1e-9)
    
    avg_cv = np.mean(cv)
    print(f"Average Coeff of Variation across dynamics: {avg_cv:.4f}")
    
    if avg_cv < 0.05:
        print(">> Hypothesis: B_val is INDEPENDENT of Velocity (Speed).")
    else:
        print(">> Hypothesis: B_val DEPENDS on Velocity (Speed).")
        
    # 2. Fit Models
    log_b_mean = np.log10(means)
    x = midis.numpy()
    ss_tot = np.sum((log_b_mean - np.mean(log_b_mean))**2)

    # --- Linear Piecewise Fit ---
    knot_pos_lin = [21, 28, 40, 70, 108]
    n_knots_lin = len(knot_pos_lin)
    
    A = np.zeros((len(x), n_knots_lin))
    for i in range(n_knots_lin):
        ctr = knot_pos_lin[i]
        if i > 0:
            left = knot_pos_lin[i-1]
            mask = (x >= left) & (x <= ctr)
            A[mask, i] = (x[mask] - left) / (ctr - left)
        if i < n_knots_lin - 1:
            right = knot_pos_lin[i+1]
            mask = (x >= ctr) & (x <= right)
            A[mask, i] = (right - x[mask]) / (right - ctr)
            
    fs_lin = np.linalg.lstsq(A, log_b_mean, rcond=None)
    w_lin = fs_lin[0]
    b_fit = 10**(A @ w_lin) # Linear Prediction
    
    res_lin = log_b_mean - (A @ w_lin)
    r2_lin = 1 - (np.sum(res_lin**2) / ss_tot)
    
    print(f"Piecewise Linear Fit (Knots: {knot_pos_lin})")
    print(f"R^2: {r2_lin:.4f}")

    # --- Spline Fit (Cubic B-Spline) ---
    k = 3
    # Typical Spline Knots setup: Clamped
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
        
    fs_spline = np.linalg.lstsq(B_mat, log_b_mean, rcond=None)
    c_opt = fs_spline[0]
    
    spl_final = BSpline(t, c_opt, k)
    log_b_spline = spl_final(x_clamped)
    b_spline = 10**log_b_spline
    
    res_spl = log_b_mean - log_b_spline
    r2_spl = 1 - (np.sum(res_spl**2) / ss_tot)
    
    print(f"Cubic B-Spline Fit (Internal Knots: {internal_knots})")
    print(f"R^2: {r2_spl:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    
    # Raw Data scatter
    plt.scatter(x, b_vals[0], c='b', alpha=0.15, s=10, label='pp raw')
    plt.scatter(x, b_vals[1], c='g', alpha=0.15, s=10, label='mf raw')
    plt.scatter(x, b_vals[2], c='r', alpha=0.15, s=10, label='ff raw')
    
    # Mean
    plt.plot(x, means, 'k--', alpha=0.4, label='Mean Data')
    
    # Fits
    plt.plot(x, b_fit, 'm-', linewidth=2.0, alpha=0.6, label=f'Piecewise Linear (R2={r2_lin:.3f})')
    plt.plot(x, b_spline, 'c-', linewidth=2.5, label=f'Cubic Spline (R2={r2_spl:.3f})')
    
    plt.yscale('log')
    plt.xlabel('MIDI Note')
    plt.ylabel('Inharmonicity Coefficient (B)')
    plt.title(f'Comparison of B_val Physical Models')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    # Plot knots (Linear)
    plt.plot(knot_pos_lin, 10**w_lin, 'mo', markersize=6)
    
    plt.savefig("results_categorical/B_val_model_fit.png")
    print("Saved plot to results_categorical/B_val_model_fit.png")

if __name__ == "__main__":
    main()
