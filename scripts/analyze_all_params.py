import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pianosynth.optimization_batch import PianoParamPerKey
from scipy.interpolate import BSpline
import pandas as pd

CHECKPOINT = Path("src/pianosynth/checkpoints/params_categorical.pt")
OUT_DIR = Path("results_categorical/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def fit_spline(x, y, knots=[28, 40, 70], k=3):
    # x: [88] 21..108. y: [88] values
    # Standard clamped knots
    t = np.concatenate(([21]*(k+1), knots, [108]*(k+1)))
    n_c = len(t) - (k + 1)
    
    x_clamped = np.clip(x, 21, 108)
    B_mat = np.zeros((len(x), n_c))
    for i in range(n_c):
        c_dummy = np.zeros(n_c)
        c_dummy[i] = 1.0
        spl = BSpline(t, c_dummy, k)
        B_mat[:, i] = spl(x_clamped)
        
    coeffs = np.linalg.lstsq(B_mat, y, rcond=None)[0]
    spl = BSpline(t, coeffs, k)
    y_pred = spl(x_clamped)
    return y_pred, spl, coeffs

def main():
    device = "cpu"
    model = PianoParamPerKey(device=device)
    if not CHECKPOINT.exists():
        print("No checkpoint found.")
        return
        
    print(f"Loading {CHECKPOINT}...")
    cpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(cpt["model_state"])
    
    midis = torch.arange(21, 109, device=device).float()
    vels = np.array([0.25, 0.50, 0.75])
    dyns = [0, 1, 2]
    
    # Get all param names
    # Run once to get keys
    with torch.no_grad():
        dummy_ov = model(midis[:1], torch.tensor([0]))
    param_names = list(dummy_ov.keys())
    # Filter some meta params if needed? No, analyze all.
    
    # Collect Data
    data_store = {}
    with torch.no_grad():
        for d in dyns:
            dyn_t = torch.full_like(midis, d).long()
            ov = model(midis, dyn_t)
            for k, val in ov.items():
                if k not in data_store: data_store[k] = []
                # Ensure numpy
                v_np = val.numpy() if hasattr(val, 'numpy') else val
                if np.ndim(v_np) == 0: v_np = np.full(88, v_np)
                data_store[k].append(v_np)
                
    results = []
    
    x = midis.numpy()
    
    print(f"Analyzing {len(param_names)} parameters...")
    
    for name in param_names:
        vals = np.stack(data_store[name]) # [3, 88]
        
        # 1. Vel Dependency Check
        means = np.mean(vals, axis=0)
        stds = np.std(vals, axis=0)
        cv = stds / (np.abs(means) + 1e-9)
        avg_cv = np.mean(cv)
        
        is_vel_dep = avg_cv > 0.05 # 5% threshold
        
        # 2. Fit Model
        r2 = 0.0
        msg = ""
        
        plt.figure(figsize=(10, 6))
        
        # Plot Raw
        plt.plot(x, vals[0], 'b.', alpha=0.3, label='pp')
        plt.plot(x, vals[1], 'g.', alpha=0.3, label='mf')
        plt.plot(x, vals[2], 'r.', alpha=0.3, label='ff')
        
        if not is_vel_dep:
            # Fit Spline to Mean
            # Use Log domain if positive?
            # Check positivity
            is_pos = np.all(means > 0)
            target = np.log(means + 1e-9) if is_pos else means
            
            fit_target, _, _ = fit_spline(x, target)
            pred = np.exp(fit_target) if is_pos else fit_target
            
            # Global R2 (against all 3 dyns raw data)
            flat_raw = vals.flatten()
            flat_pred = np.tile(pred, 3)
            ss_res = np.sum((flat_raw - flat_pred)**2)
            ss_tot = np.sum((flat_raw - np.mean(flat_raw))**2)
            r2 = 1 - (ss_res / ss_tot)
            
            plt.plot(x, pred, 'k-', linewidth=3, label=f'Model (Indep) R2={r2:.2f}')
            msg = "Velocity Independent"
            
        else:
            # Fit Power Law + Splines
            # A = G * v^p
            log_v = np.log(vels)
            log_vals = np.log(np.abs(vals) + 1e-9)
            
            exponents = []
            gains = []
            
            for k in range(88):
                y = log_vals[:, k]
                p = np.polyfit(log_v, y, 1) # slope, intercept
                exponents.append(p[0])
                gains.append(np.exp(p[1]))
                
            exponents = np.array(exponents)
            gains = np.array(gains)
            
            # Smooth P and G
            p_smooth, _, _ = fit_spline(x, exponents)
            
            # Smooth G (log domain)
            g_smooth_log, _, _ = fit_spline(x, np.log(gains + 1e-9))
            g_smooth = np.exp(g_smooth_log)
            
            # Calculate Preds
            # For each dyn, pred = G_smooth * v^P_smooth
            flat_pred = []
            for v in vels:
                row = g_smooth * (v ** p_smooth)
                plt.plot(x, row, linestyle='--', linewidth=2, alpha=0.8, color='k')
                flat_pred.append(row)
            flat_pred = np.concatenate(flat_pred) # ordering matches flat_raw (pp, mf, ff blocks)
            
            # Label just once
            plt.plot([], [], 'k--', label='Model (Dep)')
                
            flat_raw = vals.flatten()
            ss_res = np.sum((flat_raw - flat_pred)**2)
            ss_tot = np.sum((flat_raw - np.mean(flat_raw))**2)
            r2 = 1 - (ss_res / ss_tot) 
            
            msg = "Velocity Dependent (Power Law)"

        plt.title(f"{name}: {msg} (CV={avg_cv:.2f}, R2={r2:.3f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        safe_name = name.replace(".", "_")
        plt.savefig(OUT_DIR / f"{safe_name}_model.png")
        plt.close()
        
        results.append({
            "Parameter": name,
            "VelocityDep": is_vel_dep,
            "CV": avg_cv,
            "ModelR2": r2
        })
        
    # Save Summary
    df = pd.DataFrame(results)
    df = df.sort_values("ModelR2", ascending=False)
    print("\nAnalysis Summary:")
    print(df.to_string(index=False))
    df.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\nPlots saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
