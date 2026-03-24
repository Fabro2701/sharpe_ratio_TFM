from __future__ import annotations
import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from core.dgp import DGP
from core.models import AvarModel, REGISTRY
from utils.calibration_sr import calibrate_dgp



@dataclass
class DGPSpec:
    dgp:            DGP
    name:           str


def _sr_hat(x):
    s = float(x.std(ddof=1))
    return float(x.mean() / s) if s > 1e-12 else 0.0

def run_dgp_models(dgp, avar_models, true_sr, T, n_sim, alpha, th_moments, rng):
    """
    Simulates data for a single DGP and evaluates multiple models on the exact same paths.
    """
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    n_models = len(avar_models)
    
    # Pre-allocate arrays
    sr_hats   = np.empty(n_sim)
    ci_widths = np.empty((n_sim, n_models))
    V_hats    = np.empty((n_sim, n_models))
    covered   = np.zeros((n_sim, n_models), dtype=bool)

    # Optimization: Theoretical moments don't depend on the simulated path, 
    # so we fetch them once before the loop.
    th_moms = dgp.get_theo_moments() if th_moments else None

    for i in range(n_sim):
        # 1. Simulate data and calculate the Sharpe Ratio estimate ONCE
        x = dgp.simulate(T, rng)
        sr_h = _sr_hat(x)
        sr_hats[i] = sr_h
        
        # 2. Evaluate every model on this same dataset
        for j, model in enumerate(avar_models):
            if th_moments:
                V = float(model(sr_h, **th_moms))
            else:
                params_h = model.fit(x)
                V = float(model(sr_h, **params_h))
                
            if not (np.isfinite(V) and V > 0):
                print(f"Warning: Non-finite/positive variance in {model.short_name}. Using fallback.")
                V = float(model(sr_h))
                
            hw = z * np.sqrt(V / T)
            ci_widths[i, j] = 2 * hw
            V_hats[i, j]    = V
            covered[i, j]   = bool(sr_h - hw <= true_sr <= sr_h + hw)

    # 3. Aggregate metrics
    mean_sr_hat = float(sr_hats.mean())
    bias        = float(mean_sr_hat - true_sr)
    rmse        = float(np.sqrt(((sr_hats - true_sr)**2).mean()))

    # Build the results dictionary mapped by model short_name
    results = {}
    for j, model in enumerate(avar_models):
        results[model.short_name] = {
            "coverage":      float(covered[:, j].mean()),
            "mean_sr_hat":   mean_sr_hat, # Shared across models
            "bias":          bias,        # Shared across models
            "rmse":          rmse,        # Shared across models
            "mean_ci_width": float(ci_widths[:, j].mean()),
            "mean_V_hat":    float(V_hats[:, j].mean()),
        }
        
    return results

def run_coverage_study(
    dgp_specs, avar_models,
    target_sr=0.5, T=500, n_sim=2000, alpha=0.05,
    th_moments=False,
    seed=42, verbose=True
):
    master_rng = np.random.default_rng(seed)
    nominal    = 1.0 - alpha
    rows       = []

    calibrated = []
    for spec in dgp_specs:
        calibrate_dgp(spec.dgp, target_sr, mu=0.15)
        calibrated.append((spec.name, spec.dgp, target_sr))

    total_dgps = len(calibrated)
    done       = 0

    if th_moments:
        print("Using theoretical moments")

    for dgp_name, cdgp, true_sr in calibrated:
        dgp_rng = np.random.default_rng(master_rng.integers(0, 2**31))
        done += 1
        
        if verbose:
            print(f"[{done}/{total_dgps}] Simulating DGP={dgp_name:<28} ...")
            
        # Run all simulations for this DGP
        res_dict = run_dgp_models(cdgp, avar_models, true_sr, T, n_sim, alpha, th_moments, dgp_rng)
        
        # Unpack results and print formatted output for each model
        for model in avar_models:
            m_name = model.short_name
            res = res_dict[m_name]
            
            if verbose:
                flag = "OK" if abs(res["coverage"] - nominal) < 0.01 else "!!"
                print(f"  -> Model={m_name:<22} cov={res['coverage']:.3f} [{flag}]")
                
            rows.append({
                "dgp_name": dgp_name, 
                "avar_model": m_name, 
                "nominal_coverage": nominal, 
                **res
            })

    return pd.DataFrame(rows)


def pivot_table(results, metric="coverage"):
    wide = results.pivot(index="dgp_name", columns="avar_model", values=metric)
    wide.index.name = "DGP"; wide.columns.name = metric
    return wide


def coverage_report(results, alpha=0.05, tol=0.03):
    nominal = 1.0 - alpha
    cw = 20
    sep = "=" * 82

    def hdr(models): return "  {:<28}".format("DGP") + "".join(f"{m:>{cw}}" for m in models)
    def div(models): return "  " + "-" * (28 + cw * len(models))

    lines = [sep, f"  Coverage Study  |  nominal={nominal:.2f}  tol=+/-{tol:.2f}", sep]

    metrics = [
        ("coverage",      "EMPIRICAL COVERAGE",
         lambda v: f"{v:.3f} {'OK' if abs(v-nominal)<=tol else '!!':>2}".rjust(cw)),
        ("bias",          "BIAS of SR_hat",
         lambda v: f"{v:+.4f}".rjust(cw)),
        ("rmse",          "RMSE of SR_hat",
         lambda v: f"{v:.4f}".rjust(cw)),
        ("mean_ci_width", "MEAN CI WIDTH",
         lambda v: f"{v:.4f}".rjust(cw)),
    ]

    for col, title, fmt_fn in metrics:
        tbl    = pivot_table(results, col)
        models = tbl.columns.tolist()
        lines += [f"\n  {title}", hdr(models), div(models)]
        for dgp, row in tbl.iterrows():
            lines.append(f"  {dgp:<28}" + "".join(fmt_fn(v) for v in row))

    lines.append("\n" + sep)
    return "\n".join(lines)