from __future__ import annotations
import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from dgp import DGP
from models import AvarModel, REGISTRY


class CalibratedDGP(DGP):
    """Shifts any DGP mean so unconditional SR == target_sr."""

    def __init__(self, base, target_sr, calibration_n=100_000, calib_seed=0):
        self.base      = base
        self.target_sr = target_sr
        rng        = np.random.default_rng(calib_seed)
        large      = base.simulate(calibration_n, rng)
        self._mu    = float(large.mean())
        self._sigma = float(large.std())
        self._drift = target_sr * self._sigma - self._mu

    @property
    def sigma(self): return self._sigma

    def simulate(self, n, rng):
        return self.base.simulate(n, rng) + self._drift

    def _repr_params(self):
        return f"{self.base!r}, target_sr={self.target_sr}"


@dataclass
class DGPSpec:
    dgp:            DGP
    name:           str
    pre_calibrated: bool = False


def _est_skew(x):     return float(stats.skew(x))
def _est_exc_kurt(x): return float(stats.kurtosis(x, fisher=True))

def _est_rho(x):
    lag = x[:-1]; y = x[1:]
    dm  = lag - lag.mean()
    den = float(np.dot(dm, dm))
    return 0.0 if den < 1e-12 else float(np.clip(np.dot(dm, y - y.mean()) / den, -0.999, 0.999))

def _est_nu(x, lo=4.05, hi=100.0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            df, _, _ = stats.t.fit(x, floc=float(x.mean()))
            return float(np.clip(df, lo, hi))
        except Exception:
            return 10.0

_ESTIMATORS = {"skew": _est_skew, "exc_kurt": _est_exc_kurt, "rho": _est_rho, "nu": _est_nu}

def estimate_nuisance(x, model):
    return {k: _ESTIMATORS[k](x) for k in model.parameters if k in _ESTIMATORS}


def _sr_hat(x):
    s = float(x.std(ddof=1))
    return float(x.mean() / s) if s > 1e-12 else 0.0


def run_pair(dgp, model, true_sr, T, n_sim, alpha, rng):
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    sr_hats = np.empty(n_sim); ci_widths = np.empty(n_sim)
    V_hats  = np.empty(n_sim); covered   = np.zeros(n_sim, dtype=bool)
    for i in range(n_sim):
        x    = dgp.simulate(T, rng)
        sr_h = _sr_hat(x)
        nuis = estimate_nuisance(x, model) # TODO each model should have its own estimate params method
        V    = float(model(sr_h, **nuis))
        if not (np.isfinite(V) and V > 0):
            V = float(model(sr_h))
        hw = z * np.sqrt(V / T)
        sr_hats[i] = sr_h; ci_widths[i] = 2*hw; V_hats[i] = V
        covered[i] = bool(sr_h - hw <= true_sr <= sr_h + hw)
    return {
        "coverage":      float(covered.mean()),
        "mean_sr_hat":   float(sr_hats.mean()),
        "bias":          float(sr_hats.mean() - true_sr),
        "rmse":          float(np.sqrt(((sr_hats - true_sr)**2).mean())),
        "mean_ci_width": float(ci_widths.mean()),
        "mean_V_hat":    float(V_hats.mean()),
    }


def run_coverage_study(
    dgp_specs, avar_models,
    target_sr=0.5, T=500, n_sim=2000, alpha=0.05,
    seed=42, verbose=True, calibration_n=100_000,
):
    master_rng = np.random.default_rng(seed)
    nominal    = 1.0 - alpha
    rows       = []

    calibrated = []
    for spec in dgp_specs:
        if spec.pre_calibrated:
            calibrated.append((spec.name, spec.dgp, target_sr))
            if verbose: print(f"  [pre-calib]   {spec.name}")
        else:
            if verbose: print(f"  Calibrating   {spec.name} ...", end=" ", flush=True)
            cdgp = CalibratedDGP(spec.dgp, target_sr, calibration_n)
            calibrated.append((spec.name, cdgp, target_sr))
            if verbose: print(f"sigma={cdgp.sigma:.4f}  drift={cdgp._drift:+.6f}")

    total = len(calibrated) * len(avar_models)
    done  = 0
    w     = len(str(total))

    for dgp_name, cdgp, true_sr in calibrated:
        dgp_rng = np.random.default_rng(master_rng.integers(0, 2**31))
        for model in avar_models:
            pair_rng = np.random.default_rng(dgp_rng.integers(0, 2**31))
            done += 1
            if verbose:
                print(f"  [{done:{w}}/{total}]  DGP={dgp_name:<28}  Model={model.name:<22} ...", end=" ", flush=True)
            res = run_pair(cdgp, model, true_sr, T, n_sim, alpha, pair_rng)
            if verbose:
                flag = "OK" if abs(res["coverage"] - nominal) < 0.03 else "!!"
                print(f"cov={res['coverage']:.3f} [{flag}]")
            rows.append({"dgp_name": dgp_name, "avar_model": model.name, "nominal_coverage": nominal, **res})

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