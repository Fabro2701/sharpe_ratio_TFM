# sr_sim.py  —  Simulation engine for Sharpe ratio inference studies.
#
# Supports 6 study types via the StudyType enum:
#   Coverage (empirical size, null is true):
#     TWO_SIDED_COVERAGE   — two-sided CI coverage
#     ONE_SIDED_COVERAGE   — one-sided upper test size
#     TWO_SAMPLE_COVERAGE  — two-sample one-sided test size
#   Power (rejection rate, null is false):
#     TWO_SIDED_POWER      — two-sided CI power
#     ONE_SIDED_POWER      — one-sided upper test power
#     TWO_SAMPLE_POWER     — two-sample one-sided test power

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from scipy import stats

from core.dgp import DGP
from core.models import AvarModel, REGISTRY
from utils.calibration_sr import calibrate_dgp


# ─────────────────────────────────────────────────────────────────────────────
# Study type enum
# ─────────────────────────────────────────────────────────────────────────────

class StudyType(Enum):
    """
    Defines the statistical inference scenario being studied.

    Coverage variants hold the null true and measure empirical size:
        P(CI contains θ₀) or P(test does not reject | H₀ true).

    Power variants set the DGP under the alternative and measure the
    rejection rate:
        P(CI excludes θ₀) or P(test rejects | H₁ true).

    One-sided tests use the upper-tail convention:
        H₀: SR ≤ θ₀  vs  H₁: SR > θ₀

    Two-sample tests compare two independent Sharpe ratios:
        H₀: SR₁ − SR₂ ≤ Δ₀  vs  H₁: SR₁ − SR₂ > Δ₀  (default Δ₀ = 0)
    """
    # ── size / coverage ──────────────────────────────────────────────────────
    TWO_SIDED_COVERAGE  = auto()   # P(θ₀ ∈ CI | θ = θ₀)
    ONE_SIDED_COVERAGE  = auto()   # P(not reject | θ = θ₀)
    TWO_SAMPLE_COVERAGE = auto()   # P(not reject | θ₁ = θ₂)
    # ── power ────────────────────────────────────────────────────────────────
    TWO_SIDED_POWER     = auto()   # P(θ₀ ∉ CI | θ ≠ θ₀)
    ONE_SIDED_POWER     = auto()   # P(reject | θ > θ₀)
    TWO_SAMPLE_POWER    = auto()   # P(reject | θ₁ > θ₂)

    # ── convenience predicates ───────────────────────────────────────────────

    @property
    def is_power(self) -> bool:
        """True for power variants; False for coverage/size variants."""
        return self in {StudyType.TWO_SIDED_POWER,
                        StudyType.ONE_SIDED_POWER,
                        StudyType.TWO_SAMPLE_POWER}

    @property
    def is_two_sample(self) -> bool:
        return self in {StudyType.TWO_SAMPLE_COVERAGE,
                        StudyType.TWO_SAMPLE_POWER}

    @property
    def is_two_sided(self) -> bool:
        return self in {StudyType.TWO_SIDED_COVERAGE,
                        StudyType.TWO_SIDED_POWER}

    @property
    def is_one_sided(self) -> bool:
        return self in {StudyType.ONE_SIDED_COVERAGE,
                        StudyType.ONE_SIDED_POWER}

    @property
    def metric_name(self) -> str:
        """Column label for the primary outcome."""
        return "power" if self.is_power else "coverage"


# ─────────────────────────────────────────────────────────────────────────────
# DGP specification dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DGPSpec:
    dgp:  DGP
    name: str


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _sr_hat(x: np.ndarray) -> float:
    s = float(x.std(ddof=1))
    return float(x.mean() / s) if s > 1e-12 else 0.0


def _avar_estimate(model: AvarModel, sr_h: float, x: np.ndarray,
                   th_moments: bool, th_moms: dict | None) -> float:
    """Return the asymptotic variance estimate V̂ for a single model/path."""
    mom = None
    if th_moments and th_moms is not None:
        V = float(model(sr_h, **th_moms))
        mom = th_moms
    else:
        params_h = model.fit(x)
        V = float(model(sr_h, **params_h))
        mom = params_h

    if not (np.isfinite(V) and V > 0):
        print("no fitted nuisance params")
        V = float(model(sr_h))   # bare fallback — no fitted nuisance params
    return V, mom


# ─────────────────────────────────────────────────────────────────────────────
# Per-path simulation: one-sample (two-sided and one-sided)
# ─────────────────────────────────────────────────────────────────────────────

def _event_one_sample(sr_h: float, V: float, T: int, null_sr: float,
                      alpha: float, study_type: StudyType) -> tuple[bool, float]:
    """
    Returns (event, effective_half_width).

    For coverage types  event = "CI / LCB covers null"  (want True ≈ 1 - alpha).
    For power types     event = "CI / LCB excludes null" (want True → 1 as n grows).

    Two-sided:  CI = [sr_h ± z_{α/2} · √(V/T)]
    One-sided:  lower confidence bound (LCB) = sr_h − z_α · √(V/T);
                reject when LCB > null_sr.
    """
    if study_type.is_two_sided:
        z  = float(stats.norm.ppf(1.0 - alpha / 2.0))
        hw = z * np.sqrt(V / T)
        in_ci = (sr_h - hw) <= null_sr <= (sr_h + hw)
        return (not in_ci) if study_type.is_power else in_ci, 2.0 * hw

    # One-sided upper
    z        = float(stats.norm.ppf(1.0 - alpha))
    hw       = z * np.sqrt(V / T)
    rejected = (sr_h - hw) > null_sr          # LCB > θ₀  →  reject H₀
    return rejected if study_type.is_power else (not rejected), hw


def _run_one_sample_path(seed, dgp, avar_models, T, null_sr, alpha,
                         study_type, th_moments, th_moms, bias_adj):
    rng  = np.random.default_rng(seed)
    x    = dgp.simulate(T, rng)
    sr_h = _sr_hat(x)

    n  = len(avar_models)
    widths = np.empty(n)
    V_hats = np.empty(n)
    events = np.zeros(n, dtype=bool)

    for j, model in enumerate(avar_models):
        V, mom = _avar_estimate(model, sr_h, x, th_moments, th_moms)
        sr_h_adj = model.correct_bias(bias_adj, T, sr_h, **mom)
        event, width = _event_one_sample(sr_h_adj, V, T, null_sr, alpha, study_type)
        widths[j] = width
        V_hats[j] = V
        events[j] = event

    return sr_h, widths, V_hats, events


# ─────────────────────────────────────────────────────────────────────────────
# Per-path simulation: two-sample one-sided
# ─────────────────────────────────────────────────────────────────────────────

def _run_two_sample_path(seed, dgp1, dgp2, avar_models1, avar_models2,
                         T, null_diff, alpha, study_type, th_moments,
                         th_moms1, th_moms2, bias_adj):
    """
    H₀: SR₁ − SR₂ ≤ Δ₀  (null_diff = Δ₀, usually 0).

    Test statistic:  Z = (sr̂₁ − sr̂₂ − Δ₀) / √((V̂₁ + V̂₂) / T)
    Reject when Z > z_α.

    Models are matched pairwise (avar_models1[j] for DGP1,
    avar_models2[j] for DGP2).  Pair counts must match.
    """
    rng   = np.random.default_rng(seed)
    x1    = dgp1.simulate(T, rng)
    x2    = dgp2.simulate(T, rng)
    sr_h1 = _sr_hat(x1)
    sr_h2 = _sr_hat(x2)

    z  = float(stats.norm.ppf(1.0 - alpha))
    n  = len(avar_models1)
    V_hats = np.empty(n)
    events = np.zeros(n, dtype=bool)

    for j, (m1, m2) in enumerate(zip(avar_models1, avar_models2)):
        V1, _  = _avar_estimate(m1, sr_h1, x1, th_moments, th_moms1)
        V2, _ = _avar_estimate(m2, sr_h2, x2, th_moments, th_moms2)
        V_combined = V1 + V2 #check rho missing
        V_hats[j]  = V_combined

        test_stat  = (sr_h1 - sr_h2 - null_diff) / np.sqrt(V_combined / T)
        rejected   = test_stat > z
        events[j]  = rejected if study_type.is_power else (not rejected)

    sr_diff = sr_h1 - sr_h2
    # ci_widths are not well-defined for a one-sided two-sample test; use NaN
    return sr_diff, np.full(n, np.nan), V_hats, events


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator: parallel dispatch over n_sim paths
# ─────────────────────────────────────────────────────────────────────────────

def _assemble_results(results_list, n_sim, n_models):
    """Unpack list-of-tuples into pre-allocated arrays."""
    sr_hats   = np.empty(n_sim)
    widths    = np.empty((n_sim, n_models))
    V_hats    = np.empty((n_sim, n_models))
    events    = np.zeros((n_sim, n_models), dtype=bool)

    for i, (sr_h, cw, vh, ev) in enumerate(results_list):
        sr_hats[i]    = sr_h
        widths[i, :]  = cw
        V_hats[i, :]  = vh
        events[i, :]  = ev

    return sr_hats, widths, V_hats, events


def run_dgp_models(
    study_type:   StudyType,
    dgp1,         avar_models,
    true_sr1:     float,
    null_sr:      float,
    T:            int,
    n_sim:        int,
    alpha:        float,
    th_moments:   bool,
    rng,
    n_jobs:       int   = 1,
    # two-sample only ─────────────────────────────────────────────────────────
    dgp2=None,    avar_models2=None,
    true_sr2:     float = 0.0,
    null_diff:    float = 0.0,
    bias_adj      = False,
):
    """
    Simulate n_sim paths for one DGP (or a pair for two-sample studies)
    and evaluate all models on the exact same paths.

    Parameters
    ----------
    null_sr :
        The value under H₀.
        • Coverage: set equal to true_sr1 (null is true).
        • Power (one-sample): set to a value ≠ true_sr1.
    null_diff :
        Two-sample null: SR₁ − SR₂ ≤ null_diff (default 0).
    """
    th_moms1 = dgp1.get_theo_moments() if th_moments else None
    th_moms2 = dgp2.get_theo_moments() if (th_moments and dgp2 is not None) else None

    # Seed management
    if n_jobs == 1:
        seeds = rng.integers(0, 2**63, size=n_sim)
    else:
        ss       = np.random.SeedSequence(rng.integers(0, 2**63))
        seeds    = ss.spawn(n_sim)

    # Dispatch
    if study_type.is_two_sample:
        assert dgp2 is not None and avar_models2 is not None, \
            "dgp2 and avar_models2 are required for two-sample study types."
        sim_fn = _run_two_sample_path
        kwargs = dict(dgp1=dgp1, dgp2=dgp2,
                      avar_models1=avar_models, avar_models2=avar_models2,
                      T=T, null_diff=null_diff, alpha=alpha,
                      study_type=study_type, th_moments=th_moments,
                      th_moms1=th_moms1, th_moms2=th_moms2, bias_adj=bias_adj)
    else:
        sim_fn = _run_one_sample_path
        kwargs = dict(dgp=dgp1, avar_models=avar_models, T=T,
                      null_sr=null_sr, alpha=alpha,
                      study_type=study_type, th_moments=th_moments,
                      th_moms=th_moms1, bias_adj=bias_adj)

    if n_jobs == 1:
        results_list = [sim_fn(seed, **kwargs) for seed in seeds]
    else:
        results_list = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(sim_fn)(seed, **kwargs) for seed in seeds
        )

    # Assemble
    n_models = len(avar_models)
    sr_hats, widths, V_hats, events = _assemble_results(results_list, n_sim, n_models)

    # Diagnostics
    unique = len(np.unique(sr_hats.round(12)))
    if unique < n_sim:
        print(f"  WARNING: {n_sim - unique} duplicate simulated paths detected.")

    # Summary statistics
    mean_sr  = float(sr_hats.mean())
    bias     = float(mean_sr - true_sr1)
    rmse     = float(np.sqrt(((sr_hats - true_sr1) ** 2).mean()))
    metric   = study_type.metric_name

    results = {}
    for j, model in enumerate(avar_models):
        results[model.short_name] = {
            metric:          float(events[:, j].mean()),
            "mean_sr_hat":   mean_sr,
            "bias":          bias,
            "rmse":          rmse,
            "mean_ci_width": float(np.nanmean(widths[:, j])),
            "mean_V_hat":    float(V_hats[:, j].mean()),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Top-level study runner
# ─────────────────────────────────────────────────────────────────────────────

def run_study(
    study_type:   StudyType,
    dgp_specs:    list[DGPSpec],
    avar_models:  list[AvarModel],
    # primary SR target (DGP calibration)
    target_sr:    float = 0.5,
    calib_mu:     float | None = None,
    calib_sigma:  float | None = None,
    # null hypothesis value; equals target_sr for coverage, differs for power
    null_sr:      float | None = None,
    T:            int   = 500,
    n_sim:        int   = 2000,
    alpha:        float = 0.05,
    th_moments:   bool  = False,
    seed:         int   = 42,
    verbose:      bool  = True,
    n_jobs:       int   = 1,
    # two-sample specific ─────────────────────────────────────────────────────
    dgp_specs2:   list[DGPSpec] | None = None,
    avar_models2: list[AvarModel] | None = None,
    target_sr2:   float = 0.3,
    null_diff:    float = 0.0,
    # bias correction
    bias_adj = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Run a full Monte Carlo study for the chosen StudyType.

    Coverage runs
    -------------
    null_sr is automatically set to target_sr (null is true by construction).

    Power runs
    ----------
    Provide null_sr ≠ target_sr so the DGP operates under the alternative.
    For one-sided power (H₀: SR ≤ θ₀), set target_sr > null_sr.

    Two-sample runs
    ---------------
    Provide dgp_specs2 and avar_models2.  For power, calibrate dgp_specs2
    to target_sr2 < target_sr (so SR₁ > SR₂ in truth).
    null_diff sets the hypothesised difference (default 0).
    """
    # ── defaults ──────────────────────────────────────────────────────────────
    if null_sr is None:
        null_sr = target_sr   # coverage: null equals truth

    master_rng = np.random.default_rng(seed)
    nominal    = 1.0 - alpha
    metric     = study_type.metric_name
    rows       = []

    if th_moments:
        print(f"[info] Using theoretical moments  |  study_type={study_type.name}")

    # ── calibrate primary DGPs ────────────────────────────────────────────────
    calibrated1 = []
    for spec in dgp_specs:
        calibrate_dgp(spec.dgp, target_sr, mu=calib_mu, sigma=calib_sigma)
        calibrated1.append((spec.name, spec.dgp))

    # ── calibrate secondary DGPs (two-sample) ─────────────────────────────────
    calibrated2 = []
    if study_type.is_two_sample:
        assert dgp_specs2 is not None and avar_models2 is not None, (
            "Two-sample study types require dgp_specs2 and avar_models2."
        )
        for spec in dgp_specs2:
            calibrate_dgp(spec.dgp, target_sr2, mu=calib_mu, sigma=calib_sigma)
            print("check 2sample calib")
            calibrated2.append((spec.name, spec.dgp))

        # Pair primary and secondary specs by position
        if len(calibrated2) == 1:
            # broadcast single secondary DGP against all primary DGPs
            calibrated2 = calibrated2 * len(calibrated1)
        assert len(calibrated1) == len(calibrated2), (
            "dgp_specs and dgp_specs2 must have the same length "
            "(or dgp_specs2 may be a single entry broadcast to all)."
        )

    # ── main simulation loop ──────────────────────────────────────────────────
    total = len(calibrated1)
    for idx, (dgp_name, dgp1) in enumerate(calibrated1, 1):
        dgp_rng = np.random.default_rng(master_rng.integers(0, 2**63))

        dgp2_obj  = calibrated2[idx - 1][1] if calibrated2 else None
        dgp2_name = calibrated2[idx - 1][0] if calibrated2 else None

        label = dgp_name if not study_type.is_two_sample else f"{dgp_name} vs {dgp2_name}"

        if verbose:
            print(f"[{idx}/{total}] {study_type.name}  DGP={label:<34} ...")

        res_dict = run_dgp_models(
            study_type   = study_type,
            dgp1         = dgp1,
            avar_models  = avar_models,
            true_sr1     = target_sr,
            null_sr      = null_sr,
            T            = T,
            n_sim        = n_sim,
            alpha        = alpha,
            th_moments   = th_moments,
            rng          = dgp_rng,
            n_jobs       = n_jobs,
            dgp2         = dgp2_obj,
            avar_models2 = avar_models2 or [],
            true_sr2     = target_sr2,
            null_diff    = null_diff,
            bias_adj     = bias_adj,
        )

        for model in avar_models:
            m_name = model.short_name
            res    = res_dict[m_name]
            val    = res[metric]

            if verbose:
                if study_type.is_power:
                    flag = "OK" if val >= 0.5 else "!!"
                    print(f"  -> Model={m_name:<22} power={val:.3f} [{flag}]")
                else:
                    flag = "OK" if abs(val - nominal) < 0.01 else "!!"
                    print(f"  -> Model={m_name:<22} cov={val:.3f} [{flag}]")

            rows.append({
                "study_type":       study_type.name,
                "dgp_name":         label,
                "avar_model":       m_name,
                "nominal":          nominal,
                "null_sr":          null_sr if not study_type.is_two_sample else null_diff,
                "target_sr":        target_sr,
                **res,
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

def pivot_table(results: pd.DataFrame, metric: str = "coverage") -> pd.DataFrame:
    wide = results.pivot(index="dgp_name", columns="avar_model", values=metric)
    wide.index.name = "DGP"
    wide.columns.name = metric
    return wide


def study_report(results: pd.DataFrame, alpha: float = 0.05, tol: float = 0.03) -> str:
    """
    Print a formatted table for whichever metric is present
    (coverage or power).
    """
    nominal    = 1.0 - alpha
    study_type = results["study_type"].iloc[0]
    is_power   = "power" in results.columns
    metric     = "power" if is_power else "coverage"

    cw  = 20
    sep = "=" * 82

    def hdr(models):
        return "  {:<28}".format("DGP") + "".join(f"{m:>{cw}}" for m in models)

    def div(models):
        return "  " + "-" * (28 + cw * len(models))

    if is_power:
        def fmt_primary(v):
            flag = "OK" if v >= 0.5 else "!!"
            return f"{v:.3f} {flag:>2}".rjust(cw)
        primary_title = "EMPIRICAL POWER"
    else:
        def fmt_primary(v):
            flag = "OK" if abs(v - nominal) <= tol else "!!"
            return f"{v:.3f} {flag:>2}".rjust(cw)
        primary_title = "EMPIRICAL COVERAGE"

    lines = [
        sep,
        f"  {study_type}  |  nominal={nominal:.2f}"
        + ("" if is_power else f"  tol=+/-{tol:.2f}"),
        sep,
    ]

    other_metrics = [
        ("bias",          "BIAS of SR_hat",    lambda v: f"{v:+.4f}".rjust(cw)),
        ("rmse",          "RMSE of SR_hat",    lambda v: f"{v:.4f}".rjust(cw)),
        ("mean_ci_width", "MEAN CI WIDTH",     lambda v: f"{v:.4f}".rjust(cw)),
    ]

    for col, title, fmt_fn in [(metric, primary_title, fmt_primary)] + other_metrics:
        if col not in results.columns:
            continue
        tbl    = pivot_table(results, col)
        models = tbl.columns.tolist()
        lines += [f"\n  {title}", hdr(models), div(models)]
        for dgp, row in tbl.iterrows():
            lines.append(f"  {dgp:<28}" + "".join(fmt_fn(v) for v in row))

    lines.append("\n" + sep)
    return "\n".join(lines)

