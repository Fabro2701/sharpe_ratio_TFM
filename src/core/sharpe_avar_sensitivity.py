"""
Sharpe Ratio AVar Sensitivity
==========================================

Two analyses only:

1. **OAT sweeps**: vary one parameter at a time, plus a synthetic
   ``alpha+beta`` axis (persistence sum, ratio held at the base point).
   Infeasible points are dropped — only the feasible region is returned.
   ``omega`` is excluded by default (it usually doesn't appear in the
   AVar formula even when it's in the signature).

2. **Monte Carlo**: uniform sampling within bounds, rejection-sampled
   to the joint feasible region, with quantile + Spearman summary.

The framework remains function-agnostic: pass any ``avar(**params) -> float``,
optionally a list of ``Parameter`` specs (else they are sniffed from the
signature), and a list of ``Constraint`` objects.
"""
from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


# ============================================================================
# CORE
# ============================================================================

@dataclass(frozen=True)
class Parameter:
    name: str
    default: float
    bounds: tuple[float, float]
    description: str = ""
    log_scale: bool = False

    def linspace(self, n: int) -> np.ndarray:
        lo, hi = self.bounds
        return np.geomspace(lo, hi, n) if self.log_scale else np.linspace(lo, hi, n)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        lo, hi = self.bounds
        if self.log_scale:
            return np.exp(rng.uniform(np.log(lo), np.log(hi), size=n))
        return rng.uniform(lo, hi, size=n)


@dataclass(frozen=True)
class Constraint:
    """Slack > 0 means feasible."""
    name: str
    slack: Callable[[Mapping[str, float]], float]
    description: str = ""

    def is_feasible(self, params: Mapping[str, float]) -> bool:
        try:
            return float(self.slack(params)) >= 0.0
        except Exception:
            return False


class StandardConstraints:
    """Common time-series constraints."""

    @staticmethod
    def garch_stationarity(alpha="alpha", beta="beta", tol=1e-9) -> Constraint:
        return Constraint("GARCH stationarity",
                          lambda p: 1.0 - p[alpha] - p[beta] - tol,
                          f"{alpha} + {beta} < 1")

    @staticmethod
    def garch_fourth_moment(alpha="alpha", beta="beta",
                            kurt: Optional[str] = None,
                            innovation_kurt: float = 3.0,
                            tol: float = 1e-9) -> Constraint:
        """kappa_z * alpha^2 + 2*alpha*beta + beta^2 < 1.

        kappa_z = innovation kurtosis. If ``kurt`` names the *return*
        excess-kurtosis parameter, we use kappa_z ≈ kurt + 3 (a conservative
        bound).
        """
        def slack(p):
            kz = (p[kurt] + 3.0) if kurt is not None else innovation_kurt
            return 1.0 - kz * p[alpha] ** 2 - 2 * p[alpha] * p[beta] - p[beta] ** 2 - tol
        return Constraint("GARCH 4th moment", slack,
                          "kappa_z * alpha^2 + 2*alpha*beta + beta^2 < 1")

    @staticmethod
    def ar_stationarity(rho="rho", tol=1e-9) -> Constraint:
        return Constraint("AR(1) stationarity",
                          lambda p: 1.0 - abs(p[rho]) - tol, f"|{rho}| < 1")

    @staticmethod
    def kurtosis_lower_bound(kurt="exc_kurt", tol=1e-9) -> Constraint:
        return Constraint("Kurtosis lower bound",
                          lambda p: p[kurt] + 2.0 - tol, f"{kurt} > -2")

    @staticmethod
    def positive(name: str, tol=1e-9) -> Constraint:
        return Constraint(f"{name} > 0", lambda p: p[name] - tol, f"{name} > 0")


class ParameterSniffer:
    """Auto-build Parameter specs from a function signature."""

    _KNOWN: dict[str, dict[str, Any]] = {
        "sr":       dict(bounds=(-2.0, 2.0),   fallback_default=0.5),
        "omega":    dict(bounds=(1e-6, 1.0),   fallback_default=0.05, log_scale=True),
        "alpha":    dict(bounds=(1e-4, 0.5),   fallback_default=0.08),
        "beta":     dict(bounds=(1e-4, 0.98), fallback_default=0.87),
        "rho":      dict(bounds=(-0.9, 0.9), fallback_default=0.2),
        "phi":      dict(bounds=(-0.9, 0.9), fallback_default=0.2),
        "skew":     dict(bounds=(-3.0, 3.0),   fallback_default=0.0),
        "exc_kurt": dict(bounds=(0.0, 20.0),   fallback_default=3.0),
        "kurt":     dict(bounds=(3.0, 23.0),   fallback_default=6.0),
    }

    @classmethod
    def sniff(cls, fn: Callable) -> list[Parameter]:
        sig = inspect.signature(fn)
        out = []
        for n, p in sig.parameters.items():
            if p.kind == inspect.Parameter.VAR_KEYWORD or n == "self":
                continue
            spec = dict(cls._KNOWN.get(n, {}))
            fb = spec.pop("fallback_default", 0.0)
            default = float(p.default) if p.default is not inspect.Parameter.empty else float(fb)
            if spec:
                out.append(Parameter(name=n, default=default, **spec))
            else:
                lo = default - 1.0 if default else -1.0
                hi = default + 1.0 if default else 1.0
                out.append(Parameter(n, default, (lo, hi), f"(auto) {n}"))
        return out


def _safe_call(fn: Callable, **kw) -> float:
    try:
        with np.errstate(all="ignore"):
            v = float(fn(**kw))
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


@dataclass
class AVarModel:
    """Wraps an arbitrary AVar callable with parameter & constraint metadata."""
    avar_func: Callable[..., float]
    parameters: list[Parameter]
    constraints: list[Constraint] = field(default_factory=list)
    name: str = "AVarModel"

    def __post_init__(self):
        self._pmap = {p.name: p for p in self.parameters}
        sig = inspect.signature(self.avar_func)
        self._needs_filter = not any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        self._sig_names = set(sig.parameters)

    @property
    def param_names(self) -> list[str]:
        return [p.name for p in self.parameters]

    def get_default_point(self) -> dict:
        return {p.name: p.default for p in self.parameters}

    def get_parameter(self, name: str) -> Parameter:
        return self._pmap[name]

    def evaluate(self, **kwargs) -> float:
        full = self.get_default_point() | dict(kwargs)
        if self._needs_filter:
            full = {k: v for k, v in full.items() if k in self._sig_names}
        return _safe_call(self.avar_func, **full)

    def is_feasible(self, params: Mapping[str, float]) -> bool:
        return all(c.is_feasible(params) for c in self.constraints)


# ============================================================================
# 1. OAT (one-at-a-time) — feasible only, plus alpha+beta composite
# ============================================================================

class OATAnalysis:
    """Sweep one parameter at a time. Infeasible points are dropped.

    Also provides ``sweep_alpha_beta_sum`` which moves α and β jointly along
    a ray of constant α/β ratio — the natural axis for studying persistence
    approaching the IGARCH boundary.
    """

    def __init__(self, model: AVarModel):
        self.model = model

    def sweep_param(self, name: str, n: int = 200,
                    base: Optional[dict] = None,
                    bounds: Optional[tuple[float, float]] = None) -> pd.DataFrame:
        """Sweep one parameter; returns only feasible points where AVar is finite."""
        bp = base or self.model.get_default_point()
        spec = self.model.get_parameter(name)
        lo, hi = bounds or spec.bounds
        xs = np.geomspace(lo, hi, n) if spec.log_scale else np.linspace(lo, hi, n)
        rows = []
        for x in xs:
            pt = bp | {name: float(x)}
            if not self.model.is_feasible(pt):
                continue
            v = self.model.evaluate(**pt)
            if np.isfinite(v):
                rows.append({name: float(x), "avar": v})
        return pd.DataFrame(rows)

    def all_sweeps(self, n: int = 200, base: Optional[dict] = None,
                   skip: Iterable[str] = ("omega",)) -> dict[str, pd.DataFrame]:
        """Default sweep set: each parameter (minus ``skip``) + α+β composite."""
        skip_set = set(skip)
        out: dict[str, pd.DataFrame] = {
            p.name: self.sweep_param(p.name, n=n, base=base)
            for p in self.model.parameters if p.name not in skip_set
        }
        return out


# ============================================================================
# 2. Monte Carlo
# ============================================================================

class MonteCarloAnalysis:
    """Uniform random sampling over parameter bounds, rejected on infeasibility."""

    def __init__(self, model: AVarModel):
        self.model = model

    def sample(self, n: int = 10_000, seed: int = 0,
               max_attempts: Optional[int] = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        max_attempts = max_attempts or 50 * n
        rows: list[dict] = []
        attempts = 0
        while len(rows) < n and attempts < max_attempts:
            attempts += 1
            pt = {p.name: float(p.sample(1, rng)[0]) for p in self.model.parameters}
            if not self.model.is_feasible(pt):
                continue
            v = self.model.evaluate(**pt)
            if np.isfinite(v):
                rows.append({**pt, "avar": v})
        if len(rows) < n:
            warnings.warn(f"Only {len(rows)}/{n} feasible samples after {attempts} attempts.")
        return pd.DataFrame(rows)

    def summarize(self, df: pd.DataFrame) -> dict:
        """AVar quantiles + Spearman rank correlations of AVar with each parameter."""
        names = [n for n in self.model.param_names if n in df.columns]
        return {
            "n": len(df),
            "quantiles": df["avar"].quantile([0.05, 0.25, 0.5, 0.75, 0.95, 0.99]),
            "spearman": (df[names].corrwith(df["avar"], method="spearman")
                         .rename("spearman")
                         .sort_values(key=abs, ascending=False)),
        }


# ============================================================================
# Plotting helpers
# ============================================================================

import matplotlib.pyplot as plt
import os

def plot_oat(sweeps: Mapping[str, pd.DataFrame], cols: int = 3,
             figsize: Optional[tuple] = None, model_name: str = ""):
    n = len(sweeps); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize or (4.2 * cols, 3.0 * rows),
                             squeeze=False)
    flat = axes.ravel()
    for ax, (key, df) in zip(flat, sweeps.items()):
        if df.empty:
            ax.text(0.5, 0.5, "no feasible points", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(key); continue
        x_col = key if key in df.columns else df.columns[0]
        ax.plot(df[x_col], df["avar"], lw=2)
        ax.set_xlabel(x_col); ax.set_ylabel("AVar"); ax.set_title(key)
        ax.grid(alpha=0.3)
    for ax in flat[len(sweeps):]:
        ax.set_visible(False)
    if model_name:
        fig.suptitle(f"OAT sensitivity — {model_name}")
    fig.tight_layout()
    return fig


def plot_mc(df: pd.DataFrame, model: AVarModel, cols: int = 3,
            clip_quantile: float = 0.98,
            figsize: Optional[tuple] = None, model_name: str = ""):
    names = [n for n in model.param_names if n in df.columns]
    clip = df["avar"].quantile(clip_quantile)
    d = df[df["avar"] <= clip]
    n = len(names); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize or (4.2 * cols, 3.0 * rows),
                             squeeze=False)
    flat = axes.ravel()
    for ax, name in zip(flat, names):
        ax.scatter(d[name], d["avar"], s=5, alpha=0.4,
                   c=d["avar"], cmap="viridis", linewidth=0)
        ax.set_xlabel(name); ax.set_ylabel("AVar"); ax.grid(alpha=0.3)
    for ax in flat[len(names):]:
        ax.set_visible(False)
    title = f"Monte Carlo (n={len(d)} feasible, top {(1-clip_quantile)*100:.0f}% clipped)"
    fig.suptitle(f"{model_name} — {title}" if model_name else title)
    fig.tight_layout()
    return fig


# ============================================================================
# Facade
# ============================================================================

class AVarSensitivityFramework:
    """One-stop entry point: ``fw.oat`` and ``fw.mc``."""

    def __init__(self, avar_func: Callable,
                 parameters: Optional[Sequence[Parameter]] = None,
                 constraints: Sequence[Constraint] = (),
                 name: str = "model"):
        if parameters is None:
            parameters = ParameterSniffer.sniff(avar_func)
        self.model = AVarModel(avar_func, list(parameters), list(constraints), name)
        self.oat = OATAnalysis(self.model)
        self.mc = MonteCarloAnalysis(self.model)


def run_analysis(fw: AVarSensitivityFramework, save_path=None):
    print(f"Model: {fw.model.name}")
    print(f"Parameters: {fw.model.param_names}")
    print(f"Base point: {fw.model.get_default_point()}")
    print(f"Base AVar: {fw.model.evaluate(**fw.model.get_default_point()):.4f}")

    print("Constraints:")
    for c in fw.model.constraints:
        print(f"  - {c.description}")

    PLOTS = save_path
    os.makedirs(PLOTS, exist_ok=True)
    
    # --- 1. OAT sweeps -------------------------------------------------------
    print("\n" + "=" * 60)
    print("1. OAT SWEEPS (feasible only; omega skipped; alpha+beta added)")
    print("=" * 60)
    sweeps = fw.oat.all_sweeps(n=200)
    for k, df in sweeps.items():
        print(f"  {k:12s}  n={len(df):4d}   "
            f"AVar in [{df['avar'].min():.3f}, {df['avar'].max():.3f}]")
    
    # --- 2. Monte Carlo ------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. MONTE CARLO (n=4000, feasible only)")
    print("=" * 60)
    mc = fw.mc.sample(n=4000, seed=42)
    s = fw.mc.summarize(mc)
    print(f"  drew {s['n']} feasible samples")
    print(f"\n  AVar quantiles:")
    print(s["quantiles"].round(3).to_string())
    print(f"\n  Spearman rank corr(AVar, param):")
    print(s["spearman"].round(4).to_string())

    
    fig = plot_oat(sweeps, cols=3, model_name=fw.model.name)
    if save_path:
        fig.savefig(PLOTS / "oat_sweeps.png", dpi=110, bbox_inches="tight")
    plt.show()
    
    fig = plot_mc(mc, fw.model, cols=3, model_name=fw.model.name)
    if save_path:
        fig.savefig(PLOTS / "mc_scatter.png", dpi=110, bbox_inches="tight")
    plt.show()

# ============================================================================
# Example
# ============================================================================

def _example_garch() -> AVarSensitivityFramework:
    def avar(sr, alpha=0.08, beta=0.87, skew=0.0, exc_kurt=0.0, **kw):
        t1 = skew * (1 - beta) / (1 - alpha - beta)
        t2 = ((exc_kurt + 2) * (1 - beta) ** 2 * (1 + alpha + beta)
              / ((1 - alpha - beta) * (1 - 2 * alpha * beta - beta ** 2)))
        return 1 - sr * t1 + sr ** 2 / 4 * t2

    constraints = [
        StandardConstraints.garch_stationarity(),
        StandardConstraints.garch_fourth_moment(kurt="exc_kurt"),
        StandardConstraints.kurtosis_lower_bound("exc_kurt"),
        #StandardConstraints.positive("omega"),
    ]
    return AVarSensitivityFramework(avar, constraints=constraints, name="GARCH11-symm")

def _example_ar1_garch11():
    """AR(1)-GARCH(1,1) with general (not necessarily symmetric) innovations."""
    def avar(sr, rho=0.2, alpha=0.08, beta=0.87, exc_kurt=0.0, **kw):
        k_r = exc_kurt + 3
        phi2 = rho ** 2
        den_com = 1 - 2 * alpha * beta - beta ** 2
        K = (1 + phi2) * k_r - 5 * phi2 - 1
        num_A = 1 - alpha * beta - beta ** 2
        factor_A = 1 - phi2 * (alpha + beta)
        A = 6 * phi2 * alpha * (num_A / den_com) * (1 / factor_A)
        params_ratio = ((1 - beta) ** 2 * (1 + alpha + beta)) / ((1 - alpha - beta) * den_com)
        S_vv = (1 / (1 - phi2)) * (4 * phi2 + (K / (1 + A)) * ((2 / 3) * A + params_ratio))
        return (1 + rho) / (1 - rho) + 0.25 * sr ** 2 * S_vv

    # Don't pass parameters — let the sniffer demonstrate auto-introspection.
    constraints = [
        StandardConstraints.garch_stationarity(),
        StandardConstraints.garch_fourth_moment(kurt="exc_kurt"),
        StandardConstraints.ar_stationarity("rho"),
        StandardConstraints.kurtosis_lower_bound("exc_kurt"),
        StandardConstraints.positive("sr"),
        #StandardConstraints.positive("omega"),
    ]
    return AVarSensitivityFramework(avar, constraints=constraints, name="AR1-GARCH11")


if __name__ == "__main__":
    fw = _example_ar1_garch11()
    run_analysis(fw, ".")