"""
models.py
=========
Self-contained class definitions for Avar(√T · SR̂) under different
return distribution assumptions.

Design
------
  AvarModel (ABC)
    ├── IIDNormalModel
    ├── IIDNonNormalModel
    ├── AR1NormalModel
    ├── IIDStudentTModel
    └── AR1NonNormalModel

Each concrete class owns:
  • name / short_name / latex_name  — display strings
  • parameters                      — ordered dict of Parameter objects
  • formula_latex                   — LaTeX string of V(θ)
  • references                      — bibliographic list
  • plot_style                      — ready-to-unpack dict for matplotlib
  • __call__(sr, **kwargs)          — evaluate V for scalar or array inputs
  • avar(sr, **kwargs)              — same as __call__, explicit alias
  • sensitivity(param, grid, ...)   — (x, V) sweep with all others at default
  • validate(**kwargs)              — raises ValueError on bad parameter values
  • summary()                       — pretty-print everything

Key references
--------------
  Lo, A. W. (2002). The Statistics of Sharpe Ratios. FAJ 58(4).
  Christie, S. (2005). Is the Sharpe Ratio Useful in Asset Allocation? MAFC.
  Opdyke, J. D. (2007). Comparing Sharpe Ratios. Journal of Asset Management.
"""

from __future__ import annotations

import abc
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Parameter descriptor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Parameter:
    """
    Full specification of a single model parameter.

    Attributes
    ----------
    name : str
        Python identifier used as keyword argument (e.g. ``"rho"``).
    default : float
        Default / baseline value.
    domain : tuple[float, float]
        Hard domain (open/closed implied by ``domain_open``).
    domain_open : tuple[bool, bool]
        ``(left_open, right_open)`` — True means the endpoint is excluded.
    plot_range : tuple[float, float]
        Suggested x-axis range for sensitivity plots.
    label : str
        Short human-readable label (e.g. ``"ρ"``).
    latex : str
        LaTeX symbol (e.g. ``r"\\rho"``).
    description : str
        One-line description.
    n_points : int
        Default number of grid points for sensitivity sweeps.
    """

    name: str
    default: float
    domain: tuple[float, float]
    domain_open: tuple[bool, bool] = (False, False)
    plot_range: tuple[float, float] | None = None
    label: str = ""
    latex: str = ""
    description: str = ""
    n_points: int = 400

    def __post_init__(self):
        if self.label == "":
            self.label = self.name
        if self.latex == "":
            self.latex = self.name
        if self.plot_range is None:
            self.plot_range = self.domain

    # ── validation ────────────────────────────────────────────────────────

    def check(self, value: float) -> None:
        """Raise ValueError if *value* is outside the domain."""
        lo, hi = self.domain
        lo_open, hi_open = self.domain_open
        lo_ok = value > lo if lo_open else value >= lo
        hi_ok = value < hi if hi_open else value <= hi
        if not (lo_ok and hi_ok):
            lo_sym = "(" if lo_open else "["
            hi_sym = ")" if hi_open else "]"
            raise ValueError(
                f"Parameter '{self.name}' = {value} is outside "
                f"domain {lo_sym}{lo}, {hi}{hi_sym}."
            )

    # ── convenience ───────────────────────────────────────────────────────

    def grid(self, n: int | None = None) -> np.ndarray:
        """Return a uniform grid over ``plot_range``."""
        n = n or self.n_points
        return np.linspace(self.plot_range[0], self.plot_range[1], n)

    def __repr__(self) -> str:
        return (
            f"Parameter({self.name!r}, default={self.default}, "
            f"domain={self.domain}, plot_range={self.plot_range})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class AvarModel(abc.ABC):
    """
    Abstract base class for Avar(√T · SR̂) models.

    Subclasses must define
    ----------------------
    name           : str
    short_name     : str
    latex_name     : str
    formula_latex  : str
    references     : list[str]
    parameters     : dict[str, Parameter]   (excludes sr — always present)
    plot_style     : dict                   (matplotlib line kwargs)

    and implement

    _avar(sr, **kwargs) -> float | np.ndarray
        Core formula.  Receives already-validated scalar or vectorised inputs.
    """

    # ── class-level attributes (overridden in each subclass) ──────────────

    #: Full display name
    name: str = ""
    #: Short identifier (safe for filenames / dict keys)
    short_name: str = ""
    #: LaTeX name for figures
    latex_name: str = ""
    #: LaTeX string of the variance formula
    formula_latex: str = ""
    #: Bibliographic references
    references: list[str] = []
    #: Parameter specs (does NOT include sr — handled separately)
    parameters: dict[str, Parameter] = {}
    #: Default matplotlib line style
    plot_style: dict[str, Any] = {}

    # sr is universal — every model depends on it
    _sr_param: Parameter = Parameter(
        name="sr",
        default=0.5,
        domain=(0.0, np.inf),
        domain_open=(True, True),
        plot_range=(0.0, 2.5),
        label="SR",
        latex=r"\theta",
        description="Annualised Sharpe ratio",
    )

    # ── abstract interface ────────────────────────────────────────────────

    @abc.abstractmethod
    def _avar(self, sr: float | np.ndarray, **kwargs) -> float | np.ndarray:
        """Return V(θ).  Called after validation."""

    # ── public API ────────────────────────────────────────────────────────

    def avar(self, sr: float | np.ndarray, **kwargs) -> float | np.ndarray:
        """
        Evaluate Avar(√T · SR̂).

        Parameters
        ----------
        sr : float or array-like
            Sharpe ratio value(s).
        **kwargs
            Model-specific parameters (unrecognised keys are silently dropped
            so a common baseline dict can be passed to all models).

        Returns
        -------
        float or np.ndarray
            Asymptotic variance V(θ).
        """
        # drop keys unknown to this model (allows broadcasting a global dict)
        known = {k: v for k, v in kwargs.items() if k in self.parameters}
        # fill defaults for missing keys
        defaults = {k: p.default for k, p in self.parameters.items()}
        merged = {**defaults, **known}
        return self._avar(sr, **merged)

    def __call__(self, sr: float | np.ndarray, **kwargs) -> float | np.ndarray:
        """Alias for :meth:`avar`."""
        return self.avar(sr, **kwargs)

    def validate(self, sr: float | None = None, **kwargs) -> None:
        """
        Validate parameter values against their declared domains.

        Raises
        ------
        ValueError
            If any value is outside its domain.
        """
        if sr is not None:
            self._sr_param.check(sr)
        for k, v in kwargs.items():
            if k in self.parameters:
                self.parameters[k].check(v)

    def sensitivity(
        self,
        param: str,
        grid: np.ndarray | None = None,
        n: int | None = None,
        **fixed,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute a 1-D sensitivity sweep of V w.r.t. *param*.

        Parameters
        ----------
        param : str
            Parameter name to sweep (``"sr"`` is valid).
        grid  : array-like, optional
            Explicit grid values.  If None, ``Parameter.grid(n)`` is used.
        n     : int, optional
            Number of points (overrides Parameter.n_points).
        **fixed
            Values for all other parameters (defaults used for anything missing).

        Returns
        -------
        x : np.ndarray   — grid over swept parameter
        V : np.ndarray   — corresponding Avar values
        """
        # resolve grid
        if param == "sr":
            p = self._sr_param
        elif param in self.parameters:
            p = self.parameters[param]
        else:
            raise KeyError(f"Unknown parameter '{param}' for model '{self.name}'.")
        x = np.asarray(grid) if grid is not None else p.grid(n)

        # build fixed kwargs (exclude the swept param)
        defaults = {k: q.default for k, q in self.parameters.items()}
        merged   = {**defaults, **{k: v for k, v in fixed.items()
                                   if k in self.parameters}}

        if param == "sr":
            V = np.array([self._avar(xi, **merged) for xi in x], dtype=float)
        else:
            sr_val = fixed.get("sr", self._sr_param.default)
            merged.pop(param, None)
            V = np.array(
                [self._avar(sr_val, **{param: xi}, **merged) for xi in x],
                dtype=float,
            )
        return x, V

    def all_parameters(self) -> dict[str, Parameter]:
        """Return the full parameter dict including *sr*."""
        return {"sr": self._sr_param, **self.parameters}

    def defaults(self) -> dict[str, float]:
        """Return a dict of ``{param_name: default}`` for all parameters (incl. sr)."""
        return {k: p.default for k, p in self.all_parameters().items()}

    # ── display ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a formatted multi-line summary string."""
        lines = [
            f"{'─'*62}",
            f"  {self.name}",
            f"  ({self.short_name})",
            f"{'─'*62}",
            f"  Formula : V = {self.formula_latex}",
            "",
            "  Parameters",
            "  ──────────",
        ]
        for p in self.all_parameters().values():
            lo_sym = "(" if p.domain_open[0] else "["
            hi_sym = ")" if p.domain_open[1] else "]"
            lo, hi = p.domain
            lo_str = "-∞" if lo == -np.inf else str(lo)
            hi_str = "+∞" if hi ==  np.inf else str(hi)
            lines.append(
                f"    {p.name:<12} default={p.default:<8}  "
                f"domain={lo_sym}{lo_str},{hi_str}{hi_sym}  "
                f"plot={p.plot_range}  — {p.description}"
            )
        lines += ["", "  References", "  ──────────"]
        for ref in self.references:
            lines.append("    " + textwrap.fill(ref, width=58,
                                                subsequent_indent="      "))
        lines.append(f"{'─'*62}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={p.default}" for k, p in self.parameters.items()
        )
        return f"{self.__class__.__name__}({params})"

    def __str__(self) -> str:
        return self.name


# ─────────────────────────────────────────────────────────────────────────────
# Concrete models
# ─────────────────────────────────────────────────────────────────────────────

class IIDNormalModel(AvarModel):
    """
    IID Normal returns.

    V = 1 + θ²/2

    Under Gaussianity the third cumulant vanishes (γ₁ = 0) and the excess
    kurtosis is zero (γ₂ = 0), so all non-Gaussian correction terms drop out.

    Reference: Lo (2002), Eq. (6).
    """

    name          = "IID Normal"
    short_name    = "iid_normal"
    latex_name    = r"IID $\mathcal{N}$"
    formula_latex = r"1 + \theta^2/2"
    references    = [
        "Lo, A. W. (2002). The Statistics of Sharpe Ratios. "
        "Financial Analysts Journal, 58(4), 36-52.",
    ]
    parameters    = {}          # no extra parameters
    plot_style    = {"color": "#1E88E5", "ls": "-",  "lw": 2.2}

    def _avar(self, sr, **kw):
        return 1.0 + sr**2 / 2.0


class IIDNonNormalModel(AvarModel):
    """
    IID Non-Normal returns (general moments).

    V = 1 + θ²/2 − θ·γ₁ + θ²·γ₂/4

    Derivation (delta method):
      Σ_∞ for iid returns = diag(σ², σ⁴(2 + γ₂)) with off-diagonal μ₃ = σ³γ₁.
      ∇g = [1/σ, −θ/(2σ²)]  ⟹  V = ∇g·Σ_∞·∇gᵀ / σ² gives the formula above.

    Reference: Opdyke (2007).
    """

    name          = "IID Non-Normal"
    short_name    = "iid_nonnormal"
    latex_name    = r"IID Non-$\mathcal{N}$"
    formula_latex = r"1 + \theta^2/2 - \theta\gamma_1 + \theta^2\gamma_2/4"
    references    = [
        "Opdyke, J. D. (2007). Comparing Sharpe Ratios: So Where Are the "
        "p-Values? Journal of Asset Management, 8(5), 308-336.",
        "Christie, S. (2005). Is the Sharpe Ratio Useful in Asset Allocation? "
        "MAFC Research Paper No. 31.",
    ]
    parameters    = {
        "skew": Parameter(
            name="skew",
            default=0.0,
            domain=(-np.inf, np.inf),
            domain_open=(False, False),
            plot_range=(-3.0, 3.0),
            label="γ₁",
            latex=r"\gamma_1",
            description="Standardised skewness μ₃/σ³",
        ),
        "exc_kurt": Parameter(
            name="exc_kurt",
            default=0.0,
            domain=(-2.0, np.inf),
            domain_open=(False, True),
            plot_range=(0.0, 12.0),
            label="γ₂",
            latex=r"\gamma_2",
            description="Excess kurtosis μ₄/σ⁴ − 3  (≥ −2 by Pearson bound)",
        ),
    }
    plot_style    = {"color": "#E53935", "ls": "--", "lw": 2.0}

    def _avar(self, sr, skew=0.0, exc_kurt=0.0, **kw):
        return 1.0 + sr**2 / 2.0 - sr * skew + sr**2 * exc_kurt / 4.0


class AR1NormalModel(AvarModel):
    """
    AR(1) with Normal innovations.

    Model: r_t = μ + u_t,  u_t = ρ u_{t-1} + η_t,  η_t ~ N(0, σ_η²)

    Long-run (spectral) covariance matrix of (μ̂, σ̂²):
      Avar(μ̂)  = σ²(1+ρ)/(1−ρ)      [Bartlett / spectral density at 0]
      Avar(σ̂²) = 2σ⁴/(1−ρ²)          [Gaussian 4th cumulant formula]
      Acov      = 0                    [3rd cumulant = 0 under Normality]

    V = (1+ρ)/(1−ρ) + θ²/[2(1−ρ²)]

    Check: ρ → 0 ⟹ V = 1 + θ²/2  (IID Normal). ✓

    References: Lo (2002); Christie (2005).
    """

    name          = "AR(1) Normal"
    short_name    = "ar1_normal"
    latex_name    = r"AR(1) $\mathcal{N}$"
    formula_latex = r"\frac{1+\rho}{1-\rho} + \frac{\theta^2}{2(1-\rho^2)}"
    references    = [
        "Lo, A. W. (2002). The Statistics of Sharpe Ratios. "
        "Financial Analysts Journal, 58(4), 36-52.",
        "Christie, S. (2005). Is the Sharpe Ratio Useful in Asset Allocation? "
        "MAFC Research Paper No. 31.",
    ]
    parameters    = {
        "rho": Parameter(
            name="rho",
            default=0.2,
            domain=(-1.0, 1.0),
            domain_open=(True, True),
            plot_range=(-0.85, 0.85),
            label="ρ",
            latex=r"\rho",
            description="AR(1) autocorrelation coefficient  |ρ| < 1",
        ),
    }
    plot_style    = {"color": "#43A047", "ls": "-.", "lw": 2.0}

    def _avar(self, sr, rho=0.2, **kw):
        rho = np.asarray(rho, dtype=float)
        # clip to avoid division by zero (only relevant if caller bypasses validate)
        rho = np.clip(rho, -1 + 1e-9, 1 - 1e-9)
        return (1.0 + rho) / (1.0 - rho) + sr**2 / (2.0 * (1.0 - rho**2))


class IIDStudentTModel(AvarModel):
    """
    IID Student-t(ν) returns.

    Moment constraints for t(ν):
      γ₁ = 0  (symmetric distribution)
      γ₂ = 6/(ν−4)  for ν > 4  (finite kurtosis requirement)

    Substituting into the IID Non-Normal formula:

    V = 1 + θ²/2 + 3θ²/[2(ν−4)]

    Requires ν > 4 for the asymptotic variance to be finite.

    References: Christie (2005); Opdyke (2007).
    """

    name          = "IID Student-t"
    short_name    = "iid_student_t"
    latex_name    = r"IID $t_\nu$"
    formula_latex = r"1 + \theta^2/2 + \frac{3\theta^2}{2(\nu-4)}"
    references    = [
        "Christie, S. (2005). Is the Sharpe Ratio Useful in Asset Allocation? "
        "MAFC Research Paper No. 31.",
        "Opdyke, J. D. (2007). Comparing Sharpe Ratios: So Where Are the "
        "p-Values? Journal of Asset Management, 8(5), 308-336.",
    ]
    parameters    = {
        "nu": Parameter(
            name="nu",
            default=10.0,
            domain=(4.0, np.inf),
            domain_open=(True, True),
            plot_range=(4.05, 40.0),
            label="ν",
            latex=r"\nu",
            description="Degrees of freedom  (ν > 4 for finite kurtosis)",
        ),
    }
    plot_style    = {"color": "#FB8C00", "ls": ":", "lw": 2.2}

    def _avar(self, sr, nu=10.0, **kw):
        nu = np.asarray(nu, dtype=float)
        exc_kurt = np.where(nu > 4, 6.0 / (nu - 4.0), np.nan)
        return 1.0 + sr**2 / 2.0 + sr**2 * exc_kurt / 4.0


class AR1NonNormalModel(AvarModel):
    """
    AR(1) with Non-Normal innovations (general case).

    When innovations η_t are non-Gaussian, the long-run cross-covariance
    between μ̂ and σ̂² no longer vanishes.  For a linear AR(1) with
    innovation skewness γ₁ and excess kurtosis γ₂:

      Acov_∞(μ̂, σ̂²) ≈ σ³γ₁ / (1−ρ)²     [3rd cumulant channel]
      Avar_∞(σ̂²)     ≈ σ⁴(2 + γ₂)/(1−ρ²)  [4th cumulant channel]

    Applying ∇g = [1/σ, −θ/(2σ²)] yields:

    V = (1+ρ)/(1−ρ) + θ²/[2(1−ρ²)] − θγ₁/(1−ρ)² + θ²γ₂/[4(1−ρ²)]

    NOTE: This is a first-order approximation; the exact formula requires
    the full higher-order cumulant spectral density.

    Checks:
      γ₁=0, γ₂=0, ρ=0  ⟹  IID Normal  ✓
      γ₁=0, γ₂=0        ⟹  AR(1) Normal ✓
      ρ=0                ⟹  IID Non-Normal ✓

    Reference: Opdyke (2007); Christie (2005).
    """

    name          = "AR(1) Non-Normal"
    short_name    = "ar1_nonnormal"
    latex_name    = r"AR(1) Non-$\mathcal{N}$"
    formula_latex = (
        r"\frac{1+\rho}{1-\rho} + \frac{\theta^2}{2(1-\rho^2)}"
        r" - \frac{\theta\gamma_1}{(1-\rho)^2}"
        r" + \frac{\theta^2\gamma_2}{4(1-\rho^2)}"
    )
    references    = [
        "Opdyke, J. D. (2007). Comparing Sharpe Ratios: So Where Are the "
        "p-Values? Journal of Asset Management, 8(5), 308-336.",
        "Christie, S. (2005). Is the Sharpe Ratio Useful in Asset Allocation? "
        "MAFC Research Paper No. 31.",
    ]
    parameters    = {
        "rho": Parameter(
            name="rho",
            default=0.2,
            domain=(-1.0, 1.0),
            domain_open=(True, True),
            plot_range=(-0.85, 0.85),
            label="ρ",
            latex=r"\rho",
            description="AR(1) autocorrelation coefficient  |ρ| < 1",
        ),
        "skew": Parameter(
            name="skew",
            default=0.0,
            domain=(-np.inf, np.inf),
            domain_open=(False, False),
            plot_range=(-3.0, 3.0),
            label="γ₁",
            latex=r"\gamma_1",
            description="Innovation standardised skewness",
        ),
        "exc_kurt": Parameter(
            name="exc_kurt",
            default=0.0,
            domain=(-2.0, np.inf),
            domain_open=(False, True),
            plot_range=(0.0, 12.0),
            label="γ₂",
            latex=r"\gamma_2",
            description="Innovation excess kurtosis (≥ −2)",
        ),
    }
    plot_style    = {"color": "#8E24AA", "ls": (0, (3, 1, 1, 1)), "lw": 1.8}

    def _avar(self, sr, rho=0.2, skew=0.0, exc_kurt=0.0, **kw):
        rho = np.asarray(rho, dtype=float)
        rho = np.clip(rho, -1 + 1e-9, 1 - 1e-9)
        base   = (1.0 + rho)  / (1.0 - rho)
        var_c  =  sr**2        / (2.0 * (1.0 - rho**2))
        skew_c = -sr * skew   / (1.0 - rho)**2
        kurt_c =  sr**2 * exc_kurt / (4.0 * (1.0 - rho**2))
        return base + var_c + skew_c + kurt_c


# ─────────────────────────────────────────────────────────────────────────────
# Registry  —  single source of truth for downstream modules
# ─────────────────────────────────────────────────────────────────────────────

#: Ordered dict of all built-in model instances.
REGISTRY: dict[str, AvarModel] = {
    m.short_name: m
    for m in [
        IIDNormalModel(),
        IIDNonNormalModel(),
        AR1NormalModel(),
        IIDStudentTModel(),
        AR1NonNormalModel(),
    ]
}


def get_model(key: str) -> AvarModel:
    """
    Retrieve a model instance by short name.

    Parameters
    ----------
    key : str
        ``short_name`` of the model (e.g. ``"ar1_normal"``).

    Returns
    -------
    AvarModel
    """
    if key not in REGISTRY:
        raise KeyError(
            f"Unknown model '{key}'.  Available: {list(REGISTRY.keys())}"
        )
    return REGISTRY[key]


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for model in REGISTRY.values():
        print(model.summary())
        print()

    # demonstrate the API
    m = get_model("ar1_normal")
    print(f"\n{m.name}  at sr=0.5, rho=0.3 → V = {m(0.5, rho=0.3):.4f}")
    print(f"Defaults: {m.defaults()}")

    # sensitivity sweep
    x, V = m.sensitivity("rho", sr=0.5)
    print(f"\nsensitivity('rho') → x[0]={x[0]:.2f}, x[-1]={x[-1]:.2f}, "
          f"V[0]={V[0]:.4f}, V[-1]={V[-1]:.4f}")

    # broadcast a global parameter dict (unknown keys silently dropped)
    global_params = dict(sr=0.7, rho=0.15, skew=-0.5, exc_kurt=3.0, nu=12.0)
    print("\nBroadcast global_params to all models:")
    for model in REGISTRY.values():
        sr = global_params.pop("sr")
        v  = model(sr, **global_params)
        global_params["sr"] = sr
        print(f"  {model.name:<22} V = {v:.4f}")