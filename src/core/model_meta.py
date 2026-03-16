"""
model_meta.py
=============
Parameter specs, display metadata, and the REGISTRY for all models.

This module owns:
  • Parameter                — descriptor for a single model parameter
  • _SR_PARAM                — universal Sharpe-ratio parameter spec
  • _DISPLAY_META            — per-model parameter specs + plot/display fields
  • ModelMeta                — wrapper adding sensitivity(), summary(), etc.
  • REGISTRY / get_model()   — single source of truth for downstream modules

To add a new model:
  1. Add the class to models.py (just _avar, fit, param_names).
  2. Add a ``short_name -> {...}`` entry to _DISPLAY_META below.
  3. Add an instance to MODEL_CLASSES in models.py.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.models import AvarModel, MODEL_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
# Parameter descriptor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Parameter:
    """
    Full specification of a single model parameter.

    Attributes
    ----------
    name        : Python identifier used as kwarg (e.g. "rho").
    default     : Baseline / fallback value.
    domain      : Hard (lo, hi) bounds.
    domain_open : (left_open, right_open) — True means endpoint excluded.
    plot_range  : Suggested x-axis range for sensitivity plots.
    label       : Short human-readable label (e.g. "ρ").
    latex       : LaTeX symbol (e.g. r"\rho").
    description : One-line description.
    n_points    : Default grid size for sensitivity sweeps.
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

    def check(self, value: float) -> None:
        """Raise ValueError if *value* is outside the declared domain."""
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

    def grid(self, n: int | None = None) -> np.ndarray:
        """Uniform grid over ``plot_range``."""
        return np.linspace(self.plot_range[0], self.plot_range[1], n or self.n_points)

    def __repr__(self) -> str:
        return (
            f"Parameter({self.name!r}, default={self.default}, "
            f"domain={self.domain}, plot_range={self.plot_range})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Universal sr parameter
# ─────────────────────────────────────────────────────────────────────────────

_SR_PARAM = Parameter(
    name="sr",
    default=0.5,
    domain=(0.0, np.inf),
    domain_open=(True, True),
    plot_range=(0.0, 2.5),
    label="SR",
    latex=r"\theta",
    description="Annualised Sharpe ratio",
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared parameter specs (reused across models)
# ─────────────────────────────────────────────────────────────────────────────

_P_RHO = Parameter(
    name="rho", default=0.2,
    domain=(-1.0, 1.0), domain_open=(True, True),
    plot_range=(-0.85, 0.85),
    label="ρ", latex=r"\rho",
    description="AR(1) autocorrelation coefficient  |ρ| < 1",
)
_P_SKEW = Parameter(
    name="skew", default=0.0,
    domain=(-np.inf, np.inf), domain_open=(False, False),
    plot_range=(-3.0, 3.0),
    label="γ₁", latex=r"\gamma_1",
    description="Standardised skewness μ₃/σ³",
)
_P_EXC_KURT = Parameter(
    name="exc_kurt", default=0.0,
    domain=(-2.0, np.inf), domain_open=(False, True),
    plot_range=(0.0, 12.0),
    label="γ₂", latex=r"\gamma_2",
    description="Excess kurtosis μ₄/σ⁴ − 3  (≥ −2 by Pearson bound)",
)
_P_NU = Parameter(
    name="nu", default=10.0,
    domain=(4.0, np.inf), domain_open=(True, True),
    plot_range=(4.05, 40.0),
    label="ν", latex=r"\nu",
    description="Degrees of freedom  (ν > 4 for finite kurtosis)",
)

# ─────────────────────────────────────────────────────────────────────────────
# Per-model metadata
# ─────────────────────────────────────────────────────────────────────────────
# Each entry maps short_name -> {parameters, latex_name, plot_style, references,
#                                 formula_latex}
# parameters: ordered dict matching the model's param_names tuple.

_DISPLAY_META: dict[str, dict[str, Any]] = {
    "iid_normal": {
        "parameters":    {},
        "formula_latex": r"1 + \theta^2/2",
        "latex_name":    r"IID $\mathcal{N}$",
        "plot_style":    {"color": "#1E88E5", "ls": "-",  "lw": 2.2},
        "references": [
            "Lo, A. W. (2002). The Statistics of Sharpe Ratios. "
            "Financial Analysts Journal, 58(4), 36-52.",
        ],
    },
    "iid_nonnormal": {
        "parameters":    {"skew": _P_SKEW, "exc_kurt": _P_EXC_KURT},
        "formula_latex": r"1 + \theta^2/2 - \theta\gamma_1 + \theta^2\gamma_2/4",
        "latex_name":    r"IID Non-$\mathcal{N}$",
        "plot_style":    {"color": "#E53935", "ls": "--", "lw": 2.0},
        "references": [
            "Opdyke, J. D. (2007). Comparing Sharpe Ratios: So Where Are the "
            "p-Values? Journal of Asset Management, 8(5), 308-336.",
            "Christie, S. (2005). Is the Sharpe Ratio Useful in Asset Allocation? "
            "MAFC Research Paper No. 31.",
        ],
    },
    "ar1_normal": {
        "parameters":    {"rho": _P_RHO},
        "formula_latex": r"\frac{1+\rho}{1-\rho} + \frac{\theta^2}{2(1-\rho^2)}",
        "latex_name":    r"AR(1) $\mathcal{N}$",
        "plot_style":    {"color": "#43A047", "ls": "-.", "lw": 2.0},
        "references": [
            "Lo, A. W. (2002). The Statistics of Sharpe Ratios. "
            "Financial Analysts Journal, 58(4), 36-52.",
            "Christie, S. (2005). Is the Sharpe Ratio Useful in Asset Allocation? "
            "MAFC Research Paper No. 31.",
        ],
    },
    "iid_student_t": {
        "parameters":    {"exc_kurt": _P_EXC_KURT},
        "formula_latex": r"1 + \theta^2/2 + \frac{3\theta^2}{2(\nu-4)}",
        "latex_name":    r"IID $t_\nu$",
        "plot_style":    {"color": "#FB8C00", "ls": ":", "lw": 2.2},
        "references": [
            "Christie, S. (2005). Is the Sharpe Ratio Useful in Asset Allocation? "
            "MAFC Research Paper No. 31.",
            "Opdyke, J. D. (2007). Comparing Sharpe Ratios: So Where Are the "
            "p-Values? Journal of Asset Management, 8(5), 308-336.",
        ],
    },
    "ar1_nonnormal": {
        "parameters":    {"rho": _P_RHO, "skew": _P_SKEW, "exc_kurt": _P_EXC_KURT},
        "formula_latex": (
            r"\frac{1+\rho}{1-\rho} + \frac{\theta^2}{2(1-\rho^2)}"
            r" - \frac{\theta\gamma_1}{(1-\rho)^2}"
            r" + \frac{\theta^2\gamma_2}{4(1-\rho^2)}"
        ),
        "latex_name":    r"AR(1) Non-$\mathcal{N}$",
        "plot_style":    {"color": "#8E24AA", "ls": (0, (3, 1, 1, 1)), "lw": 1.8},
        "references": [
            "Opdyke, J. D. (2007). Comparing Sharpe Ratios: So Where Are the "
            "p-Values? Journal of Asset Management, 8(5), 308-336.",
            "Christie, S. (2005). Is the Sharpe Ratio Useful in Asset Allocation? "
            "MAFC Research Paper No. 31.",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# ModelMeta — wrapper adding display / plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

class ModelMeta:
    """
    Wraps an AvarModel with display and plotting helpers.

    avar() / fit() / __call__() are forwarded unchanged to the underlying
    model, so this can be used as a drop-in wherever an AvarModel is expected.
    """

    def __init__(
        self,
        model: AvarModel,
        *,
        parameters: dict[str, Parameter],
        formula_latex: str = "",
        latex_name: str = "",
        plot_style: dict[str, Any] | None = None,
        references: list[str] | None = None,
    ):
        self._model       = model
        self.parameters   = parameters          # model-specific Parameter specs
        self.formula_latex = formula_latex
        self.latex_name   = latex_name or model.name
        self.plot_style   = plot_style or {}
        self.references   = references or []

    # ── proxy identity ────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._model.name

    @property
    def short_name(self) -> str:
        return self._model.short_name

    # ── proxy compute ─────────────────────────────────────────────────────

    def avar(self, sr, **kwargs):
        return self._model.avar(sr, **kwargs)

    def __call__(self, sr, **kwargs):
        return self._model(sr, **kwargs)

    def fit(self, x):
        return self._model.fit(x)

    # ── sr parameter ──────────────────────────────────────────────────────

    @property
    def _sr_param(self) -> Parameter:
        return _SR_PARAM

    # ── display / plotting helpers ────────────────────────────────────────

    def all_parameters(self) -> dict[str, Parameter]:
        """Full parameter dict including *sr*."""
        return {"sr": _SR_PARAM, **self.parameters}

    def defaults(self) -> dict[str, float]:
        """{name: default} for every parameter (including sr)."""
        return {k: p.default for k, p in self.all_parameters().items()}

    def validate(self, sr: float | None = None, **kwargs) -> None:
        """Raise ValueError if any value lies outside its declared domain."""
        if sr is not None:
            _SR_PARAM.check(sr)
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
        1-D sensitivity sweep of V w.r.t. *param*.

        Parameters
        ----------
        param  : parameter name to sweep (``"sr"`` is valid)
        grid   : explicit grid; if None uses Parameter.grid(n)
        n      : grid size override
        **fixed: values for all other parameters (defaults fill the rest)

        Returns
        -------
        x, V : np.ndarray pair
        """
        all_params = self.all_parameters()
        if param not in all_params:
            raise KeyError(
                f"Unknown parameter '{param}' for model '{self.name}'. "
                f"Available: {list(all_params)}"
            )
        p = all_params[param]
        x = np.asarray(grid) if grid is not None else p.grid(n)

        defaults = {k: q.default for k, q in self.parameters.items()}
        merged   = {**defaults, **{k: v for k, v in fixed.items()
                                   if k in self.parameters}}
        if param == "sr":
            V = np.array([self._model._avar(xi, **merged) for xi in x], dtype=float)
        else:
            sr_val = fixed.get("sr", _SR_PARAM.default)
            merged.pop(param, None)
            V = np.array(
                [self._model._avar(sr_val, **{param: xi}, **merged) for xi in x],
                dtype=float,
            )
        return x, V

    def summary(self) -> str:
        """Formatted multi-line summary."""
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
            lines.append(
                "    " + textwrap.fill(ref, width=58, subsequent_indent="      ")
            )
        lines.append(f"{'─'*62}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ModelMeta({self._model!r})"

    def __str__(self) -> str:
        return self.name


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

def _build_registry() -> dict[str, ModelMeta]:
    registry = {}
    for m in MODEL_CLASSES:
        meta = _DISPLAY_META.get(m.short_name, {})
        params: dict[str, Parameter] = meta.get("parameters", {})
        # inject defaults back into the model so avar() can resolve them
        m._defaults = {k: p.default for k, p in params.items()}
        registry[m.short_name] = ModelMeta(
            m,
            parameters    = params,
            formula_latex = meta.get("formula_latex", ""),
            latex_name    = meta.get("latex_name", m.name),
            plot_style    = meta.get("plot_style", {}),
            references    = meta.get("references", []),
        )
    return registry


REGISTRY: dict[str, ModelMeta] = _build_registry()


def get_model(key: str) -> ModelMeta:
    """Retrieve a wrapped model by short name (e.g. ``"ar1_normal"``)."""
    if key not in REGISTRY:
        raise KeyError(
            f"Unknown model '{key}'.  Available: {list(REGISTRY.keys())}"
        )
    return REGISTRY[key]


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for m in REGISTRY.values():
        print(m.summary())
        print()

    m = get_model("ar1_normal")
    print(f"\n{m.name}  at sr=0.5, rho=0.3 → V = {m(0.5, rho=0.3):.4f}")
    print(f"Defaults: {m.defaults()}")

    x, V = m.sensitivity("rho", sr=0.5)
    print(f"\nsensitivity('rho') → x[0]={x[0]:.2f}, x[-1]={x[-1]:.2f}, "
          f"V[0]={V[0]:.4f}, V[-1]={V[-1]:.4f}")