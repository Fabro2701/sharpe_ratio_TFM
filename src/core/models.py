"""
models.py
=========
Core computation classes for Avar(√T · SR̂).

Each class owns:
  • _avar(sr, **kwargs)  — formula implementation
  • fit(x)              — parameter estimation from data
  • name / short_name   — identity strings
  • param_names         — tuple of kwarg names accepted by _avar (no defaults
                          here; defaults live in model_meta.py with the specs)

Parameter specs, display metadata, and the registry wrapper all live in
model_meta.py.  Import from there for anything beyond raw computation.
"""

from __future__ import annotations

import abc

import numpy as np
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class AvarModel(abc.ABC):
    """
    Abstract base for Avar(√T · SR̂) models.

    Subclasses must define
    ----------------------
    name       : str
    short_name : str
    param_names: tuple[str, ...]   kwargs accepted by _avar (excluding sr)

    and implement

    _avar(sr, **kwargs) -> float | np.ndarray
    fit(x)              -> dict[str, float]
    """

    name: str = ""
    short_name: str = ""
    param_names: tuple[str, ...] = ()

    @abc.abstractmethod
    def _avar(self, sr: float | np.ndarray, **kwargs) -> float | np.ndarray:
        """Return V(θ). Receives already-defaulted kwargs."""

    @abc.abstractmethod
    def fit(self, x: np.ndarray) -> dict[str, float]:
        """Estimate model-specific parameters from return series *x*."""

    def avar(self, sr: float | np.ndarray, **kwargs) -> float | np.ndarray:
        """
        Evaluate Avar(√T · SR̂).

        Unknown kwargs are silently dropped so a global baseline dict can be
        broadcast to all models without error.  Defaults are resolved here
        from the Parameter specs in model_meta.py (injected at registry-build
        time via _set_defaults).
        """
        known    = {k: v for k, v in kwargs.items() if k in self.param_names}
        return self._avar(sr, **{**self._defaults, **known})

    def __call__(self, sr: float | np.ndarray, **kwargs) -> float | np.ndarray:
        return self.avar(sr, **kwargs)

    # _defaults is populated by model_meta._build_registry(); safe fallback = {}
    _defaults: dict[str, float] = {}

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self._defaults.items())
        return f"{self.__class__.__name__}({params})"

    def __str__(self) -> str:
        return self.name


# ─────────────────────────────────────────────────────────────────────────────
# Concrete models
# ─────────────────────────────────────────────────────────────────────────────

class IIDNormalModel(AvarModel):
    """IID Normal returns."""
    name        = "IID Normal"
    short_name  = "iid_normal"
    param_names = ()

    def _avar(self, sr, **kw):
        return 1.0 + sr**2 / 2.0

    def fit(self, x):
        return {}
    
class IIDStudentTModel(AvarModel):
    """IID Student-t(ν) returns."""
    name        = "IID Student-t"
    short_name  = "iid_student_t"
    param_names = ("exc_kurt",)

    def _avar(self, sr, exc_kurt=0.0, **kw):
        #nu = np.asarray(nu, dtype=float)
        #exc_kurt = np.where(nu > 4, 6.0 / (nu - 4.0), np.nan)
        return 1.0 + sr**2 / 2.0 + sr**2 * exc_kurt / 4.0 # use nu to analyse

    def fit(self, x):
        #nu, _mu, _sigma = stats.t.fit(x, floc=float(x.mean())) #might get nu<4 and takes forever
        #return {"nu": nu}
        # if we want to analyse nu, may be  in transform the empirical kurt to nu
        return {
            "exc_kurt": float(stats.kurtosis(x, fisher=True)),
        }


class IIDNonNormalModel(AvarModel):
    """IID Non-Normal returns with general skewness and excess kurtosis."""
    name        = "IID Non-Normal"
    short_name  = "iid_nonnormal"
    param_names = ("skew", "exc_kurt")

    def _avar(self, sr, skew=0.0, exc_kurt=0.0, **kw):
        return 1.0 + sr**2 / 2.0 - sr * skew + sr**2 * exc_kurt / 4.0

    def fit(self, x):
        return {
            "skew":     float(stats.skew(x)),
            "exc_kurt": float(stats.kurtosis(x, fisher=True)),
        }


class AR1NormalModel(AvarModel):
    """AR(1) process with Normal innovations."""
    name        = "AR(1) Normal"
    short_name  = "ar1_normal"
    param_names = ("rho",)

    def _avar(self, sr, rho=0.2, **kw):
        rho = np.clip(np.asarray(rho, dtype=float), -1 + 1e-9, 1 - 1e-9)
        return (1.0 + rho) / (1.0 - rho) + sr**2 / (2.0 * (1.0 - rho**2))

    def fit(self, x):
        lag = x[:-1]; y = x[1:]
        dm  = lag - lag.mean()
        den = float(np.dot(dm, dm))
        rho = 0.0 if den < 1e-12 else float(
            np.clip(np.dot(dm, y - y.mean()) / den, -0.999, 0.999)
        )
        return {"rho": rho}



class AR1NonNormalModel(AvarModel):
    """AR(1) process with Non-Normal innovations."""
    name        = "AR(1) Non-Normal"
    short_name  = "ar1_nonnormal"
    param_names = ("rho", "skew", "exc_kurt")

    def _avar(self, sr, rho=0.2, skew=0.0, exc_kurt=0.0, **kw):
        rho    = np.clip(np.asarray(rho, dtype=float), -1 + 1e-9, 1 - 1e-9)
        base   = (1.0 + rho)  / (1.0 - rho)
        var_c  =  sr**2        / (2.0 * (1.0 - rho**2))
        skew_c = -sr * skew   / (1.0 - rho)**2
        kurt_c =  sr**2 * exc_kurt / (4.0 * (1.0 - rho**2))
        return base + var_c + skew_c + kurt_c

    def fit(self, x):
        raise NotImplementedError(
            "AR(1) Non-Normal: estimate rho, skew, exc_kurt separately."
        )



REGISTRY: dict[str, AvarModel] = {
    m.short_name: m
    for m in [
        IIDNormalModel(),
        IIDStudentTModel(),
        IIDNonNormalModel(),
        AR1NormalModel(),
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
# Raw model list — model_meta.py builds the full REGISTRY from this
# ─────────────────────────────────────────────────────────────────────────────

#: All built-in model instances in display order.
MODEL_CLASSES: list[AvarModel] = [
    IIDNormalModel(),
    IIDNonNormalModel(),
    AR1NormalModel(),
    IIDStudentTModel(),
    AR1NonNormalModel(),
]