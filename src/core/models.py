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

from arch.univariate.distribution import StudentsT
import numpy as np
from scipy import stats
from arch import arch_model


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
    
    def correct_bias(self, type, T, sr_hat, **kwargs):
        if type==False:
            return sr_hat
        raise NotImplementedError()

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
    
    def correct_bias(self, type, T, sr_hat, **kw):
        if type == False:
            return sr_hat
        elif type == "sigma":#expansion at sigma
            return sr_hat / (1 + 0.5 / T)
        else:
            raise ValueError(f"type error {type}")
    
class _CustomStudentsT(StudentsT):
        def constraints(self):
            # Mantenemos la matriz A igual, pero cambiamos el vector b
            # A * theta >= b  --->  nu >= 4.01
            return np.array([[1], [-1]]), np.array([4.01, -500.0])

        def bounds(self, resids):
            # Actualizamos la tupla de límites para el optimizador (L-BFGS-B / SLSQP)
            return [(4.05, 500.0)]
        
class IIDStudentTModel(AvarModel):
    """IID Student-t(ν) returns."""
    name        = "IID Student-t"
    short_name  = "iid_student_t"


    param_names = ("exc_kurt",)
    def _avar(self, sr, exc_kurt=0.0, **kw):
        return 1.0 + sr**2 / 2.0 + sr**2 * exc_kurt / 4.0 
    def fit(self, x):
        return {
            "exc_kurt": float(stats.kurtosis(x, fisher=True)),
        }
    
    # param_names = ("nu",)
    # def _avar(self, sr, nu=8.0, **kw):
    #     exc_kurt = 6.0 / (nu - 4.0)
    #     return 1.0 + sr**2 / 2.0 + sr**2 * exc_kurt / 4.0

    # def fit(self, x):
    #     am = arch_model(x, mean='Constant', vol='constant', dist='t', rescale=False)
    #     am.distribution = _CustomStudentsT()
    #     res_fit = am.fit(update_freq=0, disp=False)
    #     return {
    #         "nu": res_fit.params['nu'],
    #     }


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
    
    def correct_bias(self, type, T, sr_hat, exc_kurt=0.0, **kw):
        if type == False:
            return sr_hat
        elif type == "sigma":#expansion at sigma
            return sr_hat / (1 + 0.25*(exc_kurt+3 - 1) / T)
        else:
            raise ValueError(f"type error {type}")



class AR1NormalModel(AvarModel):
    """AR(1) process with Normal innovations."""
    name        = "AR(1) Normal"
    short_name  = "ar1_normal"
    param_names = ("rho",)

    def _avar(self, sr, rho=0.2, **kw):
        rho = np.clip(np.asarray(rho, dtype=float), -1 + 1e-9, 1 - 1e-9)
        return (1.0 + rho) / (1.0 - rho) + sr**2 / (2.0 * (1.0 - rho**2)) *(1.0 + rho**2)

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
        base   = (1.0 + rho)  / (1.0 - rho)
        aux1 = - sr * skew * (1 + rho+ rho**2) / (1.0 - rho**2)
        aux2 =  sr**2 * (exc_kurt + 2)/4 * (1 + rho**2) / (1.0 - rho**2)
        return base + aux1 + aux2

    def fit(self, x):
        lag = x[:-1]; y = x[1:]
        dm  = lag - lag.mean()
        den = float(np.dot(dm, dm))
        rho = 0.0 if den < 1e-12 else float(
            np.clip(np.dot(dm, y - y.mean()) / den, -0.999, 0.999)
        )
        return {
            "skew":     float(stats.skew(x)),
            "exc_kurt": float(stats.kurtosis(x, fisher=True)),
            "rho": rho
        }
    
class GARCH11Model(AvarModel):
    """Process with GARCH(1, 1) innovations."""
    name        = "GARCH(1, 1)"
    short_name  = "garch11"
    param_names = ("omega", "alpha", "beta", "skew", "exc_kurt")

    def _avar(self, sr, omega=0.05, alpha=0.08, beta=0.87, skew=0.0, exc_kurt=0.0, **kw):
        t1 = skew / (1 - alpha - beta) 
        t2 = (exc_kurt + 2) * (1-beta)**2 * (1+alpha+beta) / ((1 - alpha - beta) * (1 - 2*alpha*beta - beta**2) )
        
        return 1 - sr*t1+ sr**2/4 * t2
        

    def fit(self, x):
        #rescale changes omega but it doesnt apper in avar
        am = arch_model(x, mean='Constant', vol='GARCH',p=1,q=1, dist='normal', rescale=True)
        res_fit = am.fit(update_freq=0,disp=False)
        return {
            "omega":res_fit.params['omega'], 
            "alpha":res_fit.params['alpha[1]'], 
            "beta":res_fit.params['beta[1]'], 
            "skew": float(stats.skew(x)), 
            "exc_kurt":float(stats.kurtosis(x, fisher=True))
        }



REGISTRY: dict[str, AvarModel] = {
    m.short_name: m
    for m in [
        IIDNormalModel(),
        IIDStudentTModel(),
        IIDNonNormalModel(),
        AR1NormalModel(),
        AR1NonNormalModel(),
        GARCH11Model(),
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
        IIDStudentTModel(),
        IIDNonNormalModel(),
        AR1NormalModel(),
        AR1NonNormalModel(),
        GARCH11Model(),
]