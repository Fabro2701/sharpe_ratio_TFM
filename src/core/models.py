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
        #known    = {k: v for k, v in kwargs.items() if k in self.param_names}
        #return self._avar(sr, **{**self._defaults, **known})
        return self._avar(sr, **kwargs)

    def __call__(self, sr: float | np.ndarray, **kwargs) -> float | np.ndarray:
        return self.avar(sr, **kwargs)
    
    def _correct_bias(self, T, sr_hat, **kw):
        raise NotImplementedError()
    
    def correct_bias(self, type, T, sr_hat, **kw):
        if type == False:
            return sr_hat
        elif type == True:
            return self._correct_bias(T, sr_hat, **kw)
        else:
            raise ValueError(f"type error {type}")

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
    
    def _correct_bias(self, T, sr_hat, **kw):
        return sr_hat/(1 + 0.75/T)
        
    
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
    
    def _correct_bias(self, T, sr_hat, skew=0.0, exc_kurt=0.0, **kw):
        num = sr_hat + 0.5*skew/T
        den = 1 + 0.75*0.5/T * (exc_kurt+2)
        return num / den

class AR1NormalModel(AvarModel):
    """AR(1) process with Normal innovations."""
    name        = "AR(1) Normal"
    short_name  = "ar1_normal"
    param_names = ("rho",)

    def _avar(self, sr, rho=0.2, **kw):
        #rho = np.clip(np.asarray(rho, dtype=float), -1 + 1e-9, 1 - 1e-9)
        return (1.0 + rho) / (1.0 - rho) + sr**2 / (2.0 * (1.0 - rho**2)) *(1.0 + rho**2)

    def fit(self, x):
        lag = x[:-1]; y = x[1:]
        dm  = lag - lag.mean()
        den = float(np.dot(dm, dm))
        rho = 0.0 if den < 1e-12 else float(
            np.clip(np.dot(dm, y - y.mean()) / den, -0.999, 0.999)
        )
        return {"rho": rho}
    
    def _correct_bias(self, T, sr_hat, rho=0.2, **kw):
        return sr_hat / (1 + 3/(4*T)*(1+rho**2)/(1-rho**2))



class AR1NonNormalModel(AvarModel):
    """AR(1) process with Non-Normal innovations."""
    name        = "AR(1) Non-Normal"
    short_name  = "ar1_nonnormal"
    param_names = ("rho", "skew", "exc_kurt")

    def _avar(self, sr, rho=0.2, skew=0.0, exc_kurt=0.0, **kw):
        base   = (1.0 + rho)  / (1.0 - rho)
        aux1 = - sr * skew * (1 + rho + rho**2) / (1.0 - rho**2)
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
    
    def _correct_bias(self, T, sr_hat, rho=0.2, skew=0.0, exc_kurt=0.0, **kw):
        num = sr_hat + 0.5*skew/T * (1 + rho + rho**2) / (1.0 - rho**2)
        den = (1 + 3/(4*T)*(exc_kurt+2)*(1+rho**2)/(1-rho**2))

        return num / den

    
class GARCH11Model(AvarModel):
    """Process with GARCH(1, 1)"""
    name        = "GARCH(1, 1)"
    short_name  = "garch11"
    param_names = ("omega", "alpha", "beta", "skew", "exc_kurt")

    def _avar(self, sr, omega=0.05, alpha=0.08, beta=0.87, skew=0.0, exc_kurt=0.0, **kw):

        t1 = skew *(1-beta)/ (1 - alpha - beta) 
        t2 = (exc_kurt + 2) * (1-beta)**2 * (1+alpha+beta) / ((1 - alpha - beta) * (1 - 2*alpha*beta - beta**2) )
        
        return 1 - sr*t1+ sr**2/4 * t2
        
    def fit(self, x):
        am = arch_model(x, mean='Constant', vol='GARCH',p=1,q=1, dist='normal', rescale=True)
        res_fit = am.fit(update_freq=0,disp=False)
        return {
            "omega":res_fit.params['omega'], 
            "alpha":res_fit.params['alpha[1]'], 
            "beta":res_fit.params['beta[1]'], 
            "skew": float(stats.skew(x)), 
            "exc_kurt":float(stats.kurtosis(x, fisher=True))
        }
    
class AR1GARCH11NormalModel(AvarModel):
    """AR(1)-GARCH(1, 1) process with Normal innovations."""
    name        = "AR(1) GARCH(1, 1) Normal"
    short_name  = "ar1_garch11normal"
    param_names = ("rho", "omega", "alpha", "beta", "skew", "exc_kurt")

    def _avar(self, sr, rho=0.2, omega=0.05, alpha=0.08, beta=0.87, **kw):

        Q_num = 1 - (alpha + beta)**2
        Q_den = 1 - (alpha + beta)**2 - 2 * (alpha**2)
        Q = Q_num / Q_den

        term1 = (1 + rho) / (1 - rho)

        left = (rho**2) / (1 - (rho**2) * (alpha + beta))
        right = ((1 - alpha - beta) / (1 - rho**2)) + (3 * alpha + beta) * Q
        fact1 = left * right

        fact2 = 0.5 * (((1 - beta) / (1 - (alpha + beta)))**2) * Q

        res = term1 + (sr**2) * (fact1 + fact2)

        return res

    def fit(self, x):
        am = arch_model(x, mean='Constant', vol='GARCH',p=1,q=1, dist='normal', rescale=True)
        res_fit = am.fit(update_freq=0,disp=False)

        return {
            "rho":res_fit.params['y[1]'], 
            "omega":res_fit.params['omega'], 
            "alpha":res_fit.params['alpha[1]'], 
            "beta":res_fit.params['beta[1]'], 
        }  
    
    def _correct_bias(self, T, sr_hat, rho=0.2, skew=0.0, exc_kurt=0.0, **kw):
        num = sr_hat + 0.5*skew/T * (1 + rho + rho**2) / (1.0 - rho**2)
        den = (1 + 3/(4*T)*(exc_kurt+2)*(1+rho**2)/(1-rho**2))
        #pending
        return num / den
    

class AR1GARCH11SymmModel(AvarModel):
    """AR(1)-GARCH(1, 1) process with symmetric innovations."""
    name        = "AR(1) GARCH(1, 1) symmetric"
    short_name  = "ar1_garch11symm"
    param_names = ("rho", "omega", "alpha", "beta", "skew", "exc_kurt")

    def _avar(self, sr, rho=0.2, omega=0.05, alpha=0.08, beta=0.87, exc_kurt=0, **kw):

        D1 = 1 - 2 * alpha * beta - beta**2
        D3 = 1 - rho**2 * (alpha + beta)
        N1 = 1 - alpha * beta - beta**2

        # A expandido
        A = (6 * rho**2 * alpha * N1) / (D3 * D1)

        # Términos principales
        phi=rho
        term1 = (1 + phi) / (1 - phi)
        k_r = exc_kurt + 3

        var_a = 4 * phi**2 * (1-phi**2)**2 / (1-phi**2) * ((1/(1+A)*((1+phi**2)*k_r - 5*phi**2 - 1))*alpha*(N1)/(D1)/(D3) + 1 )
        var_eta = (1-phi**2)**2 * (1/(1+A)) / (1-phi**2) * ( (1+phi**2)*k_r - 5*phi**2 - 1 ) * (1-(alpha+beta)**2)/(D1) 

        S_22 = 1/(1-phi**2)**2 * (var_a + ((1-beta)/(1-alpha-beta))**2 * var_eta) 
        term2 = 0.25 * sr**2 * S_22

        return term1 + term2

    def fit(self, x):
        am = arch_model(x, mean='Constant', vol='GARCH',p=1,q=1, dist='normal', rescale=True)
        res_fit = am.fit(update_freq=0,disp=False)

        return {
            "rho":res_fit.params['y[1]'], 
            "omega":res_fit.params['omega'], 
            "alpha":res_fit.params['alpha[1]'], 
            "beta":res_fit.params['beta[1]'], 
        }  
    
    def _correct_bias(self, T, sr_hat, rho=0.2, skew=0.0, exc_kurt=0.0, **kw):
        num = sr_hat + 0.5*skew/T * (1 + rho + rho**2) / (1.0 - rho**2)
        den = (1 + 3/(4*T)*(exc_kurt+2)*(1+rho**2)/(1-rho**2))
        #pending

        return num / den
    
class HACModel(AvarModel):
    """non-parametric Newey-West HAC estimator"""
    name        = "HAC"
    short_name  = "hac"
    param_names = ('x')

    def _avar(self, sr, x, **kw):
        T = len(x)
        
        # 1. Calculate sample moments
        mu = np.mean(x)
        var = np.var(x)
        sigma = np.sqrt(var)
        
        # 2. Define the moment conditions (Z_t)
        # We demean them so E[Z_t] = 0. 
        # Col 1: Mean moment. Col 2: Variance moment.
        Z = np.column_stack((x - mu, (x - mu)**2 - var))
        
        # Rule of thumb for Newey-West lags if not specified
        
        lags = int(np.ceil(4 * (T / 100)**(2/9)))
            
        # 3. Compute the Newey-West HAC Covariance Matrix (S matrix)
        # Start with Lag 0 (White's robust heteroskedasticity covariance)
        S = (Z.T @ Z) / T
        
        # Add Bartlett kernel weighted cross-covariances for serial correlation
        for j in range(1, lags + 1):
            weight = 1.0 - (j / (lags + 1.0))
            # Cross-covariance at lag j
            Gamma_j = (Z[j:].T @ Z[:-j]) / T
            S += weight * (Gamma_j + Gamma_j.T)
            
        # 4. Apply the Delta Method
        # The Sharpe Ratio is a function: f(mu, var) = mu * (var)^(-1/2)
        # We need the gradient of this function with respect to [mu, var]
        grad = np.array([
            1.0 / sigma,                 # d(SR)/d(mu)
            -mu / (2.0 * var * sigma)    # d(SR)/d(var)
        ])
        
        # 5. Calculate final Asymptotic Variance
        avar = grad.T @ S @ grad
        return avar
        

    def fit(self, x):
        return {'x':x}
    


REGISTRY: dict[str, AvarModel] = {
    m.short_name: m
    for m in [
        IIDNormalModel(),
        IIDStudentTModel(),
        IIDNonNormalModel(),
        AR1NormalModel(),
        AR1NonNormalModel(),
        GARCH11Model(),
        AR1GARCH11NormalModel(),
        AR1GARCH11SymmModel(),
        HACModel(),
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
        AR1GARCH11NormalModel(),
]