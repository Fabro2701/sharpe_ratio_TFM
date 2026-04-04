"""
dgp.py  —  Data Generating Process class hierarchy.

Each DGP exposes a single method:

    simulate(n, rng) -> np.ndarray   shape (n,)

All randomness flows through the caller-supplied numpy Generator so that
the top-level seed is the single source of reproducibility.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.special import gamma as gamma_func
from statsmodels.tsa.arima_process import ArmaProcess
# --- arch imports ---
from arch import arch_model
from arch.univariate import ConstantMean, GARCH
from arch.univariate.distribution import (
    Normal,
    StudentsT,
    SkewStudent,
)


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────

class DGP(abc.ABC):
    """Abstract base for all data generating processes."""

    #: Human-readable class-level label (overridden in subclasses)
    label: str = ""

    @abc.abstractmethod
    def simulate(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return a 1-D array of length *n*."""

    @abc.abstractmethod
    def calibrate_params(self, mu: float, sigma: float):
        """Mutate internal parameters so that E[X] = mu, Std[X] = sigma."""

    def calculate_theo_moments(self):
        raise NotImplementedError()

    def get_theo_moments(self):
        return {"skew":self.th_skew,
                "exc_kurt":self.th_exc_kurt,
                "rho":self.th_rho,
                "nu":self.th_nu,
                "mean":self.th_mean,
                "sigma":self.th_sigma}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._repr_params()})"

    def _repr_params(self) -> str:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Innovation distributions  (reusable building blocks)
# ─────────────────────────────────────────────────────────────────────────────

class InnovDist(abc.ABC):
    """Callable innovation sampler:  (size, rng) -> np.ndarray."""

    @abc.abstractmethod
    def __call__(self, size: int, rng: np.random.Generator) -> np.ndarray: ...

    @abc.abstractmethod
    def calibrate_params(self, mu: float, sigma: float) -> "InnovDist":
        """Mutate internal parameters so that E[X] = mu, Std[X] = sigma."""

    def calculate_theo_moments(self):
        raise NotImplementedError()

    def get_theo_moments(self):
        return {"skew":self.th_skew,
                "exc_kurt":self.th_exc_kurt,
                "rho":self.th_rho,
                "nu":self.th_nu,
                "mean":self.th_mean,
                "sigma":self.th_sigma}

    def __repr__(self) -> str:
        return self.__class__.__name__


class NormalInnov(InnovDist):
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std  = std
        self.calculate_theo_moments()
        
    def __call__(self, size, rng):
        return rng.normal(self.mean, self.std, size=size)
    
    def calibrate_params(self, mu: float, sigma: float) -> "NormalInnov":
        self.mean = mu
        self.std  = sigma
        self.calculate_theo_moments()
        return self
    
    def calculate_theo_moments(self):
        self.th_skew = 0
        self.th_exc_kurt = 0
        self.th_rho = 0
        self.th_nu = np.inf
        self.th_mean = self.mean
        self.th_sigma = self.std

    def __repr__(self):
        return f"Normal(μ={self.mean}, σ={self.std})"


class StudentTInnov(InnovDist):
    """
    Innovation: mean + scale * t_df

    df   — shape (tail heaviness), fixed by the user; must be > 2 for calibration
    mean — location, calibrated to mu
    scale — calibrated so Std[X] = sigma  (any positive sigma is valid)

    Default (uncalibrated): mean=0, scale = sqrt((df-2)/df)  → unit variance
    """
    def __init__(self, df: float = 5.0, mean: float = 0.0):
        self.df   = df
        self.mean = mean
        self.scale = np.sqrt((df - 2) / df)
        self.calculate_theo_moments()
        

    def __call__(self, size, rng):
        return self.mean + rng.standard_t(self.df, size=size) * self.scale

    def calibrate_params(self, mu: float, sigma: float) -> "StudentTInnov":
        self.mean  = mu
        self.scale = sigma * np.sqrt((self.df - 2) / self.df)  # any sigma > 0 works
        self.calculate_theo_moments()
        return self
    
    def calculate_theo_moments(self):
        self.th_skew = 0
        #if self.df < 4:
        #    print("Warning df > 4")
        self.th_exc_kurt = 6/(self.df - 4) if self.df > 4 else np.inf
        self.th_rho = 0
        self.th_nu = self.df
        self.th_mean = self.mean
        self.th_sigma = self.scale * np.sqrt(self.df/(self.df - 2)) if self.df > 2 else np.inf

    def __repr__(self):
        return f"StudentT(μ={self.mean}, df={self.df}, scale={self.scale:.4f})"


class SkewTInnov(InnovDist):
    """
    Hansen (1994) skewed-t innovation.

    Parameterisation matches arch's SkewStudent:
        df  (nu)  — degrees of freedom, must be > 2
        eta       — skewness parameter in (-1, 1); 0 → symmetric t

    The raw draw has E[raw] = 0, Var[raw] = 1 by construction of the
    Hansen constants (a, b, c).  Location/scale are then applied:

        X = mean + scale * raw
    """

    def __init__(self, df: float = 5.0, eta: float = 0.0, mean: float = 0.0):
        self.df   = df
        self.eta  = eta
        self.mean = mean
        self._compute_constants()
        self.scale = 1.0
        self.calculate_theo_moments()

    def _compute_constants(self):
        """Pre-compute Hansen's a, b, c constants from df and eta."""
        nu, eta = self.df, self.eta
        self._c = gamma_func((nu + 1) / 2) / (
            np.sqrt(np.pi * (nu - 2)) * gamma_func(nu / 2)
        )
        self._a = 4.0 * eta * self._c * (nu - 2) / (nu - 1)
        self._b = np.sqrt(1.0 + 3.0 * eta**2 - self._a**2)

    def _raw_sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return SkewStudent(seed=rng).simulate([self.df, self.eta])(size)

    def __call__(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return self.mean + self.scale * self._raw_sample(size, rng)

    def calibrate_params(self, mu: float, sigma: float) -> "SkewTInnov":
        self.mean  = mu
        self.scale = sigma
        self.calculate_theo_moments()
        return self

    def calculate_theo_moments(self):
        nu, eta = self.df, self.eta
        
        # Unstandardized moments of Y
        m1 = self._a
        m2 = 1.0 + 3.0 * eta**2
        
        # --- Skewness (requires nu > 3) ---
        if nu > 3:
            m3 = (16.0 * self._c * eta * (1.0 + eta**2) * (nu - 2)**2) / ((nu - 1) * (nu - 3))
            self.th_skew = (m3 - 3*self._a*m2 + 2*self._a**3) / (self._b**3)
        else:
            print(f"No skewness for skewt df={nu} (needs df > 3)")
            self.th_skew = np.nan

        # --- Excess Kurtosis (requires nu > 4) ---
        if nu > 4:
            m4 = 3.0 * ((nu - 2) / (nu - 4)) * (1.0 + 10.0*eta**2 + 5.0*eta**4)
            # Full kurtosis for standard variable Z = (Y-a)/b
            kurtosis = (m4 - 4*self._a*m3 + 6*self._a**2*m2 - 3*self._a**4) / (self._b**4)
            self.th_exc_kurt = kurtosis - 3.0 
        else:
            print(f"No kurtosis for skewt df={nu} (needs df > 4)")
            self.th_exc_kurt = np.nan

        self.th_rho   = 0.0
        self.th_nu    = nu
        self.th_mean  = self.mean
        # The scale of the standard variable is 1, so variance of X is scale^2
        self.th_sigma = self.scale

    def __repr__(self):
        return (
            f"SkewT(μ={self.mean}, df={self.df}, η={self.eta}, "
            f"scale={self.scale:.4f})"
        )
    
from scipy.special import gamma
class APDInnov(InnovDist):
    """
    """

    def __init__(self, alpha: float = 0.7, lam: float = 1.35, mean: float = 0.0):
        self.alpha   = alpha
        self.lam  = lam
        self.mean = mean
        self._compute_constants()
        self.scale = 1.0
        self.calculate_theo_moments()

    def _compute_constants(self):
        alpha_lam = self.alpha ** self.lam
        one_minus_alpha_lam = (1.0 - self.alpha) ** self.lam
        self.delta = (2.0 * alpha_lam * one_minus_alpha_lam) / (alpha_lam + one_minus_alpha_lam)

    def _raw_sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        alpha, lam, delta = self.alpha, self.lam, self.delta
        w = rng.gamma(shape=1.0/lam, scale=1.0, size=size)
        u = rng.uniform(0, 1, size=size)
        x = np.where(
            u <= alpha,
            -alpha * (w / delta) ** (1.0 / lam),
            (1.0 - alpha) * (w / delta) ** (1.0 / lam)
        )
        return (x - x.mean())/x.std()

    def __call__(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return self.mean + self.scale * self._raw_sample(size, rng)
    
    def calibrate_params(self, mu: float, sigma: float) -> "SkewTInnov":
        self.mean  = mu
        self.scale = sigma
        self.calculate_theo_moments()
        return self
    
    def calculate_theo_moments(self):
        alpha, lam, delta = self.alpha, self.lam, self.delta

        def raw_moment(k):
            M_k = (1 - alpha)**(k + 1) + ((-1)**k) * (alpha**(k + 1))
            G_k = gamma((k + 1) / lam) / ((delta**(k / lam)) * gamma(1 / lam))
            return G_k * M_k
        
        m1 = raw_moment(1)
        m2 = raw_moment(2)
        m3 = raw_moment(3)
        m4 = raw_moment(4)
        
        var = m2 - m1**2
        sd = np.sqrt(var)
        
        mu3 = m3 - 3*m1*m2 + 2*(m1**3)
        mu4 = m4 - 4*m1*m3 + 6*(m1**2)*m2 - 3*(m1**4)
        
        self.th_skew = mu3 / (sd**3)
        self.th_exc_kurt = mu4 / (sd**4) - 3

        self.th_rho   = 0.0
        self.th_nu    = 0.0
        self.th_mean  = self.mean
        self.th_sigma = self.scale
    
    def __repr__(self):
        return (
            f"APD(μ={self.mean}, alpha={self.alpha}, lam={self.lam}, "
            f"scale={self.scale:.4f})"
        )
    


class UniformInnov(InnovDist):
    def __init__(self, low: float = -1.0, high: float = 1.0):
        self.low  = low
        self.high = high

    def __call__(self, size, rng):
        return rng.uniform(self.low, self.high, size=size)
    
    
    def calibrate_params(self, mu: float, sigma: float):
        raise NotImplementedError(self.__class__)

    def __repr__(self):
        return f"Uniform({self.low}, {self.high})"


class ChiSquareInnov(InnovDist):
    """Centered chi-square innovations."""
    def __init__(self, df: float = 2.0):
        self.df = df

    def __call__(self, size, rng):
        return rng.chisquare(self.df, size=size) - self.df
    
    
    def calibrate_params(self, mu: float, sigma: float):
        raise NotImplementedError(self.__class__)

    def __repr__(self):
        return f"ChiSq(df={self.df})"


class GaussianMixtureInnov(InnovDist):
    def __init__(
        self,
        means:   Sequence[float] = (-2.0, 2.0),
        stds:    Sequence[float] = (1.0, 1.0),
        weights: Sequence[float] | None = None,
    ):
        self.means   = np.asarray(means,   dtype=float)
        self.stds    = np.asarray(stds,    dtype=float)
        w = np.ones(len(means)) if weights is None else np.asarray(weights, dtype=float)
        self.weights = w / w.sum()

    def __call__(self, size, rng):
        k          = rng.choice(len(self.means), size=size, p=self.weights)
        return np.array([rng.normal(self.means[i], self.stds[i]) for i in k])
    
    
    def calibrate_params(self, mu: float, sigma: float):
        raise NotImplementedError(self.__class__)

    def __repr__(self):
        return f"GaussianMixture(means={self.means.tolist()}, stds={self.stds.tolist()})"


# ─────────────────────────────────────────────────────────────────────────────
# IID processes
# ─────────────────────────────────────────────────────────────────────────────

class IIDProcess(DGP):
    """
    IID draws from an arbitrary innovation distribution.

    Parameters
    ----------
    innov : InnovDist or callable(size, rng) -> np.ndarray
    """

    label = "IID"

    def __init__(self, innov: InnovDist | Callable = NormalInnov()):
        self.innov = innov
        self.calculate_theo_moments()
        

    def simulate(self, n, rng):
        return self.innov(n, rng)

    def calibrate_params(self, mu: float, sigma: float) -> "IIDProcess":
        self.innov.calibrate_params(mu, sigma)
        self.calculate_theo_moments()
        return self
    
    def calculate_theo_moments(self):
        self.th_skew = self.innov.th_skew
        self.th_exc_kurt = self.innov.th_exc_kurt
        self.th_rho = 0
        self.th_nu = self.innov.th_nu
        self.th_mean = self.innov.th_mean
        self.th_sigma = self.innov.th_sigma

    def _repr_params(self):
        return repr(self.innov)


# ─────────────────────────────────────────────────────────────────────────────
# AR(p) process
# ─────────────────────────────────────────────────────────────────────────────

def _make_stationary_ar(order: int, rng: np.random.Generator) -> np.ndarray:
    """Sample AR coefficients whose roots lie strictly outside the unit circle."""
    while True:
        phi = rng.uniform(-0.9, 0.9, size=order)
        poly = np.r_[-phi[::-1], 1.0]
        if np.all(np.abs(np.roots(poly)) > 1):
            return phi


class ARProcess(DGP):
    """
    Stationary AR(1) process with pluggable innovation distribution.

    Parameters
    ----------
    phi : 
        AR coefficients φ₁.  If None, the
        coefficients are drawn randomly at each ``simulate`` call.
    innov : InnovDist or callable
    drift : float
        Constant drift added to every observation.
    """

    label = "AR1"

    def __init__(
        self,
        phi:   float | None = None,
        innov: InnovDist | Callable = NormalInnov(),
        drift: float = 0.0,
    ):
        self.phi   = phi 
        self.innov = innov
        self.drift = drift
        self.calculate_theo_moments()

    def simulate(self, n, rng):
        phi = (
            _make_stationary_ar(1, rng)[0]
            if self.phi is None
            else self.phi
        )
        ar   = np.r_[1.0, -phi]
        ma   = np.array([1.0])
        proc = ArmaProcess(ar, ma)
        noise = self.innov(n, rng)

        def _distrvs(size):
            # statsmodels passes size as a tuple like (n+burnin,)
            k = size[0] if isinstance(size, tuple) else int(size)
            return noise[:k] if k <= len(noise) else self.innov(k, rng)

        return proc.generate_sample(nsample=n, distrvs=_distrvs) + self.drift
    
    def calibrate_params(self, mu: float, sigma: float) -> "ARProcess":
        
        sigma_innov = sigma * np.sqrt(1.0 - self.phi**2)
        self.drift = mu
        self.innov.calibrate_params(0.0, sigma_innov)
        self.calculate_theo_moments()
        return self
    
    def calculate_theo_moments(self):
        self.th_skew = self.innov.th_skew #not sure
        phi_sum_sq = float(self.phi**2)
        self.th_exc_kurt = self.innov.th_exc_kurt * (1-phi_sum_sq) / (1+phi_sum_sq)
        self.th_rho = self.phi
        self.th_nu = self.innov.th_nu # x wouldnt be t if ep is t
        self.th_mean = self.drift
        self.th_sigma = self.innov.th_sigma / np.sqrt((1 - phi_sum_sq))

    def _repr_params(self):
        phi_str = "random" if self.phi is None else self.phi
        return f"phi={phi_str}, innov={self.innov!r}, drift={self.drift}"


# ─────────────────────────────────────────────────────────────────────────────
# GARCH family  (via arch)
# ─────────────────────────────────────────────────────────────────────────────

# Default extra distribution parameters for each supported distribution type.
# These are appended to [mu, omega, alpha, beta] when calling arch's simulate().
#
#   normal  : no extra params
#   t       : [df]            — degrees of freedom, must be > 2
#   skewt   : [df, lambda]    — df > 2, lambda (skewness) in (-1, 1)
#
_GARCH_DIST_DEFAULTS: dict[str, list[float]] = {
    "normal": [],
    "t":      [8.0],
    "skewt":  [8.0, 0.0],
}


class GARCHProcess(DGP):
    """
    GARCH(1,1) with pluggable innovation distribution.
 
    Parameters
    ----------
    mu, omega, alpha, beta : GARCH mean / vol parameters.
    dist : one of ``"normal"``, ``"t"``, ``"skewt"``.
    dist_params : extra parameters for the chosen distribution.
        • normal  → []
        • t       → [df]          (default: [8.0])
        • skewt   → [df, lambda]  (default: [8.0, 0.0])
      If *None*, sensible defaults are used (see above).
    """
 
    label = "GARCH"
 
    def __init__(
        self,
        mu:    float = 0.00,
        omega: float = 0.05,
        alpha: float | list = 0.10,
        beta:  float | list = 0.85,
        dist:  str = "normal",
        dist_params: list[float] | None = None,
    ):
        if dist not in _GARCH_DIST_DEFAULTS:
            raise ValueError(
                f"dist must be one of {list(_GARCH_DIST_DEFAULTS)}; got {dist!r}"
            )
        self.mu    = mu
        self.omega = omega
        self.alpha = alpha
        self.beta  = beta
        self.dist  = dist
        self.dist_params = (
            list(dist_params)
            if dist_params is not None
            else list(_GARCH_DIST_DEFAULTS[dist])
        )
        self.calculate_theo_moments()
 
    # ------------------------------------------------------------------
    # Properties for convenience access to dist_params by name
    # ------------------------------------------------------------------
 
    @property
    def df(self) -> float | None:
        """Degrees of freedom (t and skewt only)."""
        return self.dist_params[0] if len(self.dist_params) >= 1 else None
 
    @property
    def lam(self) -> float | None:
        """Skewness lambda (skewt only)."""
        return self.dist_params[1] if len(self.dist_params) >= 2 else None
 
    # ------------------------------------------------------------------
    # DGP interface
    # ------------------------------------------------------------------
 
    def simulate(self, n: int, rng: np.random.Generator) -> np.ndarray:
        m = ConstantMean()
        m.volatility = GARCH(p=1, q=1)
 
        if self.dist == "normal":
            m.distribution = Normal(seed=rng)
        elif self.dist == "t":
            m.distribution = StudentsT(seed=rng)
        elif self.dist == "skewt":
            m.distribution = SkewStudent(seed=rng)
 
        # Full parameter vector: [mu] + [omega, alpha, beta] + dist_params
        params = [self.mu, self.omega, self.alpha, self.beta] + self.dist_params
 
        result = m.simulate(params, nobs=n, burn=100)
        return result["data"].values
 
    def calibrate_params(self, mu: float, sigma: float) -> "GARCHProcess":
        self.mu    = mu
        self.omega = sigma**2 * (1 - self.alpha - self.beta)
        self.calculate_theo_moments()
        return self
 
    # ------------------------------------------------------------------
    # Innovation-distribution moment helpers
    # ------------------------------------------------------------------
 
    def _innov_exc_kurt(self) -> float:
        """Excess kurtosis κ_ε of the standardised innovation."""
        if self.dist == "normal":
            return 0.0
        df = self.dist_params[0]
        return 6.0 / (df - 4.0) if df > 4.0 else np.inf
 
    def _innov_skew(self) -> float:
        """Skewness of the standardised innovation."""
        if self.dist in ("normal", "t"):
            return 0.0
        # Hansen (1994) skewed-t — same algebra as SkewTInnov.calculate_theo_moments
        df, lam = self.dist_params[0], self.dist_params[1]
        if df <= 3.0:
            return np.nan
        from scipy.special import gamma as gf
        c   = gf((df + 1) / 2) / (np.sqrt(np.pi * (df - 2)) * gf(df / 2))
        a   = 4.0 * lam * c * (df - 2) / (df - 1)
        b   = np.sqrt(1.0 + 3.0 * lam**2 - a**2)
        # E[|T|^3] for T ~ t(df), valid for df > 3
        # Using E[|T|^k] = df^(k/2) Γ((k+1)/2) Γ((df-k)/2) / (√π Γ(df/2))
        # with k=3 and Γ(2) = 1:
        e_abs_t3 = df ** (3 / 2) * gf((df - 3) / 2) / (np.sqrt(np.pi) * gf(df / 2))
        p_l  = (1.0 + lam) / 2.0
        e_z3 = (
            p_l  * (-(1.0 - lam) ** 3) * e_abs_t3
            + (1.0 - p_l) * (1.0 + lam) ** 3 * e_abs_t3
        )
        # raw = (z - a) / b  =>  E[raw^3] via binomial expansion
        e_z  = a
        e_z2 = b**2 + a**2   # Var(raw)=1  =>  Var(z)=b^2
        e_raw3 = (e_z3 - 3.0 * a * e_z2 + 3.0 * a**2 * e_z - a**3) / b**3
        return float(e_raw3)
 
    # ------------------------------------------------------------------
 
    def calculate_theo_moments(self):
        alpha, beta = self.alpha, self.beta
        p = alpha + beta                          # GARCH persistence
 
        kappa_e = self._innov_exc_kurt()          # innovation excess kurtosis
        skew_e  = self._innov_skew()              # innovation skewness
 
        self.th_mean  = self.mu
        self.th_sigma = np.sqrt(self.omega / (1.0 - p))
        self.th_rho   = 0.0
        self.th_nu    = self.df if self.dist in ("t", "skewt") else np.inf
 
        # Denominator shared by both kurtosis and Var(σ²)/σ̄⁴.
        # The 4th moment of y_t (and hence Var(σ²)) exists iff denom > 0.
        denom = 1.0 - p**2 - alpha**2 * (kappa_e + 2.0)
 
        if np.isfinite(kappa_e) and denom > 0.0:
            # --- excess kurtosis (exact closed-form for GARCH(1,1)) ---
            # κ_y = [κ_ε(1-p²+3α²) + 6α²] / denom
            self.th_exc_kurt = (
                kappa_e * (1.0 - p**2 + 3.0 * alpha**2) + 6.0 * alpha**2
            ) / denom
 
            # --- skewness: Taylor expansion of E[σ_t^3] up to the 3/8 term ---
            # E[(σ²)^(3/2)] ≈ σ̄³ · (1 + 3/8 · Var(σ²)/σ̄⁴)
            # Var(σ²)/σ̄⁴ = α²(κ_ε+2) / denom
            var_sig2_norm = alpha**2 * (kappa_e + 2.0) / denom
            self.th_skew = skew_e * (1.0 + (3.0 / 8.0) * var_sig2_norm)
        else:
            self.th_exc_kurt = np.inf
            self.th_skew     = np.nan
 
    def _repr_params(self):
        dp = f", dist_params={self.dist_params}" if self.dist_params else ""
        return (
            f"omega={self.omega}, alpha={self.alpha}, beta={self.beta}, "
            f"dist={self.dist!r}{dp}"
        )
 
 

# ─────────────────────────────────────────────────────────────────────────────
# Outlier injection wrapper
# ─────────────────────────────────────────────────────────────────────────────

class WithOutliers(DGP):
    """ aun no uso este, asi que no te preocupes
    Wraps any DGP and injects additive outliers.

    Parameters
    ----------
    base : DGP
    fraction : float
        Proportion of observations that become outliers.
    scale : float
        Outlier magnitude multiplier (applied to std of the series).
    """

    label = "WithOutliers"

    def __init__(self, base: DGP, fraction: float = 0.05, scale: float = 5.0):
        self.base     = base
        self.fraction = fraction
        self.scale    = scale

    def simulate(self, n, rng):
        data = self.base.simulate(n, rng).copy()
        std  = data.std()
        k    = max(1, int(n * self.fraction))
        idx  = rng.choice(n, size=k, replace=False)
        data[idx] += rng.normal(0, self.scale * std, size=k)
        return data
    
    
    def calibrate_params(self, mu: float, sigma: float):
        raise NotImplementedError(self.__class__)

    def _repr_params(self):
        return f"{self.base!r}, fraction={self.fraction}, scale={self.scale}"
    

DGP_EXAMPLES: dict[str, callable] = {
    "iid_normal": (
        lambda: IIDProcess(NormalInnov()).calibrate_params(mu=0.5, sigma=2.0)
    ),
    "iid_t3": (
        lambda: IIDProcess(StudentTInnov(df=3)).calibrate_params(mu=1.5, sigma=1.2)
    ),
    "iid_t6": (
        lambda: IIDProcess(StudentTInnov(df=6)).calibrate_params(mu=1.5, sigma=1.2)
    ),
    "iid_skewt60_m05": (
        lambda: IIDProcess(SkewTInnov(df=60, eta=-0.5)).calibrate_params(mu=1.5, sigma=1.2)
    ),
    "iid_skewt6_m05": (
        lambda: IIDProcess(SkewTInnov(df=6, eta=-0.5)).calibrate_params(mu=1.5, sigma=1.2)
    ),
    "iid_apd": (
        lambda: IIDProcess(APDInnov(alpha=0.7, lam=1.35)).calibrate_params(mu=1.5, sigma=1.2)
    ),
    "ar1_06_normal": (
        lambda: ARProcess(phi=0.6, innov=NormalInnov()).calibrate_params(mu=1.5, sigma=0.4)
    ),
    "ar1_m06_normal": (
        lambda: ARProcess(phi=-0.6, innov=NormalInnov()).calibrate_params(mu=1.5, sigma=0.4)
    ),
    "ar1_06_t6": (
        lambda: ARProcess(phi=0.6, innov=StudentTInnov(df=6)).calibrate_params(mu=1.5, sigma=0.4)
    ),
    "ar1_m06_t6": (
        lambda: ARProcess(phi=-0.6, innov=StudentTInnov(df=6)).calibrate_params(mu=1.5, sigma=0.4)
    ),
    "garch_normal": (
        lambda: GARCHProcess(mu=0.05, omega=0.05, alpha=0.10, beta=0.85,
                             dist="normal").calibrate_params(mu=1.5, sigma=0.4)
    ),
}