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
from arch import arch_model
from statsmodels.tsa.arima_process import ArmaProcess


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

    def __repr__(self) -> str:
        return self.__class__.__name__


class NormalInnov(InnovDist):
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std  = std

    def __call__(self, size, rng):
        return rng.normal(self.mean, self.std, size=size)
    
    def calibrate_params(self, mu: float, sigma: float) -> "NormalInnov":
        self.mean = mu
        self.std  = sigma
        return self

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

    def __call__(self, size, rng):
        return self.mean + rng.standard_t(self.df, size=size) * self.scale

    def calibrate_params(self, mu: float, sigma: float) -> "StudentTInnov":
        self.mean  = mu
        self.scale = sigma * np.sqrt((self.df - 2) / self.df)  # any sigma > 0 works
        return self

    def __repr__(self):
        return f"StudentT(μ={self.mean}, df={self.df}, scale={self.scale:.4f})"


class UniformInnov(InnovDist):
    def __init__(self, low: float = -1.0, high: float = 1.0):
        self.low  = low
        self.high = high

    def __call__(self, size, rng):
        return rng.uniform(self.low, self.high, size=size)

    def __repr__(self):
        return f"Uniform({self.low}, {self.high})"


class ChiSquareInnov(InnovDist):
    """Centered chi-square innovations."""
    def __init__(self, df: float = 2.0):
        self.df = df

    def __call__(self, size, rng):
        return rng.chisquare(self.df, size=size) - self.df

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

    def simulate(self, n, rng):
        return self.innov(n, rng)

    def calibrate_params(self, mu: float, sigma: float) -> "IIDProcess":
        self.innov.calibrate_params(mu, sigma)
        return self

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
    Stationary AR(p) process with pluggable innovation distribution.

    Parameters
    ----------
    order : int or None
        AR order.  If None, ``phi`` must be provided explicitly.
    phi : array-like or None
        AR coefficients [φ₁, …, φₚ].  If None and ``order`` is set, the
        coefficients are drawn randomly at each ``simulate`` call.
    innov : InnovDist or callable
    drift : float
        Constant drift added to every observation.
    """

    label = "AR"

    def __init__(
        self,
        order: int | None = 1,
        phi:   Sequence[float] | None = None,
        innov: InnovDist | Callable = NormalInnov(),
        drift: float = 0.0,
    ):
        if phi is None and order is None:
            raise ValueError("Provide either 'order' (random coeffs) or 'phi'.")
        self.order = order if phi is None else len(phi)
        self.phi   = None if phi is None else np.asarray(phi, dtype=float)
        self.innov = innov
        self.drift = drift

    def simulate(self, n, rng):
        phi = (
            _make_stationary_ar(self.order, rng)
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

    def _repr_params(self):
        phi_str = "random" if self.phi is None else self.phi.tolist()
        return f"phi={phi_str}, innov={self.innov!r}, drift={self.drift}"


# ─────────────────────────────────────────────────────────────────────────────
# GARCH family  (via arch)
# ─────────────────────────────────────────────────────────────────────────────

class GARCHProcess(DGP):
    """
    Pure GARCH(p, q) with zero mean, driven by arch.

    Parameters
    ----------
    p, q : int
        GARCH lag orders.
    omega, alpha, beta : float
        Standard GARCH(1,1) parameters.  For higher orders pass lists.
    dist : str
        Innovation distribution recognised by arch: ``'normal'``, ``'t'``,
        ``'skewt'``, ``'ged'``.
    dist_params : dict
        Extra distribution parameters, e.g. ``{"nu": 8}`` for Student-t.
    """

    label = "GARCH"

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        omega: float = 0.05,
        alpha: float | list = 0.10,
        beta:  float | list = 0.85,
        dist:  str = "normal",
        dist_params: dict | None = None,
    ):
        self.p    = p
        self.q    = q
        self.dist = dist

        alpha_ = [alpha] if np.isscalar(alpha) else list(alpha)
        beta_  = [beta]  if np.isscalar(beta)  else list(beta)
        self._params = [omega] + alpha_ + beta_

        dp = dist_params or {}
        if dist == "t":
            self._params += [dp.get("nu", 8.0)]
        elif dist == "skewt":
            self._params += [dp.get("nu", 8.0), dp.get("lam", 0.0)]
        elif dist == "ged":
            self._params += [dp.get("nu", 1.5)]

    def simulate(self, n, rng):
        """GARCH reproducibility PENDING"""
        am  = arch_model(y=None, mean="Zero", vol="GARCH",
                         p=self.p, q=self.q, dist=self.dist)
        sim = am.simulate(params=self._params, nobs=n, burn=500)
        return sim["data"].values

    def _repr_params(self):
        return f"p={self.p}, q={self.q}, params={self._params}, dist={self.dist!r}"


class ARGARCHProcess(DGP):
    """
    AR(p_mean)-GARCH(p_vol, q_vol) process with optional Student-t innovations.

    This is the workhorse for realistic financial return simulation.

    Parameters
    ----------
    ar_lags : int
        Number of AR lags in the mean equation.
    p_vol, q_vol : int
        GARCH orders for the variance equation.
    params : list
        Full parameter vector in arch order:
        [const, φ₁, …, φₚ, ω, α₁, …, αₚ, β₁, …, βq, (ν if dist='t')]
        If None, sensible defaults are used.
    dist : str
        ``'normal'``, ``'t'``, ``'skewt'``, ``'ged'``.
    """

    label = "AR-GARCH"

    def __init__(
        self,
        ar_lags: int = 1,
        p_vol:   int = 1,
        q_vol:   int = 1,
        params:  list | None = None,
        dist:    str = "t",
    ):
        self.ar_lags = ar_lags
        self.p_vol   = p_vol
        self.q_vol   = q_vol
        self.dist    = dist
        self._params = params or self._default_params()

    def _default_params(self) -> list:
        const  = 0.0
        phi    = [0.1] * self.ar_lags
        omega  = 0.05
        alpha  = [0.08] * self.p_vol
        beta   = [0.87] * self.q_vol
        p      = [const] + phi + [omega] + alpha + beta
        if self.dist == "t":
            p += [8.0]
        return p

    def simulate(self, n, rng):
        am  = arch_model(y=None, mean="AR", lags=self.ar_lags,
                         vol="GARCH", p=self.p_vol, q=self.q_vol, dist=self.dist)
        sim = am.simulate(params=self._params, nobs=n, burn=500)
        return sim["data"].values

    def _repr_params(self):
        return (f"ar_lags={self.ar_lags}, p_vol={self.p_vol}, "
                f"q_vol={self.q_vol}, dist={self.dist!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Outlier injection wrapper
# ─────────────────────────────────────────────────────────────────────────────

class WithOutliers(DGP):
    """
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

    def _repr_params(self):
        return f"{self.base!r}, fraction={self.fraction}, scale={self.scale}"


# ─────────────────────────────────────────────────────────────────────────────
# Constant-mean GARCH  (arch 'Constant' mean equation)
# ─────────────────────────────────────────────────────────────────────────────

class ConstMeanGARCHProcess(DGP):
    """
    GARCH(p, q) with a constant mean:  y_t = μ + ε_t,  ε_t ~ GARCH.

    Calibration for unit unconditional variance and target SR θ₀:
        ω = 1 − Σα − Σβ,   μ = θ₀

    Parameters
    ----------
    mu : float           constant mean (= target SR when σ_uncond = 1)
    omega, alpha, beta : GARCH(1,1) parameters
    p, q : int           GARCH orders
    dist : str           'normal', 't', 'skewt', 'ged'
    dist_params : dict   e.g. {"nu": 8} for Student-t
    """

    label = "ConstMean-GARCH"

    def __init__(
        self,
        mu:    float = 0.0,
        p:     int   = 1,
        q:     int   = 1,
        omega: float = 0.05,
        alpha: float | list = 0.10,
        beta:  float | list = 0.85,
        dist:  str = "normal",
        dist_params: dict | None = None,
    ):
        self.mu, self.p, self.q, self.dist = mu, p, q, dist
        alpha_ = [alpha] if np.isscalar(alpha) else list(alpha)
        beta_  = [beta]  if np.isscalar(beta)  else list(beta)
        self._params = [mu, omega] + alpha_ + beta_
        dp = dist_params or {}
        if dist == "t":
            self._params += [dp.get("nu", 8.0)]
        elif dist == "skewt":
            self._params += [dp.get("nu", 8.0), dp.get("lam", 0.0)]
        elif dist == "ged":
            self._params += [dp.get("nu", 1.5)]

    def simulate(self, n, rng):
        am  = arch_model(y=None, mean="Constant", vol="GARCH",
                         p=self.p, q=self.q, dist=self.dist)
        sim = am.simulate(params=self._params, nobs=n, burn=500)
        return sim["data"].values

    def _repr_params(self):
        return f"mu={self.mu}, p={self.p}, q={self.q}, params={self._params}, dist={self.dist!r}"