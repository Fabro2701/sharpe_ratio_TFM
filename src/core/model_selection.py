"""
model_selection.py

Time Series Model Selection API
================================
Runs a battery of diagnostic tests on each series and ranks a list of candidate
models by BIC / AIC / log-likelihood so you can see which model fits best.

All models are built from the `arch` library's composable building blocks:

    Mean        : ConstantMean | ARX(lags=1)
    Volatility  : ConstantVariance | GARCH(p=1, q=1)
    Distribution: Normal | StudentsT | SkewStudent | GeneralizedError

Dependencies
------------
    pip install numpy scipy statsmodels arch pandas
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from arch.univariate import (
    ARX,
    ConstantMean,
    ConstantVariance,
    GARCH,
    GeneralizedError,
    Normal,
    SkewStudent,
    StudentsT,
)
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, kpss, acf

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1.  DATA CLASSES FOR RESULTS
# ─────────────────────────────────────────────

@dataclass
class DiagnosticResult:
    """Outcome of every statistical test run on a single series."""
    # Autocorrelation
    ljung_box_stat: float = float("nan")
    ljung_box_pval: float = float("nan")          # H0: no autocorrelation
    durbin_watson: float = float("nan")            # ~2 → no autocorr
    acf_lag1: float = float("nan")                 # sample ACF at lag 1

    # Heteroscedasticity
    arch_lm_stat: float = float("nan")
    arch_lm_pval: float = float("nan")             # H0: no ARCH effects
    bp_stat: float = float("nan")
    bp_pval: float = float("nan")                  # Breusch-Pagan H0: homoscedastic

    # Stationarity
    adf_stat: float = float("nan")
    adf_pval: float = float("nan")                 # H0: unit root (non-stationary)
    kpss_stat: float = float("nan")
    kpss_pval: float = float("nan")                # H0: stationary

    # Normality
    jarque_bera_stat: float = float("nan")
    jarque_bera_pval: float = float("nan")         # H0: normal
    shapiro_stat: float = float("nan")
    shapiro_pval: float = float("nan")             # H0: normal (small samples)

    # Descriptive
    n: int = 0
    mean: float = float("nan")
    std: float = float("nan")
    skewness: float = float("nan")
    excess_kurtosis: float = float("nan")          # normal → 0

    def summary(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    def flags(self, alpha: float = 0.05) -> dict[str, bool]:
        """Return True where H0 is *rejected* at the given significance level."""
        return {
            "autocorrelation_lag1":   self.ljung_box_pval < alpha,
            "arch_effects":           self.arch_lm_pval  < alpha,
            "heteroscedastic_bp":     self.bp_pval        < alpha,
            "non_stationary_adf":     self.adf_pval       > alpha,   # fail to reject unit root
            "non_stationary_kpss":    self.kpss_pval      < alpha,   # reject stationarity
            "non_normal_jb":          self.jarque_bera_pval < alpha,
            "non_normal_shapiro":     self.shapiro_pval   < alpha,
        }


@dataclass
class FitResult:
    """Goodness-of-fit metrics for one (series, model) pair."""
    model_name: str
    n_params: int
    n_obs: int
    log_likelihood: float
    aic: float
    bic: float
    extra: dict[str, Any] = field(default_factory=dict)   # model-specific params

    @property
    def hqic(self) -> float:
        """Hannan-Quinn information criterion."""
        return -2 * self.log_likelihood + 2 * self.n_params * np.log(np.log(self.n_obs))


@dataclass
class SeriesReport:
    """Full report for a single series."""
    series_index: int
    n_obs: int
    diagnostics: DiagnosticResult
    fit_results: list[FitResult]          # sorted best → worst by BIC
    best_model_bic: str
    best_model_aic: str
    best_model_ll: str


# ─────────────────────────────────────────────
# 2.  DIAGNOSTIC ENGINE
# ─────────────────────────────────────────────

def run_diagnostics(x: np.ndarray) -> DiagnosticResult:
    """
    Run a full battery of statistical tests on a 1-D array x.

    Tests
    -----
    - Ljung-Box (lag 1) – autocorrelation in levels
    - Durbin-Watson – first-order autocorrelation
    - ARCH-LM (lag 1) – conditional heteroscedasticity
    - Breusch-Pagan – heteroscedasticity against a time trend
    - ADF – unit-root (non-stationarity)
    - KPSS – stationarity
    - Jarque-Bera – normality
    - Shapiro-Wilk – normality (reliable up to n ≈ 5000)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    d = DiagnosticResult(n=n)

    d.mean = float(np.mean(x))
    d.std = float(np.std(x, ddof=1))
    d.skewness = float(stats.skew(x))
    d.excess_kurtosis = float(stats.kurtosis(x))   # Fisher definition (normal=0)

    # -- ACF lag 1 --
    acf_vals = acf(x, nlags=1, fft=True)
    d.acf_lag1 = float(acf_vals[1])

    # -- Ljung-Box at lag 1 --
    try:
        lb = acorr_ljungbox(x, lags=[1], return_df=True)
        d.ljung_box_stat = float(lb["lb_stat"].iloc[0])
        d.ljung_box_pval = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        pass

    # -- Durbin-Watson --
    try:
        d.durbin_watson = float(durbin_watson(x))
    except Exception:
        pass

    # -- ARCH-LM (squared demeaned series, lag 1) --
    try:
        from statsmodels.stats.diagnostic import het_arch
        arch_stat, arch_pval, _, _ = het_arch(x, nlags=1)
        d.arch_lm_stat = float(arch_stat)
        d.arch_lm_pval = float(arch_pval)
    except Exception:
        pass

    # -- Breusch-Pagan (regress x² on time trend) --
    try:
        t = np.arange(n)
        exog = np.column_stack([np.ones(n), t])
        resid = x - np.polyval(np.polyfit(t, x, 1), t)
        bp_lm, bp_pval, _, _ = het_breuschpagan(resid ** 2, exog)
        d.bp_stat = float(bp_lm)
        d.bp_pval = float(bp_pval)
    except Exception:
        pass

    # -- ADF --
    try:
        adf_res = adfuller(x, autolag="AIC")
        d.adf_stat = float(adf_res[0])
        d.adf_pval = float(adf_res[1])
    except Exception:
        pass

    # -- KPSS --
    try:
        kpss_res = kpss(x, regression="c", nlags="auto")
        d.kpss_stat = float(kpss_res[0])
        d.kpss_pval = float(kpss_res[1])
    except Exception:
        pass

    # -- Jarque-Bera --
    try:
        jb_stat, jb_pval = stats.jarque_bera(x)
        d.jarque_bera_stat = float(jb_stat)
        d.jarque_bera_pval = float(jb_pval)
    except Exception:
        pass

    # -- Shapiro-Wilk (cap at 5000) --
    try:
        sw_stat, sw_pval = stats.shapiro(x[:5000])
        d.shapiro_stat = float(sw_stat)
        d.shapiro_pval = float(sw_pval)
    except Exception:
        pass

    return d


# ─────────────────────────────────────────────
# 3.  MODEL BASE CLASS  +  IMPLEMENTATIONS
# ─────────────────────────────────────────────
#
# Every model is three composable arch choices:
#
#   Mean        ConstantMean          μ (constant drift)
#               ARX(lags=1)           μ + φ·x_{t-1}  (AR(1))
#
#   Volatility  ConstantVariance      σ² constant   (IID / AR)
#               GARCH(p=1, q=1)       σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
#
#   Distribution Normal               z_t ~ N(0,1)
#               StudentsT             z_t ~ t(ν)        fat tails
#               SkewStudent           z_t ~ skew-t(ν,λ) fat tails + asymmetry
#               GeneralizedError      z_t ~ GED(ν)      sub/super-Gaussian tails
#
# ─────────────────────────────────────────────

class BaseModel(ABC):
    """
    Contract every model must satisfy.
    Subclasses only need to set `short_name` and implement `_build()`,
    which returns a configured (but unfitted) arch mean-model object.
    """

    short_name: str = "base"

    @abstractmethod
    def _build(self, x: np.ndarray):
        """Return an arch univariate mean-model ready to call .fit() on."""

    def fit(self, x: np.ndarray) -> "FitResult":
        x = np.asarray(x, dtype=float)
        model = self._build(x)
        res = model.fit(disp="off", show_warning=False)
        return FitResult(
            model_name=self.short_name,
            n_params=int(res.num_params),
            n_obs=int(res.nobs),
            log_likelihood=float(res.loglikelihood),
            aic=float(res.aic),
            bic=float(res.bic),
            extra=res.params.to_dict(),
        )


# ── IID models  (ConstantMean + ConstantVariance) ────────────────────────────

class IIDNormal(BaseModel):
    """x_t = μ + σ·z_t,  z_t ~ N(0,1)  iid   [k=2: μ, σ]"""
    short_name = "iid_normal"

    def _build(self, x):
        m = ConstantMean(x)
        m.volatility = ConstantVariance()
        m.distribution = Normal()
        return m


class IIDStudent(BaseModel):
    """x_t = μ + σ·z_t,  z_t ~ t(ν)  iid   [k=3: μ, σ, ν]"""
    short_name = "iid_t"

    def _build(self, x):
        m = ConstantMean(x)
        m.volatility = ConstantVariance()
        m.distribution = StudentsT()
        return m


class IIDSkewStudent(BaseModel):
    """x_t = μ + σ·z_t,  z_t ~ skew-t(ν, λ)  iid   [k=4: μ, σ, ν, λ]

    Hansen (1994) skewed-t: asymmetry + fat tails.
    λ ∈ (−1, 1);  ν > 2 for finite variance.
    """
    short_name = "iid_skew_t"

    def _build(self, x):
        m = ConstantMean(x)
        m.volatility = ConstantVariance()
        m.distribution = SkewStudent()
        return m


class IIDGeneralizedError(BaseModel):
    """x_t = μ + σ·z_t,  z_t ~ GED(ν)  iid   [k=3: μ, σ, ν]

    Generalised Error Distribution: ν=2 → Normal; ν=1 → Laplace; ν→∞ → Uniform.
    """
    short_name = "iid_ged"

    def _build(self, x):
        m = ConstantMean(x)
        m.volatility = ConstantVariance()
        m.distribution = GeneralizedError()
        return m


# ── AR(1) models  (ARX(lags=1) + ConstantVariance) ───────────────────────────

class AR1Normal(BaseModel):
    """x_t = μ + φ·x_{t-1} + σ·z_t,  z_t ~ N(0,1)   [k=3: μ, φ, σ]"""
    short_name = "ar1_normal"

    def _build(self, x):
        m = ARX(x, lags=1)
        m.volatility = ConstantVariance()
        m.distribution = Normal()
        return m


class AR1Student(BaseModel):
    """x_t = μ + φ·x_{t-1} + σ·z_t,  z_t ~ t(ν)   [k=4: μ, φ, σ, ν]"""
    short_name = "ar1_t"

    def _build(self, x):
        m = ARX(x, lags=1)
        m.volatility = ConstantVariance()
        m.distribution = StudentsT()
        return m


class AR1SkewStudent(BaseModel):
    """x_t = μ + φ·x_{t-1} + σ·z_t,  z_t ~ skew-t(ν,λ)   [k=5]"""
    short_name = "ar1_skew_t"

    def _build(self, x):
        m = ARX(x, lags=1)
        m.volatility = ConstantVariance()
        m.distribution = SkewStudent()
        return m


# ── GARCH(1,1) models  (ConstantMean + GARCH(1,1)) ───────────────────────────

class GARCH11Normal(BaseModel):
    """σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1},  z_t ~ N(0,1)   [k=4: μ, ω, α, β]"""
    short_name = "garch11_normal"

    def _build(self, x):
        m = ConstantMean(x)
        m.volatility = GARCH(p=1, q=1)
        m.distribution = Normal()
        return m


class GARCH11Student(BaseModel):
    """GARCH(1,1) with Student-t innovations   [k=5: μ, ω, α, β, ν]"""
    short_name = "garch11_t"

    def _build(self, x):
        m = ConstantMean(x)
        m.volatility = GARCH(p=1, q=1)
        m.distribution = StudentsT()
        return m


class GARCH11SkewStudent(BaseModel):
    """GARCH(1,1) with skewed-t innovations   [k=6: μ, ω, α, β, ν, λ]"""
    short_name = "garch11_skew_t"

    def _build(self, x):
        m = ConstantMean(x)
        m.volatility = GARCH(p=1, q=1)
        m.distribution = SkewStudent()
        return m


# ── AR(1)-GARCH(1,1) models  (the full combo) ────────────────────────────────

class AR1GARCH11Normal(BaseModel):
    """AR(1) mean + GARCH(1,1) vol + Normal   [k=5: μ, φ, ω, α, β]"""
    short_name = "ar1_garch11_normal"

    def _build(self, x):
        m = ARX(x, lags=1)
        m.volatility = GARCH(p=1, q=1)
        m.distribution = Normal()
        return m


class AR1GARCH11Student(BaseModel):
    """AR(1) mean + GARCH(1,1) vol + Student-t   [k=6: μ, φ, ω, α, β, ν]"""
    short_name = "ar1_garch11_t"

    def _build(self, x):
        m = ARX(x, lags=1)
        m.volatility = GARCH(p=1, q=1)
        m.distribution = StudentsT()
        return m
    
class AR1GARCH11SkewStudent(BaseModel):
    """AR(1) mean + GARCH(1,1) vol + skewed-t innovations   [k=7: μ, φ, ω, α, β, ν, λ]"""
    short_name = "ar1_garch11_skew_t"

    def _build(self, x):
        m = ARX(x, lags=1)
        m.volatility = GARCH(p=1, q=1)
        m.distribution = SkewStudent()
        return m


# ─────────────────────────────────────────────
# 4.  MAIN API
# ─────────────────────────────────────────────

def evaluate_models(
    series_list: list[np.ndarray],
    model_list: list[BaseModel],
) -> list[SeriesReport]:
    """
    Evaluate a battery of diagnostic tests and fit a set of models to each
    series in `series_list`.

    Parameters
    ----------
    series_list : list of array-like
        Each element is a 1-D sequence of observations (can differ in length).
    model_list : list of BaseModel
        Instantiated model objects; each must have `short_name` and `fit()`.
    alpha : float
        Significance level used to flag test rejections (default 0.05).

    Returns
    -------
    list[SeriesReport]
        One report per series.  Each report contains:
        - Diagnostic test results + flags
        - FitResult for every model, sorted best → worst by BIC
        - The name of the best model under BIC, AIC, and log-likelihood

    Example
    -------
    >>> models = [IIDNormal(), IIDStudent(), AR1Normal(), GARCH11Normal()]
    >>> reports = evaluate_models([returns_1, returns_2], models)
    >>> for r in reports:
    ...     print(r.best_model_bic, r.diagnostics.flags())
    """
    reports: list[SeriesReport] = []

    for idx, raw in enumerate(series_list):
        x = np.asarray(raw, dtype=float)

        # ── diagnostics ──────────────────────────────────────────
        diag = run_diagnostics(x)

        # ── fit every model ──────────────────────────────────────
        fit_results: list[FitResult] = []
        for model in model_list:
            try:
                fr = model.fit(x)
            except Exception as exc:
                # Return a sentinel so the series is still reported
                fr = FitResult(
                    model_name=model.short_name,
                    n_params=0,
                    n_obs=len(x),
                    log_likelihood=float("-inf"),
                    aic=float("inf"),
                    bic=float("inf"),
                    extra={"error": str(exc)},
                )
            fit_results.append(fr)

        # ── rank ─────────────────────────────────────────────────
        fit_results_bic = sorted(fit_results, key=lambda r: r.bic)
        best_bic = fit_results_bic[0].model_name
        best_aic = min(fit_results, key=lambda r: r.aic).model_name
        best_ll  = max(fit_results, key=lambda r: r.log_likelihood).model_name

        reports.append(SeriesReport(
            series_index=idx,
            n_obs=len(x),
            diagnostics=diag,
            fit_results=fit_results_bic,
            best_model_bic=best_bic,
            best_model_aic=best_aic,
            best_model_ll=best_ll,
        ))

    return reports


def summary_table(reports: list[SeriesReport]) -> pd.DataFrame:
    """
    Convert a list of SeriesReports into a tidy DataFrame for quick inspection.

    Columns: series_idx | model | n_params | log_lik | aic | bic | hqic
    """
    rows = []
    for rep in reports:
        for fr in rep.fit_results:
            rows.append({
                "series_idx": rep.series_index,
                "model":      fr.model_name,
                "n_params":   fr.n_params,
                "n_obs":      fr.n_obs,
                "log_lik":    round(fr.log_likelihood, 4),
                "aic":        round(fr.aic, 4),
                "bic":        round(fr.bic, 4),
                "hqic":       round(fr.hqic, 4),
            })
    return pd.DataFrame(rows)


def diagnostics_table(reports: list[SeriesReport], alpha: float = 0.05) -> pd.DataFrame:
    """Return a DataFrame with one row per series summarising all test flags."""
    rows = []
    for rep in reports:
        row = {"series_idx": rep.series_index, "n_obs": rep.n_obs}
        row.update(rep.diagnostics.summary())
        row.update({f"flag_{k}": v for k, v in rep.diagnostics.flags(alpha).items()})
        row["best_bic"] = rep.best_model_bic
        row["best_aic"] = rep.best_model_aic
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 5.  QUICK DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Simulate three series with different DGPs
    n1, n2, n3 = 500, 300, 700

    # Series 0: pure iid normal
    s0 = rng.normal(0.05, 1.0, n1)

    # Series 1: AR(1) with fat-tailed shocks
    phi = 0.6
    eps = stats.t.rvs(df=5, size=n2, random_state=42)
    s1 = np.zeros(n2)
    for t in range(1, n2):
        s1[t] = 0.02 + phi * s1[t - 1] + eps[t]

    # Series 2: GARCH-like volatility clustering
    omega, alpha_g, beta_g = 0.1, 0.1, 0.85
    s2 = np.zeros(n3)
    sigma2 = np.zeros(n3)
    sigma2[0] = omega / (1 - alpha_g - beta_g)
    for t in range(1, n3):
        sigma2[t] = omega + alpha_g * s2[t - 1] ** 2 + beta_g * sigma2[t - 1]
        s2[t] = rng.normal(0, np.sqrt(sigma2[t]))

    # Define models
    models = [
        IIDNormal(),
        IIDStudent(),
        IIDSkewStudent(),
        IIDGeneralizedError(),
        AR1Normal(),
        AR1Student(),
        AR1SkewStudent(),
        GARCH11Normal(),
        GARCH11Student(),
        GARCH11SkewStudent(),
        AR1GARCH11Normal(),
        AR1GARCH11Student(),
    ]

    reports = evaluate_models([s0, s1, s2], models)
    #print(diagnostics_table(reports))

    print("=" * 60)
    print("GOODNESS-OF-FIT TABLE (sorted by BIC within each series)")
    print("=" * 60)
    print(summary_table(reports).to_string(index=False))

    print("\n" + "=" * 60)
    print("BEST MODELS")
    print("=" * 60)
    alpha = 0.05
    for rep in reports:
        flags = rep.diagnostics.flags(alpha)
        active = [k for k, v in flags.items() if v]
        print(
            f"Series {rep.series_index} (n={rep.n_obs}): "
            f"BIC→{rep.best_model_bic}  AIC→{rep.best_model_aic}  "
            f"Flags: {active or 'none'}"
        )