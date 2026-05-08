"""
Hierarchical model selection for return series.

Procedure:
    Step 1 - Ljung-Box on returns           -> AR(1) vs constant mean
    Step 2 - ARCH-LM on mean-eq residuals   -> GARCH(1,1) vs homoskedastic
    Step 3 - JB & skew test on standardized
             residuals from full Gaussian   -> Normal / t / skew-t
             fit
    Step 4 - Refit with chosen distribution,
             Ljung-Box on z and z^2         -> misspecification check
"""

import numpy as np
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy import stats



# ----------------------------------------------------------------------
# Hierarchical analysis
# ----------------------------------------------------------------------
def _build_model(returns, use_ar, use_garch, dist):
    mean_kw = {"mean": "AR", "lags": 1} if use_ar else {"mean": "Constant"}
    vol_kw  = {"vol": "GARCH", "p": 1, "q": 1} if use_garch else {"vol": "Constant"}
    return arch_model(returns, dist=dist, rescale=True, **mean_kw, **vol_kw)


def hierarchical_analysis(returns, alpha=0.05, lags=10, name=""):
    out = {"series": name}

    # ---- Step 1: serial correlation in returns ----
    lb_r        = acorr_ljungbox(returns, lags=[lags])
    p_lb_r      = float(lb_r["lb_pvalue"].iloc[0])
    use_ar      = p_lb_r < alpha
    out["S1_LB_ret_p"] = round(p_lb_r, 4)
    out["S1_AR?"]      = use_ar

    # ---- Step 2: ARCH effects in residuals from the mean equation ----
    mean_only       = _build_model(returns, use_ar, use_garch=False, dist="normal")
    fit_mean        = mean_only.fit(disp="off")
    resid           = np.asarray(fit_mean.resid)
    resid           = resid[~np.isnan(resid)]

    arch_test       = het_arch(resid, nlags=lags)
    p_arch          = float(arch_test[1])
    use_garch       = p_arch < alpha
    out["S2_ARCH_LM_p"] = round(p_arch, 4)
    out["S2_GARCH?"]    = use_garch

    # ---- Step 3: innovation distribution from a Gaussian full fit ----
    full_gauss      = _build_model(returns, use_ar, use_garch, dist="normal")
    fit_full        = full_gauss.fit(disp="off")
    z               = np.asarray(fit_full.std_resid)
    z               = z[~np.isnan(z)]

    jb_stat, jb_p   = stats.jarque_bera(z)
    sk_stat, sk_p   = stats.skewtest(z)
    is_normal       = jb_p >= alpha
    is_symmetric    = sk_p >= alpha

    if is_normal:
        innov, dist_str = "Normal", "normal"
    elif is_symmetric:
        innov, dist_str = "Student-t", "t"
    else:
        innov, dist_str = "Skew-t", "skewt"

    out["S3_JB_p"]      = round(float(jb_p), 4)
    out["S3_skew_p"]    = round(float(sk_p), 4)
    out["S3_skewness"]  = round(float(stats.skew(z)), 3)
    out["S3_exkurt"]    = round(float(stats.kurtosis(z)), 3)
    out["S3_innov"]     = innov

    # ---- Step 4: refit with chosen distribution, misspecification checks ----
    final           = _build_model(returns, use_ar, use_garch, dist=dist_str)
    fit_final       = final.fit(disp="off")
    z_f             = np.asarray(fit_final.std_resid)
    z_f             = z_f[~np.isnan(z_f)]

    p_lb_z          = float(acorr_ljungbox(z_f,    lags=[lags])["lb_pvalue"].iloc[0])
    p_lb_z2         = float(acorr_ljungbox(z_f**2, lags=[lags])["lb_pvalue"].iloc[0])
    out["S4_LB_z_p"]  = round(p_lb_z, 4)
    out["S4_LB_z2_p"] = round(p_lb_z2, 4)

    mean_part = "AR(1)"      if use_ar    else "Const"
    var_part  = "GARCH(1,1)" if use_garch else "Const"
    out["final_spec"] = f"{mean_part} | {var_part} | {innov}"

    return out
