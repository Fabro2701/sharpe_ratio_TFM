import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from typing import Any

from core.model_selection import evaluate_models, SeriesReport
from core import model_selection
from core import models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _avar_col_name(model) -> str:
    """
    Canonical name used BOTH as CI column suffix and in fit_to_avar_map values.

    Always uses the class name so the mapping is unambiguous regardless of
    whether a model carries a short_name attribute.
    """
    return type(model).__name__


def pval_to_stars(pval: float) -> str:
    """Converts a p-value into standard statistical significance stars."""
    if np.isnan(pval):
        return ""
    if pval < 0.01:
        return "***"
    elif pval < 0.05:
        return "**"
    elif pval < 0.10:
        return "*"
    return ""


def _parse_ci(ci_str: str) -> tuple[float, float] | None:
    """
    Parse "[lower, upper]" back to floats.

    BUG FIX 4: split on ", " (comma-space) rather than bare "," so that
    negative lower bounds (e.g. "[-1.2345, -0.0010]") are handled correctly.
    """
    if ci_str in ("NaN", "Error", ""):
        return None
    try:
        clean = ci_str.strip("[] ")
        lower_str, upper_str = clean.split(", ", maxsplit=1)
        return float(lower_str), float(upper_str)
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def build_summary_table(
    series_list: list[np.ndarray],
    reports: list[Any],
    avar_models: list[Any],
    series_names: list[str] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Builds a summary table containing T, point-estimate SR, SR confidence
    intervals for each avar model, the best-fit model name, and diagnostic
    significance stars.

    Column naming
    -------------
    CI columns are named  ``CI_<ClassName>``  where <ClassName> is always the
    Python class name of the avar model (via _avar_col_name).  This makes the
    name unambiguous and consistent with the keys expected by
    plot_sr_intervals.
    """
    z_crit = stats.norm.ppf(1 - alpha / 2)
    rows = []

    for i, (x, rep) in enumerate(zip(series_list, reports)):
        T = rep.n_obs
        s_name = series_names[i] if series_names else f"Series_{rep.series_index}"

        std_x = np.std(x, ddof=1)
        sr_h = float(np.mean(x) / std_x) if std_x != 0 else float("nan")

        row_data: dict[str, Any] = {
            "Series": s_name,
            "T": T,
            "SR": sr_h,
        }

        # --- confidence intervals ----------------------------------------
        for amodel in avar_models:
            col_name = _avar_col_name(amodel)   # BUG FIX 1: always class name

            try:
                params_h = amodel.fit(x)
                V = float(amodel.avar(sr_h, **params_h))

                if np.isnan(V) or V < 0:
                    ci_str = "NaN"
                else:
                    se = np.sqrt(V / T)
                    lower = sr_h - z_crit * se
                    upper = sr_h + z_crit * se
                    ci_str = f"[{lower:.4f}, {upper:.4f}]"
            except Exception:
                ci_str = "Error"

            row_data[f"CI_{col_name}"] = ci_str

        # --- model selection ---------------------------------------------
        row_data["Best_Fit_BIC"] = rep.best_model_bic

        # --- diagnostic stars --------------------------------------------
        diag = rep.diagnostics
        row_data["Ljung-Box"]     = pval_to_stars(diag.ljung_box_pval)
        row_data["ARCH-LM"]       = pval_to_stars(diag.arch_lm_pval)
        row_data["Breusch-Pagan"] = pval_to_stars(diag.bp_pval)
        row_data["ADF"]           = pval_to_stars(diag.adf_pval)
        row_data["KPSS"]          = pval_to_stars(diag.kpss_pval)
        row_data["Jarque-Bera"]   = pval_to_stars(diag.jarque_bera_pval)
        row_data["Shapiro"]       = pval_to_stars(diag.shapiro_pval)

        rows.append(row_data)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_sr_intervals(
    df_summary: pd.DataFrame,
    fit_to_avar_map: dict[str, str] | None = None,
    manual_overrides: dict[str, str] | None = None,
) -> None:
    """
    Plots horizontal confidence intervals for each series.

    fit_to_avar_map  : {best_fit_model_class_name -> avar_model_class_name}
                       Used to highlight the "appropriate" avar CI automatically.
    manual_overrides : {series_name -> avar_model_class_name}
                       Takes priority over fit_to_avar_map for specific series.
    """
    fit_to_avar_map  = fit_to_avar_map  or {}
    manual_overrides = manual_overrides or {}

    ci_cols = [col for col in df_summary.columns if col.startswith("CI_")]

    for _, row in df_summary.iterrows():
        series_name = row["Series"]
        sr_point    = row["SR"]
        best_fit    = row["Best_Fit_BIC"]

        # BUG FIX 2: use "is None" so an explicit empty-string override is
        # respected rather than silently falling through to the map.
        target_model = manual_overrides.get(series_name)   # None if absent
        if target_model is None and best_fit in fit_to_avar_map:
            target_model = fit_to_avar_map[best_fit]

        fig, ax = plt.subplots(figsize=(8, max(3, len(ci_cols) * 0.8 + 1)))

        y_ticks:  list[int]   = []
        y_labels: list[str]   = []
        y_counter = 0   # BUG FIX 3: independent counter so positions are
                        # always contiguous even when CIs are skipped.

        for col in ci_cols:
            model_name = col.removeprefix("CI_")
            ci_str     = row[col]

            parsed = _parse_ci(ci_str)   # BUG FIX 4: robust parser
            if parsed is None:
                continue

            lower, upper = parsed
            y_counter += 1
            y_pos = y_counter

            y_ticks.append(y_pos)
            y_labels.append(model_name)

            # BUG FIX 1 (continued): model_name is now always the class name,
            # matching the values stored in fit_to_avar_map.
            is_highlight = (model_name == target_model)
            color   = "#D32F2F" if is_highlight else "#1976D2"
            lw      = 3         if is_highlight else 1.5
            zorder  = 5         if is_highlight else 3

            ax.plot([lower, upper], [y_pos, y_pos],
                    color=color, lw=lw, zorder=zorder, solid_capstyle="round")
            ax.plot(sr_point, y_pos,
                    marker="o", color=color, markersize=8, zorder=zorder)

        if not y_ticks:
            ax.set_title(f"SR Confidence Intervals – {series_name}\n(no valid CIs)")
            plt.tight_layout()
            plt.show()
            continue

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_ylim(0.5, y_counter + 0.5)
        ax.set_xlabel("Sharpe Ratio")
        ax.grid(axis="x", linestyle="--", alpha=0.4)

        title = f"SR Confidence Intervals – {series_name}\nBest fit (BIC): {best_fit}"
        if target_model:
            title += f"  |  highlighted: {target_model}"
        ax.set_title(title)

        ax.axvline(sr_point, color="gray", linestyle="--", alpha=0.5, zorder=1)

        if target_model:
            ax.plot([], [], color="#D32F2F", lw=3, label=f"Selected: {target_model}")
            ax.plot([], [], color="#1976D2", lw=1.5, label="Other models")
            ax.legend(loc="lower right", framealpha=0.8)

        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    n1, n2, n3 = 500, 300, 700

    s0 = rng.normal(0.05, 1.0, n1)

    phi = 0.6
    eps = stats.t.rvs(df=5, size=n2, random_state=42)
    s1  = np.zeros(n2)
    for t in range(1, n2):
        s1[t] = 0.02 + phi * s1[t - 1] + eps[t]

    omega, alpha_g, beta_g = 0.1, 0.1, 0.85
    s2     = np.zeros(n3)
    sigma2 = np.zeros(n3)
    sigma2[0] = omega / (1 - alpha_g - beta_g)
    for t in range(1, n3):
        sigma2[t] = omega + alpha_g * s2[t - 1] ** 2 + beta_g * sigma2[t - 1]
        s2[t]     = rng.normal(0, np.sqrt(sigma2[t]))

    fit_models = [
        model_selection.IIDNormal(),
        model_selection.IIDStudent(),
        model_selection.IIDSkewStudent(),
        model_selection.AR1Normal(),
        model_selection.AR1Student(),
    ]

    reports = evaluate_models(series_list=[s0, s1, s2], model_list=fit_models)

    avar_models = [
        models.IIDNormalModel(),
        models.IIDStudentTModel(),
        models.IIDNonNormalModel(),
        models.AR1NormalModel(),
        models.AR1NonNormalModel(),
    ]
    series_names = ["Pure IID Normal", "AR(1) Fat Tails", "GARCH Volatility"]

    df_summary = build_summary_table(
        series_list=[s0, s1, s2],
        reports=reports,
        avar_models=avar_models,
        series_names=series_names,
        alpha=0.05,
    )

    df_summary["SR"] = df_summary["SR"].round(4)
    print(df_summary.to_string())

    # Map: fit-model class name  →  avar-model class name
    # Both sides must be class names (that's what _avar_col_name returns).
    fit_to_avar_mapping = {
        "iid_normal":      "IIDNormalModel",
        "iid_t":     "IIDStudentTModel",
        "iid_skew_t": "IIDNonNormalModel",
        "ar1_normal":      "AR1NormalModel",
        "ar1_t":     "AR1NonNormalModel",
    }

    overrides: dict[str, str] = {
        # "Pure IID Normal": "IIDNonNormalModel"
    }

    plot_sr_intervals(
        df_summary=df_summary,
        fit_to_avar_map=fit_to_avar_mapping,
        manual_overrides=overrides,
    )