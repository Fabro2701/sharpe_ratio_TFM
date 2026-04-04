# sr_study_analysis.py  —  Orchestration and visualisation for SR inference studies.
#
# Supports all one-sample StudyTypes:
#   TWO_SIDED_COVERAGE, ONE_SIDED_COVERAGE
#   TWO_SIDED_POWER,    ONE_SIDED_POWER
#
# Two-sample variants are intentionally excluded here (separate workflow).

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from core.sr_sim import StudyType, DGPSpec, run_study
from core.dgp import DGP_EXAMPLES
from config import RESULTS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Experiment specification dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentSpec:
    """
    Full description of one sweep experiment.

    Parameters
    ----------
    scenario : (list[str], list[str])
        Tuple of (dgp_names, model_short_names).
    param_name : str
        Which parameter is swept: 'T', 'sr', or 'n_sim'.
    param_values : list
        Values to sweep over.
    study_type : StudyType
        Which inference scenario to run (coverage or power).
    th_moments : bool
        Whether to use theoretical moments.
    null_sr : float | None
        Null hypothesis SR for power studies. Ignored for coverage.
        If None and study_type is a power variant, raises at runtime.
    n_default / T_default / sr_default : float
        Fixed values for the two non-swept parameters.
    alpha : float
        Nominal error rate.
    seed : int
    n_jobs : int
    """
    scenario:     tuple[list[str], list[str]]
    param_name:   str
    param_values: list
    study_type:   StudyType                = StudyType.TWO_SIDED_COVERAGE
    th_moments:   bool                     = False
    null_sr:      float | None             = None
    n_default:    int                      = 10_000
    T_default:    int                      = 500
    sr_default:   float                    = 0.5
    calib_mu:     float                    = None
    calib_sigma:  float                    = None
    alpha:        float                    = 0.05
    seed:         int                      = 42
    n_jobs:       int                      = 1

    # ── convenience ──────────────────────────────────────────────────────────

    @property
    def dgps(self)   -> list[str]: return self.scenario[0]
    @property
    def models(self) -> list[str]: return self.scenario[1]

    @property
    def metric(self) -> str:
        return self.study_type.metric_name

    def file_stem(self, param_value) -> str:
        """Unique filename stem for one cell of the sweep."""
        st_tag = self.study_type.name.lower()
        th_tag = "theo_" if self.th_moments else ""
        return f"{st_tag}_{th_tag}{self.param_name}{param_value}"

    def _resolve_null_sr(self, target_sr: float) -> float:
        """
        For coverage: null_sr = target_sr (null is true).
        For power:    null_sr must be set explicitly in the spec.
        """
        if not self.study_type.is_power:
            return target_sr
        if self.null_sr is None:
            raise ValueError(
                f"ExperimentSpec.null_sr must be set for power study '{self.study_type.name}'."
            )
        return self.null_sr

    def _param_kwargs(self, param_value) -> dict:
        """Map the swept parameter to (T, n_sim, target_sr)."""
        T, n_sim, sr = self.T_default, self.n_default, self.sr_default
        if self.param_name == "T":
            T = int(param_value)
        elif self.param_name == "sr":
            sr = float(param_value)
        elif self.param_name == "n_sim":
            n_sim = int(param_value)
        else:
            raise ValueError(
                f"Unsupported param_name '{self.param_name}'. "
                "Use 'T', 'sr', or 'n_sim'."
            )
        return dict(T=T, n_sim=n_sim, target_sr=sr)


# ─────────────────────────────────────────────────────────────────────────────
# DGP helper
# ─────────────────────────────────────────────────────────────────────────────

EXTRA_DGPS = {}
def set_extra_dgps(dgps):
    global EXTRA_DGPS
    EXTRA_DGPS = dgps

def _build_dgp_specs(names: list[str]) -> list[DGPSpec]:
    dgps = DGP_EXAMPLES | EXTRA_DGPS
    missing = [n for n in names if n not in dgps]
    if missing:
        raise ValueError(f"Unknown DGP name(s): {missing}. "
                         f"Available: {sorted(DGP_EXAMPLES)}")
    return [DGPSpec(dgps[n](), n) for n in names]


def _build_models(names: list[str]):
    from core.models import REGISTRY
    missing = [n for n in names if n not in REGISTRY]
    if missing:
        raise ValueError(f"Unknown model short_name(s): {missing}. "
                         f"Available: {sorted(REGISTRY)}")
    return [REGISTRY[n] for n in names]


# ─────────────────────────────────────────────────────────────────────────────
# Core runner  (replaces run_coverage_setups — calls run_study directly)
# ─────────────────────────────────────────────────────────────────────────────

def run_setups(
    spec:   ExperimentSpec,
    prefix: str = "",
    out_dir: Path | None = None,
) -> None:
    """
    Run one full parameter sweep defined by *spec*, saving one CSV per value.

    Calls run_study() directly instead of going through the CLI.
    """
    out_dir    = Path(out_dir) if out_dir is not None else RESULTS_DIR
    dgp_specs  = _build_dgp_specs(spec.dgps)
    avar_models = _build_models(spec.models)
    total      = len(spec.param_values)

    for i, param in enumerate(spec.param_values, 1):
        print(f"  [{i}/{total}]  {spec.param_name}={param}  "
              f"({spec.study_type.name})")

        sim_kwargs = spec._param_kwargs(param)
        null_sr    = spec._resolve_null_sr(sim_kwargs["target_sr"])

        results = run_study(
            study_type  = spec.study_type,
            dgp_specs   = dgp_specs,
            avar_models = avar_models,
            null_sr     = null_sr,
            th_moments  = spec.th_moments,
            calib_mu    = spec.calib_mu,
            calib_sigma = spec.calib_sigma,
            alpha       = spec.alpha,
            seed        = spec.seed,
            verbose     = False,
            n_jobs      = spec.n_jobs,
            **sim_kwargs,
        )

        out_path = out_dir / f"{prefix}{spec.file_stem(param)}.csv"
        results.to_csv(out_path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_setups(
    spec:         ExperimentSpec,
    prefix:       str = "",
    out_dir:      Path | None = None,
    dgps:         list[str] | None = None,
    models:       list[str] | None = None,
    param_values: list | None = None,
) -> pd.DataFrame:
    """
    Read saved CSVs for a sweep and return a combined DataFrame.

    *dgps* and *models* can narrow the loaded data to a subset.
    *param_values* overrides spec.param_values (e.g. for partial re-analysis).
    """
    out_dir      = Path(out_dir) if out_dir is not None else RESULTS_DIR
    param_values = param_values if param_values is not None else spec.param_values
    metric       = spec.metric

    all_data = []
    for param in param_values:
        path = out_dir / f"{prefix}{spec.file_stem(param)}.csv"
        df   = pd.read_csv(path)
        df[spec.param_name] = param
        all_data.append(df)

    df_all = pd.concat(all_data, ignore_index=True)

    if dgps:
        df_all = df_all[df_all["dgp_name"].isin(dgps)]
    if models:
        df_all = df_all[df_all["avar_model"].isin(models)]

    # ── summary table ─────────────────────────────────────────────────────────
    cols = [spec.param_name, "dgp_name", "avar_model", "nominal", metric,
            "bias", "rmse"]
    cols = [c for c in cols if c in df_all.columns]
    print(f"=== {spec.study_type.name} — {spec.param_name} sweep ===")
    print(df_all[cols].to_string(index=False))

    df_all["dgp_model_pair"] = df_all["dgp_name"] + " + " + df_all["avar_model"]
    return df_all


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _metric_info(study_type: StudyType, alpha: float) -> tuple[str, float, str]:
    """Return (column_name, target_line_value, y_label)."""
    if study_type.is_power:
        return "power", None, "Empirical Power"          # no fixed target for power
    else:
        return "coverage", 1.0 - alpha, "Empirical Coverage"


def plot_results_by_pair(
    df:          pd.DataFrame,
    spec:        ExperimentSpec,
    alpha:       float = 0.05,
    title:       str   = "",
):
    """
    Bar chart of coverage or power vs. the swept parameter.
    Color = avar_model, hatch = dgp_name.
    """
    metric, target_val, y_label = _metric_info(spec.study_type, alpha)
    param_name = spec.param_name

    sns.set_theme(style="whitegrid")
    df = df.copy()
    df["model_dgp"] = df["avar_model"].astype(str) + " | " + df["dgp_name"].astype(str)

    models     = df["avar_model"].unique()
    dgps       = df["dgp_name"].unique()
    t_levels   = df[param_name].unique()

    base_colors = sns.color_palette("tab10", len(models))
    color_map   = dict(zip(models, base_colors))

    hatch_patterns = ['.', '/', 'O', '-', '*']
    hatch_map      = dict(zip(dgps, hatch_patterns[:len(dgps)]))

    hue_order      = []
    custom_palette = {}
    for m in models:
        for d in dgps:
            combo = f"{m} | {d}"
            hue_order.append(combo)
            custom_palette[combo] = color_map[m]

    g = sns.catplot(
        data=df, x=param_name, y=metric,
        hue="model_dgp", hue_order=hue_order, palette=custom_palette,
        kind="bar", height=5, aspect=1.5, errorbar=None,
    )

    for ax in g.axes.flat:
        if target_val is not None:
            ax.axhline(target_val, color="black", linestyle="--",
                       linewidth=2, label=f"Target ({target_val:.2f})")
        for i, bar in enumerate(ax.patches):
            hue_idx = i // len(t_levels)
            if hue_idx < len(hue_order):
                _, dgp_name = hue_order[hue_idx].split(" | ")
                bar.set_hatch(hatch_map[dgp_name])
                bar.set_edgecolor("black")
                bar.set_linewidth(0.5)

    if g.legend:
        handles = getattr(g.legend, "legend_handles", g.legend.get_patches())
        for handle, text in zip(handles, g.legend.texts):
            label = text.get_text()
            if " | " in label:
                _, dgp_name = label.split(" | ")
                handle.set_hatch(hatch_map[dgp_name])
                handle.set_edgecolor("black")
                handle.set_linewidth(0.5)

    g.fig.suptitle(
        f"{spec.study_type.name} — {y_label} vs. {param_name}  {title}", y=1.05
    )
    g.set_axis_labels(param_name, y_label)
    plt.show()


def plot_results_convergence(
    df:        pd.DataFrame,
    spec:      ExperimentSpec,
    alpha:     float = 0.05,
    title:     str   = "",
):
    """
    Line chart showing how empirical coverage / power evolves as the
    swept parameter grows.
    Color = avar_model, marker shape = dgp_name.
    """
    metric, target_val, y_label = _metric_info(spec.study_type, alpha)
    param_name = spec.param_name

    sns.set_theme(style="whitegrid")

    g = sns.relplot(
        data=df, x=param_name, y=metric,
        hue="avar_model", style="dgp_name",
        dashes=False, markers=True, kind="line",
        marker="o", height=4.5, aspect=1.2, errorbar=None, alpha=0.7,
    )

    for ax in g.axes.flat:
        if target_val is not None:
            ax.axhline(target_val, color="black", linestyle="--",
                       linewidth=2, zorder=0)
        if param_name in ("T", "n_sim"):
            ax.set_xscale("log")

    g.fig.suptitle(
        f"{spec.study_type.name} — {y_label} vs. {param_name}  {title}", y=1.05
    )
    g.set_axis_labels(param_name, y_label)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Public API  (notebook-facing)
# ─────────────────────────────────────────────────────────────────────────────

def run_selected_configs(
    experiments:          dict[str, ExperimentSpec],
    selected_experiments: list[str],
    prefix:               str = "",
    out_dir:              Path | None = None,
) -> None:
    """Run and save results for a subset of experiments."""
    for exp_id in selected_experiments:
        print(f"\n{'='*60}")
        print(f"  Experiment: {exp_id}")
        print(f"{'='*60}")
        spec = experiments[exp_id]
        run_setups(spec, prefix=prefix + exp_id + "_", out_dir=out_dir)


def run_analysis(
    experiments:  dict[str, ExperimentSpec],
    experiment_name: str,
    alpha:        float       = 0.05,
    param_values: list | None = None,
    prefix:       str         = "",
    out_dir:      Path | None = None,
) -> None:
    """Load saved results for one experiment and produce both plot types."""
    spec = experiments[experiment_name]

    df = parse_setups(
        spec,
        prefix=prefix + experiment_name + "_",
        out_dir=out_dir,
        param_values=param_values,
    )

    th_label = " (Theo moments)" if spec.th_moments else ""
    plot_results_by_pair(df, spec, alpha=alpha, title=th_label)
    plot_results_convergence(df, spec, alpha=alpha, title=th_label)