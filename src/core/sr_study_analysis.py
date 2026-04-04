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
    label_param:  str  | None              = None   # e.g. "sr" or "bias_adj"
    label_values: list | None              = None   # e.g. [0.3, 0.5, 0.8]
    # ── convenience ──────────────────────────────────────────────────────────

    @property
    def dgps(self)   -> list[str]: return self.scenario[0]
    @property
    def models(self) -> list[str]: return self.scenario[1]

    @property
    def metric(self) -> str:
        return self.study_type.metric_name

    def file_stem(self, param_value, label_value=None) -> str:
        """Unique filename stem; encodes label value when present."""
        st_tag    = self.study_type.name.lower()
        th_tag    = "theo_" if self.th_moments else ""
        label_tag = (f"_{self.label_param}{label_value}"
                     if label_value is not None else "")
        return f"{st_tag}_{th_tag}{self.param_name}{param_value}{label_tag}"

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
    
    def _label_kwargs(self, label_value) -> dict:
        
        # Maps label_param aliases → actual run_study kwarg names
        LABEL_ALIASES = {
            "sr": "null_sr",
            # add more here as needed, e.g. "bias": "bias_adj"
        }

        kwarg_name = LABEL_ALIASES.get(self.label_param, self.label_param)
        return {kwarg_name: label_value}

    def _effective_label_values(self) -> list:
        """Returns [None] when no label dimension is configured."""
        return self.label_values if self.label_values is not None else [None]


# ─────────────────────────────────────────────────────────────────────────────
# DGP helper
# ─────────────────────────────────────────────────────────────────────────────

EXTRA_DGPS = {}
def set_extra_dgps(dgps):
    global EXTRA_DGPS
    EXTRA_DGPS = dgps

def _build_dgp_specs(names: list[str], **kwargs) -> list[DGPSpec]:
    #dgps = DGP_EXAMPLES | EXTRA_DGPS
    dgps = EXTRA_DGPS
    missing = [n for n in names if n not in dgps]
    if missing:
        raise ValueError(f"Unknown DGP name(s): {missing}. "
                         f"Available: {sorted(DGP_EXAMPLES)}")
    return [DGPSpec(dgps[n](**kwargs), n) for n in names] #collisions not avoided (kwargs)TODO


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
    spec:    ExperimentSpec,
    prefix:  str = "",
    out_dir: Path | None = None,
) -> None:
    out_dir     = Path(out_dir) if out_dir is not None else RESULTS_DIR
    avar_models = _build_models(spec.models)

    label_values = spec._effective_label_values()
    total = len(spec.param_values) * len(label_values)
    done  = 0

    for label_val in label_values:
        label_kwargs = spec._label_kwargs(label_val) if label_val is not None else {}
        dgp_specs = _build_dgp_specs(spec.dgps, **label_kwargs)

        for param in spec.param_values:
            done += 1
            label_str = (f"  {spec.label_param}={label_val}" if label_val is not None else "")
            print(f"  [{done}/{total}]  {spec.param_name}={param}{label_str}"
                  f"  ({spec.study_type.name})")

            # merge: label_kwargs can override param_kwargs (e.g. both touch target_sr)
            sim_kwargs   = {**spec._param_kwargs(param), **label_kwargs}

            if "null_sr" in label_kwargs:
                null_sr = label_kwargs["null_sr"]
                sim_kwargs["target_sr"] = spec._resolve_null_sr(null_sr)
            else:
                null_sr = spec._resolve_null_sr(sim_kwargs["target_sr"])
            sim_kwargs.pop("null_sr", None)

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

            # stamp the label column so parse_setups can reconstruct it
            if label_val is not None:
                results[spec.label_param] = label_val

            out_path = out_dir / f"{prefix}{spec.file_stem(param, label_val)}.csv"
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
    label_values: list | None = None,
    print_table:  bool = True,
) -> pd.DataFrame:
    out_dir      = Path(out_dir) if out_dir is not None else RESULTS_DIR
    param_values = param_values if param_values is not None else spec.param_values
    label_values = label_values if label_values is not None else spec._effective_label_values()
    metric       = spec.metric

    all_data = []
    for label_val in label_values:
        for param in param_values:
            path = out_dir / f"{prefix}{spec.file_stem(param, label_val)}.csv"
            df   = pd.read_csv(path)
            df[spec.param_name] = param
            if label_val is not None and spec.label_param not in df.columns:
                df[spec.label_param] = label_val   # back-fill if missing
            all_data.append(df)

    df_all = pd.concat(all_data, ignore_index=True)

    if dgps:   df_all = df_all[df_all["dgp_name"].isin(dgps)]
    if models: df_all = df_all[df_all["avar_model"].isin(models)]

    cols = [spec.param_name, "dgp_name", "avar_model", "nominal", metric, "bias", "rmse"]
    if spec.label_param and spec.label_param in df_all.columns:
        cols = [spec.label_param] + cols
    cols = [c for c in cols if c in df_all.columns]

    if print_table:
        print(f"=== {spec.study_type.name} — {spec.param_name} sweep ===")
        print(df_all[cols].to_string(index=False))

    df_all["dgp_model_pair"] = df_all["dgp_name"] + " + " + df_all["avar_model"]
    return df_all


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hue_column(df: pd.DataFrame, spec: ExperimentSpec) -> str:
    """
    When a label dimension exists and there's only one DGP+model pair,
    use the label column as the hue; otherwise fall back to dgp_model_pair.
    """
    if spec.label_param and spec.label_param in df.columns:
        return spec.label_param
    return "dgp_model_pair"

def _metric_info(study_type: StudyType, alpha: float) -> tuple[str, float, str]:
    """Return (column_name, target_line_value, y_label)."""
    if study_type.is_power:
        return "power", None, "Empirical Power"          # no fixed target for power
    else:
        return "coverage", 1.0 - alpha, "Empirical Coverage"


def plot_results_by_pair(
    df:    pd.DataFrame,
    spec:  ExperimentSpec,
    alpha: float = 0.05,
    title: str   = "",
):
    """
    Bar chart of coverage or power vs. the swept parameter.

    When spec.label_param is set (and there is a single DGP+model pair),
    color = label_param values.  Otherwise: color = avar_model, hatch = dgp_name.
    """
    metric, target_val, y_label = _metric_info(spec.study_type, alpha)
    param_name = spec.param_name
    hue_col    = _hue_column(df, spec)

    sns.set_theme(style="whitegrid")
    df = df.copy()
    df["dgp_model_pair"] = df["avar_model"].astype(str) + " | " + df["dgp_name"].astype(str)

    t_levels = sorted(df[param_name].unique())

    # ── simple path: label_param is the hue ──────────────────────────────────
    if hue_col == spec.label_param:
        hue_values = sorted(df[hue_col].unique(), key=str)
        palette    = dict(zip(hue_values, sns.color_palette("tab10", len(hue_values))))

        g = sns.catplot(
            data=df, x=param_name, y=metric,
            hue=hue_col, palette=palette,
            order=t_levels,
            kind="bar", height=5, aspect=1.5, errorbar=None,
        )

        for ax in g.axes.flat:
            if target_val is not None:
                ax.axhline(target_val, color="black", linestyle="--",
                           linewidth=2, label=f"Target ({target_val:.2f})")
            for bar in ax.patches:
                bar.set_edgecolor("black")
                bar.set_linewidth(0.5)

    # ── original path: color = model, hatch = dgp ────────────────────────────
    else:
        df["model_dgp"] = df["avar_model"].astype(str) + " | " + df["dgp_name"].astype(str)

        models = df["avar_model"].unique()
        dgps   = df["dgp_name"].unique()

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
            order=t_levels,
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


from itertools import cycle
def _zip_cycled(levels: list, values: list, default) -> dict:
    """Map levels to values, cycling if values is shorter than levels."""
    source = cycle(values) if values else cycle([default])
    return {level: val for level, val in zip(levels, source)}

def plot_results_convergence(
    df,
    spec,
    alpha:         float       = 0.05,
    title:         str         = "",
    reverse:       bool        = False,
    log:           bool        = False,
    yticks:        list | None = None,
    ylim:          tuple | None = None,
    markers:       list | None = None,   # per hue level, e.g. ['o', 's', '^']
    dashes:        list | None = None,   # per style level
    palette:       list | None = None,   # override default colour cycle
    linewidth:     float       = 2.0,
    markersize:    float       = 7.0,
    relplot_kwargs: dict | None = None,  # escape hatch for anything else
):
    metric, target_val, y_label = _metric_info(spec.study_type, alpha)
    param_name = spec.param_name
    hue_col    = _hue_column(df, spec)

    if reverse:
        df = df.copy()
        df[metric] = 1 - df[metric]

    sns.set_theme(style="whitegrid")

    # ── build marker / dash maps if explicit lists are provided ──────────────
    hue_levels = sorted(df[hue_col].unique(), key=str)

    # style encodes dgp_name only when hue is already doing dgp_name (classic multi-pair mode)
    # otherwise fold style onto hue so markers still vary per hue level
    style_col    = "dgp_name" if hue_col == "dgp_model_pair" else hue_col
    style_levels = sorted(df[style_col].unique(), key=str)

    marker_map  = _zip_cycled(style_levels, markers, "o")
    dash_map    = _zip_cycled(style_levels, dashes,  "")
    palette_map = _zip_cycled(hue_levels,  palette,  None) if palette else None
    extra = relplot_kwargs or {}

    g = sns.relplot(
        data=df, x=param_name, y=metric,
        hue=hue_col,
        style=style_col,
        markers=marker_map,
        dashes=dash_map if dash_map else False,
        palette=palette_map,
        kind="line",
        height=4.5, aspect=1.2, errorbar=None, alpha=0.7,
        **extra,
    )

    # ── per-axes decorations ─────────────────────────────────────────────────
    for ax in g.axes.flat:
        if target_val is not None:
            ax.axhline(
                1 - target_val if reverse else target_val,
                color="black", linestyle="--", linewidth=2, zorder=0,
            )
        if log:
            ax.set_xscale("log")
        if yticks is not None:
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(t) for t in yticks])
        if ylim is not None:
            ax.set_ylim(ylim)

        # uniform line / marker size
        for line in ax.get_lines():
            line.set_linewidth(linewidth)
            line.set_markersize(markersize)

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
    plot_mask:    list = [True, True, True], # [table, bar, line]
    line_plot_kargs = {}
) -> None:
    """Load saved results for one experiment and produce both plot types."""
    spec = experiments[experiment_name]

    df = parse_setups(
        spec,
        prefix=prefix + experiment_name + "_",
        out_dir=out_dir,
        param_values=param_values,
        print_table=plot_mask[0]
    )

    th_label = " (Theo moments)" if spec.th_moments else ""
    if plot_mask[1]:
        plot_results_by_pair(df, spec, alpha=alpha, title=th_label)
    if plot_mask[2]:
        plot_results_convergence(df, spec, alpha=alpha, title=th_label, **line_plot_kargs)