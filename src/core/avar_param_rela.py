"""
avar_sharpe.py
==============
Plotting layer for Avar(√T · SR̂).  Consumes model classes from models.py.

Usage
-----
    python avar_sharpe.py                          # built-in demo
    from avar_sharpe import plot_avar_sharpe
    from models import REGISTRY, get_model

    fig = plot_avar_sharpe()                       # all registered models
    fig = plot_avar_sharpe(models=["iid_normal", "ar1_normal"])
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from models import REGISTRY, AvarModel, get_model


def _style_axis(ax):
    ax.set_facecolor("#161B27")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2E3A4E")
    ax.tick_params(colors="#889AAB", labelsize=7.5)
    ax.grid(True, color="#1F2A3A", lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def _autoscale_y(ax, pad=0.10):
    lines = [l for l in ax.get_lines() if np.asarray(l.get_ydata()).size > 0]
    if not lines:
        return
    all_y = np.concatenate([np.asarray(l.get_ydata()) for l in lines])
    all_y = all_y[np.isfinite(all_y)]
    if all_y.size == 0:
        return
    ymin = max(0.0, all_y.min() * (1 - pad))
    ymax = all_y.max() * (1 + pad)
    ax.set_ylim(ymin, ymax)


def _collect_param_panels(model_instances):
    """One panel per unique parameter (sr first, then model-specific)."""
    panels = {
        "sr": {
            "models":    model_instances,
            "parameter": model_instances[0]._sr_param,
        }
    }
    for m in model_instances:
        for pname, p in m.parameters.items():
            if pname not in panels:
                panels[pname] = {"models": [], "parameter": p}
            if m not in panels[pname]["models"]:
                panels[pname]["models"].append(m)
    return panels


def plot_avar_sharpe(
    models=None,
    baseline=None,
    ncols=3,
    figsize=(18, 11),
    title=r"Asymptotic Variance of the Sharpe Ratio Estimator  $V(\hat{\theta})$",
    ylim_pad=0.10,
    save_path=None,
    dpi=150,
):
    """
    Plot Avar(sqrt(T) * SR_hat) for each model vs every parameter.

    Parameters
    ----------
    models : list of short_name strings or AvarModel instances, optional
    baseline : dict, optional   override defaults for fixed params
    ncols, figsize, title, ylim_pad, save_path, dpi : standard options
    """
    instances = (
        list(REGISTRY.values()) if models is None
        else [get_model(m) if isinstance(m, str) else m for m in models]
    )

    # global defaults (later models override earlier ones on conflict)
    global_defaults = {}
    for m in instances:
        global_defaults.update(m.defaults())
    if baseline:
        global_defaults.update(baseline)

    panels    = _collect_param_panels(instances)
    n_panels  = len(panels)
    nrows     = (n_panels + ncols - 1) // ncols

    fig = plt.figure(figsize=figsize, facecolor="#0F1117")
    fig.suptitle(title, fontsize=15, fontweight="bold",
                 color="white", y=0.98, fontfamily="monospace")
    outer = gridspec.GridSpec(
        nrows, ncols, figure=fig,
        hspace=0.52, wspace=0.35,
        left=0.06, right=0.97, top=0.91, bottom=0.08,
    )
    axes = [fig.add_subplot(outer[i // ncols, i % ncols]) for i in range(n_panels)]

    for ax, (pname, panel) in zip(axes, panels.items()):
        _style_axis(ax)
        p = panel["parameter"]
        x = p.grid()

        for m in panel["models"]:
            fixed = {k: v for k, v in global_defaults.items() if k != pname}
            x_plot, V = m.sensitivity(pname, grid=x, **fixed)
            mask = np.isfinite(V)
            ax.plot(x_plot[mask], V[mask], label=m.name, **m.plot_style, zorder=3)

        ax.axhline(1.0, color="#555566", lw=0.8, ls="--", zorder=1, alpha=0.7)
        if pname != "sr":
            bv = global_defaults.get(pname, p.default)
            ax.axvline(bv, color="#778899", lw=0.9, ls=":", zorder=1, alpha=0.8)

        _autoscale_y(ax, ylim_pad)

        relevant_fixed = {}
        for m in panel["models"]:
            for k in m.all_parameters():
                if k != pname and k in global_defaults:
                    relevant_fixed[k] = global_defaults[k]
        fixed_str = ",  ".join(
            f"{k}={v}" for k, v in list(relevant_fixed.items())[:3]
        )
        ax.set_title(f"V  vs  {p.label}   [{fixed_str}]",
                     fontsize=9, color="#CCDDEE", pad=6, fontfamily="monospace")
        ax.set_xlabel(f"{p.label}  ${p.latex}$", fontsize=9,
                      color="#AABBCC", labelpad=4)
        ax.set_ylabel(r"$V(\hat{\theta})$", fontsize=9,
                      color="#AABBCC", labelpad=4)
        ax.legend(fontsize=7.5, loc="upper left",
                  framealpha=0.25, edgecolor="#334455",
                  labelcolor="white", facecolor="#1A1E2A")

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    handles = [Line2D([0], [0], label=m.name, **m.plot_style) for m in instances]
    fig.legend(handles=handles, loc="lower center", ncol=len(instances),
               fontsize=8, framealpha=0.3, edgecolor="#445566",
               labelcolor="white", facecolor="#151820",
               bbox_to_anchor=(0.5, 0.01), title="Model", title_fontsize=8)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Figure saved -> {save_path}")

    return fig


from config import PLOTS_DIR

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    fig = plot_avar_sharpe(save_path=PLOTS_DIR / "avar_sharpe_plot.png")
    print("\nFormula summary")
    print("-" * 64)
    for m in REGISTRY.values():
        print(f"  {m.name:<22}  V = {m.formula_latex}")