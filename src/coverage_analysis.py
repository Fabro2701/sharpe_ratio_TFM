
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from core import run_coverage
from config import RESULTS_DIR

def run_coverage_setups(param_name, param_values,
                        n=10_000, alpha=0.05, 
                        dgps=["iid_normal", "iid_t5"],
                        models=["iid_normal", "iid_student_t"],
                        th_moments=False):

    for i, param in enumerate(param_values):
        print(f"{i+1} / {len(param_values)}")
        test_args = [
                "--n_sim", str(n),
                "--alpha", str(alpha),
                "--seed", "42",
                "--out", str(RESULTS_DIR / f"coverage_an_{param_name}{param}_n{n}.csv"),
                "--quiet"
            ]
        if param_name=='T':
            test_args.extend(["--T", str(param), "--theta", "0.5"])
        elif param_name=='sr':
            test_args.extend(["--theta", str(param), "--T", "500"])
        else:
            raise ValueError(f"{param_name} not supp")

        test_args.extend(["--dgps"] + dgps)
        test_args.extend(["--models"] + models)
        if th_moments:
            test_args.append("--th_moments")

        run_coverage.main(test_args)


def parse_coverage_setups(param_name, param_values, 
                          dgps=None, models=None,
                          n=10_000,
                          ):
    
    # 1. Read and combine the data
    all_data = []
    for param in param_values:
        file_path = RESULTS_DIR / f"coverage_an_{param_name}{param}_n{n}.csv"
        
        # Read the CSV
        df_temp = pd.read_csv(file_path)
        df_temp[param_name] = param
        
        all_data.append(df_temp)

    # Combine into a single DataFrame
    df_results = pd.concat(all_data, ignore_index=True)
    if dgps:
        df_results = df_results[df_results["dgp_name"].isin(dgps)]
    if models:
        df_results = df_results[df_results["avar_model"].isin(models)]

    # Display the table focusing on the key metrics
    columns_to_show = [param_name, "dgp_name", "avar_model", "nominal_coverage", "coverage", "bias", "rmse"]
    print("=== Combined Results Table ===")
    print(df_results[columns_to_show].to_string(index=False))

    df_results["dgp_model_pair"] = df_results["dgp_name"] + " + " + df_results["avar_model"]
    return df_results



def plot_coverage_results_by_pair(df, param_name, target_val=0.95, ):
    """
    Plots coverage vs. param_name.
    Same avar_model gets the same base color.
    Different dgp_name gets a different hatch pattern.
    """
    sns.set_theme(style="whitegrid")
    df = df.copy()
    
    # 1. Create a combined category so seaborn plots them side-by-side
    df["model_dgp"] = df["avar_model"].astype(str) + " | " + df["dgp_name"].astype(str)
    
    # Extract unique values to assign consistent colors and hatches
    models = df["avar_model"].unique()
    dgps = df["dgp_name"].unique()
    t_levels = df[param_name].unique()
    
    # 2. Map colors (Model) and hatches (DGP)
    base_colors = sns.color_palette("tab10", len(models))
    color_map = dict(zip(models, base_colors))
    
    hatch_patterns = ['.', '/', 'O', '-', '*']
    hatch_map = dict(zip(dgps, hatch_patterns[:len(dgps)]))
    
    # 3. Build a custom palette mapping the combined name to the Model's color
    hue_order = []
    custom_palette = {}
    for m in models:
        for d in dgps:
            combo_name = f"{m} | {d}"
            hue_order.append(combo_name)
            custom_palette[combo_name] = color_map[m]
            
    # 4. Plot the bars
    g = sns.catplot(
        data=df,
        x=param_name,
        y="coverage",
        hue="model_dgp", 
        hue_order=hue_order,
        palette=custom_palette,
        kind="bar",
        height=5,
        aspect=1.5,
        errorbar=None
    )
    
    # 5. Loop through the generated bars and apply hatches
    for ax in g.axes.flat:
        ax.axhline(target_val, color='red', linestyle='--', linewidth=2, label=f'Target ({target_val})')
        
        # Seaborn draws bars in the exact order of 'hue_order', across all 'T' values
        for i, bar in enumerate(ax.patches):
            hue_idx = i // len(t_levels) # Figure out which hue group this bar belongs to
            if hue_idx < len(hue_order):
                combo_name = hue_order[hue_idx]
                _, dgp_name = combo_name.split(" | ")
                
                # Apply hatch and add a thin black edge so the hatch is visible
                bar.set_hatch(hatch_map[dgp_name])
                bar.set_edgecolor("black")
                bar.set_linewidth(0.5)
                
    # 6. Apply hatches to the legend so it matches the plot
    if g.legend:
        # Depending on the Matplotlib version, handles are accessed differently.
        handles = getattr(g.legend, "legend_handles", g.legend.get_patches())
        for handle, text in zip(handles, g.legend.texts):
            label = text.get_text()
            if " | " in label:
                _, dgp_name = label.split(" | ")
                handle.set_hatch(hatch_map[dgp_name])
                handle.set_edgecolor("black")
                handle.set_linewidth(0.5)

    # Formatting
    g.fig.suptitle(f"Empirical Coverage vs. {param_name} (Grouped by Model & DGP)", y=1.05)
    g.set_axis_labels(param_name, "Empirical Coverage")
    
    plt.show()



def plot_coverage_convergence(df, param_name, target_val=0.95):
    """
    Plots a line chart showing the convergence of empirical coverage 
    as trajectory length (T) increases.
    - Color = avar_model (aggregated across DGPs)
    - Line style = dgp_name
    """
    sns.set_theme(style="whitegrid")

    g = sns.relplot(
        data=df,
        x=param_name,
        y="coverage",
        hue="avar_model",
        style="dgp_name",       # Still drives marker variation
        dashes=False,           # All lines solid
        markers=True,           # Vary marker shape per dgp_name
        kind="line",
        marker="o",
        height=4.5,
        aspect=1.2,
        errorbar=None
    )

    for ax in g.axes.flat:
        ax.axhline(
            target_val,
            color='red',
            linestyle='--',
            linewidth=2,
            zorder=0
        )
        if param_name=='T':
            ax.set_xscale("log")

    g.fig.suptitle(f"Convergence of Empirical Coverage as {param_name} Increases", y=1.05)
    g.set_axis_labels(param_name, "Empirical Coverage")

    plt.show()

