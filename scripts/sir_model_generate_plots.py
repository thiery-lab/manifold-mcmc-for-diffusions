"""Generate plot(s) for SIR model experiments."""

import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import arviz
from utils import (
    get_args_and_create_output_dir,
    set_matplotlib_style,
    load_summary_data,
    load_traces,
)


args = get_args_and_create_output_dir(
    "Generate plot(s) for SIR model experiments", ("sir_chmc", "sir_hmc")
)

set_matplotlib_style()

σ_vals = ["variable"]
orig_param_names = ["α₀", "β", "γ", "ζ", "ϵ", "σ"]
param_rename_map = {
    "α₀": "c(0)",
    "β": "\\gamma",
    "γ": "\\alpha",
    "ζ": "\\beta",
    "ϵ": "\\sigma",
    "σ": "\\sigma_y",
}
param_names = [param_rename_map[param] for param in orig_param_names]

summary_data = {
    "Constrained HMC": load_summary_data(
        exp_dir_pattern=os.path.join(
            args.experiment_dir, "sir_chmc", "σ_{σ}_H_1_standard_splitting_*"
        ),
        exp_param_sets=[{"σ": σ} for σ in σ_vals],
        var_names=orig_param_names,
        var_rename_map=param_rename_map,
    ),
    "Standard HMC": load_summary_data(
        exp_dir_pattern=os.path.join(
            args.experiment_dir, "sir_hmc", "σ_{σ}_standard_splitting_diagonal_metric_*"
        ),
        exp_param_sets=[{"σ": σ} for σ in σ_vals],
        var_names=orig_param_names,
        var_rename_map=param_rename_map,
    ),
}

trace_data = {
    "Constrained HMC": load_traces(
        exp_dir_pattern=os.path.join(
            args.experiment_dir, "sir_chmc", "σ_{σ}_H_1_standard_splitting_*"
        ),
        exp_param_sets=[{"σ": σ} for σ in σ_vals],
        var_names=orig_param_names,
        var_rename_map=param_rename_map,
    )["variable",][0],
    "Standard HMC": load_traces(
        exp_dir_pattern=os.path.join(
            args.experiment_dir, "sir_hmc", "σ_{σ}_standard_splitting_diagonal_metric_*"
        ),
        exp_param_sets=[{"σ": σ} for σ in σ_vals],
        var_names=orig_param_names,
        var_rename_map=param_rename_map,
    )["variable",][0],
}

fig, ax = plt.subplots(figsize=(2, 2.5))
width = 0.35
x = np.arange(len(param_names))
for i, (label, data) in enumerate(summary_data.items()):
    ax.bar(
        x - width / 2 + i * width,
        data.loc[0, [f"time_per_ess_bulk.{param}" for param in param_names]],
        width=width,
        label=label,
    )
ax.set_xticks(x)
ax.set_xticklabels(
    [f"${param}$" for param in param_names],
    fontsize=9,
    fontdict={"verticalalignment": "baseline"},
)
for tick in ax.xaxis.get_major_ticks():
    tick.set_pad(10)
ax.set_ylabel("$\\hat{\\tau}_{\\rm eff} ~/~ {\\rm s}$")
ax.legend()
fig.tight_layout()

fig.savefig(
    os.path.join(args.output_dir, "sir-hmc-chmc-time-per-effective-sample-bar.pdf"),
    pad_inches=0,
    bbox_inches="tight",
)

param_ranges = {
    "c(0)": (-4, 3),
    "\\gamma": (0.4, 0.6),
    "\\alpha": (0, 15),
    "\\beta": (0, 1.5),
    "\\sigma": (0, 1.0),
    "\\sigma_y": (0, 25),
}

fig, axes = plt.subplots(2, len(param_names) // 2, figsize=(4, 2.5))
for ax, param_name in zip(axes.flatten(), param_names):
    ax.set_xlabel(f"${param_name}$")
    ax.set_yticks([])
    for i, (label, traces) in enumerate(trace_data.items()):
        ax.hist(
            np.concatenate(traces[param_name], 0),
            range=param_ranges[param_name],
            bins=50,
            density=True,
            alpha=0.5,
            color=f"C{i}",
            label=label,
        )
legend = fig.legend(
    *ax.get_legend_handles_labels(),
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(0.5, 1.05),
    bbox_transform=fig.transFigure,
)
fig.tight_layout(pad=0.5)

fig.savefig(
    os.path.join(args.output_dir, "sir-hmc-chmc-histograms.pdf"),
    pad_inches=0,
    bbox_inches="tight",
)

fig, axes = plt.subplots(5, 5, figsize=(6, 6), dpi=150)
log_transform_params = {"\\alpha", "\\sigma", "\\sigma_y"}
for i, (label, traces) in enumerate(trace_data.items()):
    renamed_traces = {
        f"$\\log\\,{param}$"
        if param in log_transform_params
        else f"${param}$": np.log(trace)
        if param in log_transform_params
        else trace
        for param, trace in traces.items()
    }
    _ = arviz.plot_pair(
        renamed_traces,
        ax=axes,
        plot_kwargs={"ms": 0.5, "color": f"C{i}", "label": label},
    )
handles, labels = axes[0, 0].get_legend_handles_labels()
handles = [copy.copy(h) for h in handles]
for handle in handles:
    handle.set_markersize(10)
legend = axes[0, 0].legend(
    handles,
    labels,
    loc="center",
    bbox_to_anchor=(0.35, 0.95),
    bbox_transform=fig.transFigure,
)
legend.set_in_layout(False)
fig.tight_layout(pad=0.5, h_pad=0.2, w_pad=0.5)

fig.savefig(
    os.path.join(args.output_dir, "sir-hmc-chmc-pair-plots.pdf"),
    pad_inches=0,
    bbox_inches="tight",
)
