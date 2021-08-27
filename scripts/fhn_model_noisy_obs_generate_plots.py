"""Generate plot(s) for FitzHugh-Nagumo model (noisy observations) experiments."""

import os
import matplotlib.pyplot as plt
from utils import (
    get_args_and_create_output_dir,
    set_matplotlib_style,
    load_summary_data,
)


args = get_args_and_create_output_dir(
    "Generate plot(s) for FitzHugh-Nagumo model (noisy observations) experiments",
    ("fhn_noisy_chmc", "fhn_noisy_hmc", "fhn_noisy_bridge")
)

set_matplotlib_style()

σ_vals = [0.01, 0.03162, 0.1, 0.3162]
param_names = ["β", "γ", "σ", "ϵ"]

summary_data = {
    "Constrained HMC": load_summary_data(
        exp_dir_pattern=os.path.join(
            args.experiment_dir, "fhn_noisy_chmc", "σ_{σ:.2g}_H_1_standard_splitting_*"
        ),
        exp_param_sets=[{"σ": σ} for σ in σ_vals],
        var_names=param_names,
    ),
    "Standard HMC": load_summary_data(
        exp_dir_pattern=os.path.join(
            args.experiment_dir,
            "fhn_noisy_hmc",
            "σ_{σ:.2g}_standard_splitting_diagonal_metric_*",
        ),
        exp_param_sets=[{"σ": σ} for σ in σ_vals],
        var_names=param_names,
    ),
    "Guided proposals / RWM": load_summary_data(
        exp_dir_pattern=os.path.join(
            args.experiment_dir, "fhn_noisy_bridge", "σ_{σ:.2g}_*"
        ),
        exp_param_sets=[{"σ": σ} for σ in σ_vals],
        var_names=["s", "γ", "σ", "ϵ"],
        var_rename_map={"s": "β"},
    ),
}

fig, axes = plt.subplots(
    1, len(param_names), sharex=True, sharey=True, figsize=(6, 1.5)
)
for ax, param in zip(axes, param_names):
    for i, (label, data) in enumerate(summary_data.items()):
        y_col = f"time_per_ess_bulk.{param}"
        ax.plot(data["σ"], data[y_col], "o:", color=f"C{i}", label=label)
        ax.plot(
            data.loc[data[f"r_hat.{param}"] > 1.01, "σ"],
            data.loc[data[f"r_hat.{param}"] > 1.01, y_col],
            marker="x",
            linestyle="",
            markersize=5,
            color=f"C{i}",
        )
    ax.set(
        title=f"${param}$",
        xticks=σ_vals,
        xlabel="$\\sigma_y$",
        xscale="log",
        yscale="log",
    )
axes[0].legend(
    *ax.get_legend_handles_labels(),
    loc="upper center",
    ncol=4,
    bbox_to_anchor=(0.5, 1.15),
    bbox_transform=fig.transFigure,
)
axes[0].set_ylabel("$\\hat{\\tau}_{\\rm eff} ~/~ {\\rm s}$")

fig.savefig(
    os.path.join(
        args.output_dir,
        "fhn-noisy-hmc-chmc-bridge-av-time-per-effective-sample-vs-obs-noise-std.pdf",
    ),
    pad_inches=0,
    bbox_inches="tight",
)
