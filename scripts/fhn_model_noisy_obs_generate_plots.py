"""Generate plot(s) for FitzHugh-Nagumo model (noisy observations) experiments."""

import argparse
import os
import matplotlib.pyplot as plt
from utils import (
    add_plot_args,
    check_experiment_dir_and_create_output_dir,
    set_matplotlib_style,
    load_summary_data,
)


experiment_subdirectories = ("fhn_noisy_chmc", "fhn_noisy_hmc", "fhn_noisy_bridge")
parser = argparse.ArgumentParser(
    description="Generate plot(s) for FitzHugh-Nagumo model (noisy obs.) experiments",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
add_plot_args(parser, experiment_subdirectories)
parser.add_argument(
    "--obs-noise-std-grid",
    type=float,
    nargs="+",
    default=[0.01, 0.03162, 0.1, 0.3162],
    help="Grid of observation noise standard deviations to use",
)
args = parser.parse_args()

check_experiment_dir_and_create_output_dir(args, experiment_subdirectories)

set_matplotlib_style()

param_names = ["β", "γ", "σ", "ϵ"]

summary_data = {
    "Constrained HMC": load_summary_data(
        exp_dir_pattern=os.path.join(
            args.experiment_dir, "fhn_noisy_chmc", "σ_{σ:.2g}_H_1_standard_splitting_*"
        ),
        exp_param_sets=[{"σ": σ} for σ in args.obs_noise_std_grid],
        var_names=param_names,
    ),
    "Standard HMC": load_summary_data(
        exp_dir_pattern=os.path.join(
            args.experiment_dir,
            "fhn_noisy_hmc",
            "σ_{σ:.2g}_standard_splitting_diagonal_metric_*",
        ),
        exp_param_sets=[{"σ": σ} for σ in args.obs_noise_std_grid],
        var_names=param_names,
    ),
    "Guided proposals / RWM": load_summary_data(
        exp_dir_pattern=os.path.join(
            args.experiment_dir, "fhn_noisy_bridge", "σ_{σ:.2g}_*"
        ),
        exp_param_sets=[{"σ": σ} for σ in args.obs_noise_std_grid],
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
        xticks=args.obs_noise_std_grid,
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
