"""Generate plot(s) for FitzHugh-Nagumo model (noisless observations) experiments."""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    add_experiment_grid_args,
    add_plot_args,
    check_experiment_dir_and_create_output_dir,
    set_matplotlib_style,
    load_summary_data,
    load_traces,
    plot_log_log_least_squares,
)

experiment_subdirectories = ("fhn_noiseless_chmc",)
parser = argparse.ArgumentParser(
    description="Generate plot(s) for FitzHugh-Nagumo model (noiseless obs) experiments",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
add_plot_args(parser, experiment_subdirectories)
add_experiment_grid_args(parser)
args = parser.parse_args()

check_experiment_dir_and_create_output_dir(args, experiment_subdirectories)

set_matplotlib_style()

errorbar_kwargs = {"capsize": 2, "elinewidth": 1, "markeredgewidth": 1, "markersize": 3}

param_names = ["β", "γ", "σ", "ϵ"]

func_names = [
    "constr",
    "jacob_constr_blocks",
    "chol_gram_blocks",
    "neg_log_dens",
    "grad_neg_log_dens",
    "log_det_sqrt_gram",
    "grad_log_det_sqrt_gram",
    "lu_jacob_product_blocks",
]

with open(
    os.path.join(
        args.experiment_dir, "fhn_noiseless_chmc", "fhn_noiseless_call_times.json"
    )
) as f:
    av_call_times_list = json.load(f)

av_call_times = {
    (
        call_times_set.pop("num_obs_per_subseq"),
        call_times_set.pop("num_steps_per_obs"),
        call_times_set.pop("num_obs"),
    ): call_times_set
    for call_times_set in av_call_times_list
}

splittings = ("standard", "gaussian")
exp_param_grids = {
    "R": [
        {
            "R": num_obs_per_subseq,
            "S": args.default_num_steps_per_obs,
            "T": args.default_num_obs,
        }
        for num_obs_per_subseq in args.num_obs_per_subseq_grid
    ],
    "S": [
        {
            "R": args.default_num_obs_per_subseq,
            "S": num_steps_per_obs,
            "T": args.default_num_obs,
        }
        for num_steps_per_obs in args.num_steps_per_obs_grid
    ],
    "T": [
        {
            "R": args.default_num_obs_per_subseq,
            "S": args.default_num_steps_per_obs,
            "T": num_obs,
        }
        for num_obs in args.num_obs_grid
    ],
}

func_names = [
    "constr",
    "jacob_constr_blocks",
    "chol_gram_blocks",
    "neg_log_dens",
    "grad_neg_log_dens",
    "log_det_sqrt_gram",
    "grad_log_det_sqrt_gram",
    "lu_jacob_product_blocks",
]


summary_data = {}
chain_stats_data = {}
for splitting in splittings:
    exp_dir_pattern = os.path.join(
        args.experiment_dir,
        "fhn_noiseless_chmc",
        f"R_{{R}}_S_{{S}}_T_{{T}}_H_1_{splitting}_splitting_*",
    )
    for grid_param, exp_params in exp_param_grids.items():
        summary_data[grid_param, splitting] = load_summary_data(
            exp_dir_pattern, exp_params, param_names, av_call_times=av_call_times
        )
        chain_stats_data[grid_param, splitting] = load_traces(
            exp_dir_pattern, exp_params, ["integration_n_step"], prefix="stats"
        )

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(6, 1.5), dpi=150)
for ax, plot_param in zip(axes, ("R", "S", "T")):
    trend_lines = []
    for i, splitting in enumerate(splittings):
        summary_data[plot_param, splitting]["total_integrator_steps"] = np.array(
            [
                stats["integration_n_step"].sum()
                for stats_set in chain_stats_data[plot_param, splitting].values()
                for stats in stats_set
            ]
        )
        summary_data[plot_param, splitting]["av_time_per_integrator_step"] = (
            summary_data[plot_param, splitting]["total_main_call_time"]
            / summary_data[plot_param, splitting]["total_integrator_steps"]
        )
        grouped_data = summary_data[plot_param, splitting].groupby([plot_param])
        series_mid = grouped_data.median()[f"av_time_per_integrator_step"]
        series_err_neg = series_mid - grouped_data.min()[f"av_time_per_integrator_step"]
        series_err_pos = grouped_data.max()[f"av_time_per_integrator_step"] - series_mid
        ax.errorbar(
            series_mid.index,
            series_mid.values,
            yerr=[series_err_neg.values, series_err_pos.values],
            fmt=".:" if plot_param == "R" else ".",
            **errorbar_kwargs,
        )
        if plot_param != "R":
            trend_lines.append(
                plot_log_log_least_squares(
                    ax,
                    series_mid.index,
                    series_mid.values,
                    f"$\\tt {plot_param}$",
                    "$\\hat{\\tau}_{\\rm step}$",
                    color=f"C{i}",
                )
            )
    if plot_param != "R":
        leg = ax.legend(*zip(*trend_lines), loc="best", labelspacing=0.1)
        for i, txt in enumerate(leg.get_texts()):
            txt.set_color(f"C{i}")
    ax.set(xscale="log", yscale="log")
    ax.set(
        xticks=series_mid.index,
        xticklabels=series_mid.index,
        xlabel=f"$\\mathtt{{{plot_param}}}$",
    )


axes[0].set_ylabel("$\\hat{\\tau}_{\\rm step} ~/~ {\\rm s}$")
R_vals = np.array(args.num_obs_per_subseq_grid)
axes[0].autoscale(False)
(handle2,) = axes[0].plot(R_vals, R_vals ** 2 * 0.000012, "k", ls="--", dashes=(1, 5))
axes[0].text(25, 0.003, "$\\hat{\\tau}_{\\rm step} \\propto {\\tt R^2}$", fontsize=8)
fig.tight_layout()

fig.savefig(
    os.path.join(args.output_dir, "fhn-noiseless-chmc-av-time-per-integrator-step.pdf"),
    pad_inches=0,
    bbox_inches="tight",
)

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(6, 1.5), dpi=150)
for ax, plot_param in zip(axes, ("R", "S", "T")):
    for i, splitting in enumerate(splittings):
        summary_data[plot_param, splitting]["av_newton_iters_per_step"] = summary_data[
            plot_param, splitting
        ]["total_main_constr_calls"] / (
            2 * summary_data[plot_param, splitting]["total_integrator_steps"]
        )
        grouped_data = summary_data[plot_param, splitting].groupby([plot_param])
        series_mid = grouped_data.median()[f"av_newton_iters_per_step"]
        series_err_neg = series_mid - grouped_data.min()[f"av_newton_iters_per_step"]
        series_err_pos = grouped_data.max()[f"av_newton_iters_per_step"] - series_mid
        ax.errorbar(
            series_mid.index,
            series_mid.values,
            yerr=[series_err_neg.values, series_err_pos.values],
            fmt=".",
            **errorbar_kwargs,
        )
    ax.set(xscale="log")
    ax.set(
        xticks=series_mid.index,
        xticklabels=series_mid.index,
        xlabel=f"$\\mathtt{{{plot_param}}}$",
        ylim=(0, 7),
    )
axes[0].set_ylabel("$\\bar{\\tt n}$")
fig.tight_layout()

fig.savefig(
    os.path.join(args.output_dir, "fhn-noiseless-chmc-av-number-newton-iterations.pdf"),
    pad_inches=0,
    bbox_inches="tight",
)

fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(6, 1.75), dpi=150)
for ax, param in zip(axes, ("β", "γ", "σ", "ϵ")):
    for splitting in splittings:
        grouped_data = summary_data["R", splitting].groupby(["R"])
        series_mid = grouped_data.median()[f"call_time_per_ess_bulk.{param}"]
        series_err_neg = (
            series_mid - grouped_data.min()[f"call_time_per_ess_bulk.{param}"]
        )
        series_err_pos = (
            grouped_data.max()[f"call_time_per_ess_bulk.{param}"] - series_mid
        )
        ax.errorbar(
            series_mid.index,
            series_mid.values,
            yerr=[series_err_neg.values, series_err_pos.values],
            fmt=".",
            **errorbar_kwargs,
        )
    ax.set(xscale="log", yscale="log")
    ax.set(
        title=f"${param}$",
        xlabel="$\\mathtt{R}$",
        xticks=args.num_obs_per_subseq_grid,
        xticklabels=args.num_obs_per_subseq_grid,
        ylim=(0.1, 10),
        yticks=[1e-1, 1e0, 1e1],
    )
axes[0].set_ylabel("$\\hat{\\tau}_{\\rm eff} ~/~ {\\rm s}$")
fig.tight_layout()
fig.savefig(
    os.path.join(
        args.output_dir,
        "fhn-noiseless-chmc-av-time-per-effective-sample-vs-num-obs-per-subseq.pdf",
    ),
    pad_inches=0,
    bbox_inches="tight",
)

fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(6, 1.75), dpi=150)
for ax, param in zip(axes, ("β", "γ", "σ", "ϵ")):
    trend_lines = []
    for i, splitting in enumerate(splittings):
        grouped_data = summary_data["S", splitting].groupby(["S"])
        series_mid = grouped_data.mean()[f"call_time_per_ess_bulk.{param}"]
        series_err_neg = (
            series_mid - grouped_data.min()[f"call_time_per_ess_bulk.{param}"]
        )
        series_err_pos = (
            grouped_data.max()[f"call_time_per_ess_bulk.{param}"] - series_mid
        )
        series_min = grouped_data.min()[f"call_time_per_ess_bulk.{param}"]
        series_max = grouped_data.max()[f"call_time_per_ess_bulk.{param}"]
        ax.errorbar(
            series_mid.index,
            series_mid.values,
            yerr=[series_err_neg.values, series_err_pos.values],
            fmt=".",
            color=f"C{i}",
            **errorbar_kwargs,
        )
        trend_lines.append(
            plot_log_log_least_squares(
                ax,
                series_mid.index,
                series_mid.values,
                "$S$",
                "$\\hat{\\tau}_{\\rm eff}$",
                color=f"C{i}",
            )
        )
    ax.legend(*zip(*trend_lines), loc="upper center", labelspacing=0.1)
    ax.set(xscale="log", yscale="log")
    ax.set(
        title=f"${param}$",
        xlabel="$\\mathtt{S}$",
        xticks=args.num_steps_per_obs_grid,
        xticklabels=args.num_steps_per_obs_grid,
        ylim=(0.1, 100),
        yticks=[1e-1, 1e0, 1e1, 1e2],
    )
axes[0].set_ylabel("$\\hat{\\tau}_{\\rm eff} ~/~ {\\rm s}$")
fig.tight_layout()

fig.savefig(
    os.path.join(
        args.output_dir,
        "fhn-noiseless-chmc-av-time-per-effective-sample-vs-num-steps-per-obs.pdf",
    ),
    pad_inches=0,
    bbox_inches="tight",
)

fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(6, 1.75), dpi=150)
for ax, param in zip(axes, ("β", "γ", "σ", "ϵ")):
    trend_lines = []
    for i, splitting in enumerate(splittings):
        grouped_data = summary_data["T", splitting].groupby(["T"])
        series_mid = grouped_data.median()[f"call_time_per_ess_bulk.{param}"]
        series_err_neg = (
            series_mid - grouped_data.min()[f"call_time_per_ess_bulk.{param}"]
        )
        series_err_pos = (
            grouped_data.max()[f"call_time_per_ess_bulk.{param}"] - series_mid
        )
        ax.errorbar(
            series_mid.index,
            series_mid.values,
            yerr=[series_err_neg.values, series_err_pos.values],
            fmt=".",
            color=f"C{i}",
            **errorbar_kwargs,
        )
        trend_lines.append(
            plot_log_log_least_squares(
                ax,
                series_mid.index,
                series_mid.values,
                "${\\tt S}$",
                "$\\hat{\\tau}_{\\rm eff}$",
                color=f"C{i}",
            )
        )
    ax.legend(*zip(*trend_lines), ncol=1, loc="best", labelspacing=0.1)
    ax.set(xscale="log", yscale="log")
    ax.set(
        title=f"${param}$",
        xlabel="$\\mathtt{T}$",
        xticks=args.num_obs_grid,
        xticklabels=args.num_obs_grid,
        ylim=(0.01, 10),
        yticks=[1e-2, 1e-1, 1e0, 1e1],
    )
axes[0].set_ylabel("$\\hat{\\tau}_{\\rm eff} ~/~ {\\rm s}$")
fig.tight_layout()

fig.savefig(
    os.path.join(
        args.output_dir,
        "fhn-noiseless-chmc-av-time-per-effective-sample-vs-num-obs.pdf",
    ),
    pad_inches=0,
    bbox_inches="tight",
)
