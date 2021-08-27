import argparse
from pathlib import Path
import os
import glob
import json
import pandas
import numpy as np
import matplotlib.pyplot as plt


def get_args_and_create_output_dir(description, experiment_subdirectories):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default="experiments",
        help=(
            "Root directory containing the experiment output subdirectories: " +
            (
                ", ".join(experiment_subdirectories[:-1])
                + " and "
                + experiment_subdirectories[-1]
            )
            if len(experiment_subdirectories) > 1
            else experiment_subdirectories[0]
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="figures",
        help="Directory to save figures to",
    )
    args = parser.parse_args()
    for subdir in experiment_subdirectories:
        if not os.path.exists(os.path.join(args.experiment_dir, subdir)):
            raise ValueError(
                f"Specified experiment directory ({args.experiment_dir}) does not "
                f"contain required subdirectory ({subdir})"
            )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    return args


def set_matplotlib_style():
    plt.style.use("seaborn-darkgrid")
    plt.style.use(
        {
            "mathtext.fontset": "cm",
            "font.family": "Latin Modern Roman",
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "axes.linewidth": 0.5,
            "lines.linewidth": 0.75,
            "grid.linewidth": 0.5,
            "axes.labelpad": 2.0,
            "figure.dpi": 150,
            "lines.markersize": 2,
            "animation.html": "html5",
        }
    )


def load_summary_data(
    exp_dir_pattern, exp_param_sets, var_names, var_rename_map=None, av_call_times=None
):
    summary_data = []
    for exp_params in exp_param_sets:
        exp_dirs = glob.glob(exp_dir_pattern.format(**exp_params))
        for exp_dir in exp_dirs:
            with open(os.path.join(exp_dir, "summary.json")) as f:
                summary = json.load(f)
            if var_rename_map is not None:
                for value in summary.values():
                    if isinstance(value, dict):
                        for old_name, new_name in var_rename_map.items():
                            value[new_name] = value.pop(old_name)
            if av_call_times is not None:
                exp_key = tuple(exp_params.values())
                total_call_time = 0
                total_main_call_time = 0
                for func_name, av_time in av_call_times[exp_key].items():
                    n_calls = summary.get(f"total_{func_name}_calls", 0)
                    summary[f"total_{func_name}_call_time"] = n_calls * av_time
                    total_call_time += n_calls * av_time
                    call_trace_files = glob.glob(
                        os.path.join(exp_dir, f"trace_*_{func_name}_calls.npy")
                    )
                    if len(call_trace_files) > 0:
                        call_traces = np.stack(
                            [np.load(trace_file) for trace_file in call_trace_files]
                        )
                        # Calls in main chain phase only i.e. excluding warm-up phase
                        n_main_calls = (call_traces[:, -1] - call_traces[:, 0]).sum()
                    else:
                        n_main_calls = 0
                    summary[f"total_main_{func_name}_calls"] = n_main_calls
                    summary[f"total_main_{func_name}_call_time"] = (
                        n_main_calls * av_time
                    )
                    total_main_call_time += n_main_calls * av_time
                summary["total_call_time"] = total_call_time
                summary["total_main_call_time"] = total_main_call_time
            summary.update(exp_params)
            summary_data.append(summary)
    summary_data_df = pandas.json_normalize(summary_data)
    for name in var_names:
        if var_rename_map is not None and name in var_rename_map:
            name = var_rename_map[name]
        summary_data_df[f"time_per_ess_bulk.{name}"] = (
            summary_data_df["total_sampling_time"] / summary_data_df[f"ess_bulk.{name}"]
        )
        if av_call_times is not None:
            summary_data_df[f"call_time_per_ess_bulk.{name}"] = (
                summary_data_df["total_call_time"] / summary_data_df[f"ess_bulk.{name}"]
            )
            summary_data_df[f"main_call_time_per_ess_bulk.{name}"] = (
                summary_data_df["total_main_call_time"]
                / summary_data_df[f"ess_bulk.{name}"]
            )
    return summary_data_df


def load_traces(
    exp_dir_pattern, exp_param_sets, var_names, var_rename_map=None, prefix="trace"
):
    traces = {}
    for exp_params in exp_param_sets:
        exp_dirs = glob.glob(exp_dir_pattern.format(**exp_params))
        exp_key = tuple(exp_params.values())
        traces[exp_key] = []
        for exp_dir in exp_dirs:
            trace_set = {}
            for name in var_names:
                trace_file_paths = glob.glob(
                    os.path.join(exp_dir, f"{prefix}_*_{name}.npy")
                )
                if len(trace_file_paths) > 0:
                    if var_rename_map is not None and name in var_rename_map:
                        name = var_rename_map[name]
                    trace_set[name] = np.stack(
                        [
                            np.load(trace_file_path)
                            for trace_file_path in trace_file_paths
                        ],
                        0,
                    )
            traces[exp_key].append(trace_set)
    return traces


def plot_log_log_least_squares(ax, x, y, x_label, y_label, linestyle="--", color="C0"):
    log_x, log_y = np.log(x), np.log(y)
    gradient, intercept = np.polyfit(log_x, log_y, 1)
    (line,) = ax.plot(
        x,
        np.exp(np.poly1d((gradient, intercept))(log_x)),
        linestyle=linestyle,
        color=color,
        lw=0.75,
    )
    label = f'${y_label.strip("$")} \\propto {x_label.strip("$")}^{{{gradient:.2f}}}$'
    return line, label
