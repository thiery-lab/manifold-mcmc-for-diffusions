"""Shared helper functions for running experiments and generating plots."""

from pathlib import Path
import os
import time
import glob
import json
import logging
import pandas
import numpy as np
import matplotlib.pyplot as plt
import jax.config
import arviz
import sde
import mici
from mici.states import _cache_key_func


def setup_jax():
    # Ensure Jax configured to use double-precision and to run on CPU
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")


def add_common_experiment_args(
    parser, default_num_steps_per_obs, default_num_warm_up_iter, default_num_main_iter
):
    parser.add_argument(
        "--output-root-dir",
        default="experiments",
        help="Root directory to make experiment output subdirectory in",
    )
    parser.add_argument(
        "--num-steps-per-obs",
        type=int,
        default=default_num_steps_per_obs,
        help="Number of time steps per interobservation interval to use in inference",
    )
    parser.add_argument(
        "--splitting",
        choices=("standard", "gaussian"),
        default="standard",
        help="Hamiltonian splitting to use to define integrator step",
    )
    parser.add_argument(
        "--num-chain",
        type=int,
        default=4,
        help="Number of independent chains to sample",
    )
    parser.add_argument(
        "--num-warm-up-iter",
        type=int,
        default=default_num_warm_up_iter,
        help="Number of chain iterations in adaptive warm-up sampling stage",
    )
    parser.add_argument(
        "--num-main-iter",
        type=int,
        default=default_num_main_iter,
        help="Number of chain iterations in main sampling stage",
    )
    parser.add_argument(
        "--step-size-adaptation-target",
        type=float,
        default=0.8,
        help="Target acceptance statistic for step size adaptation",
    )
    parser.add_argument(
        "--step-size-reg-coefficient",
        type=float,
        default=0.1,
        help="Regularisation coefficient for step size adaptation",
    )
    parser.add_argument(
        "--seed", type=int, default=20200710, help="Seed for random number generator"
    )


def add_observation_noise_std_arg(parser, default_val=0.1):
    parser.add_argument(
        "--observation-noise-std",
        type=float,
        default=default_val,
        help=(
            "Standard deviation of observation noise. "
            "A negative value indicates instead treated as unknown variable to infer, "
            "with the absolute value indicating the true value used to generate the data."
        ),
    )


def add_hmc_experiment_args(parser):
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=20,
        help=(
            "Maximum depth of trajectory binary tree used by dynamic HMC transition. "
            "Maximum number of integrator steps per  iteration will be `2**max_tree_depth`."
        ),
    )
    parser.add_argument(
        "--metric-type",
        type=str,
        choices=("identity", "diagonal", "block", "dense"),
        default="identity",
        help=(
            "Type of metric (mass matrix) to use. If 'identity' (default) no adaptation "
            "is performed and the metric matrix representation is fixed to the identity. If"
            " 'diagonal' a diagonal metric matrix representation is adapted based on "
            "estimates of the variances of each state component. If 'block' a block "
            "diagonal metric matrix representation is used, with a dense upper-left block "
            "corresponding to the state components mapping to global parameters ('u' part "
            "state vector) and a diagonal lower-right block, with the diagonal block fixed "
            " to the identity with only the dense block adapted based on an estimate of the"
            "(marginal) posterior covariance matrix of the 'u' state vector component. If "
            "'dense' a dense metric matrix representation is used based on an estimate of "
            "the covariance matrix of the full state vector."
        ),
    )


def add_chmc_experiment_args(parser, default_num_obs_per_subseq):
    parser.add_argument(
        "--num-obs-per-subseq",
        type=int,
        default=default_num_obs_per_subseq,
        help="Average number of observations per blocked subsequence",
    )
    parser.add_argument(
        "--num-inner-h2-step",
        type=int,
        default=1,
        help="Number of inner h2 flow steps in each constrained integrator step",
    )
    parser.add_argument(
        "--projection-solver",
        choices=("newton", "quasi-newton"),
        default="newton",
        help="Non-linear iterative solver to use to solve projection onto manifold",
    )
    parser.add_argument(
        "--projection-solver-max-iters",
        type=int,
        default=50,
        help="Maximum number of iterations to try in projection solver",
    )
    parser.add_argument(
        "--projection-solver-constraint-tol",
        type=float,
        default=1e-9,
        help="Tolerance for norm of constraint function in projection solver",
    )
    parser.add_argument(
        "--projection-solver-position-tol",
        type=float,
        default=1e-8,
        help="Tolerance for norm of change in position in projection solver",
    )
    parser.add_argument(
        "--reverse-check-tol",
        type=float,
        default=2e-8,
        help="Tolerance for reversibility check on constrained integrator steps",
    )


def create_experiment_output_dir_and_save_args(output_dir, args):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


def setup_logger(output_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(os.path.join(output_dir, "info.log"))
    logger.addHandler(fh)
    return logger


def get_call_counts(system, state, include_chmc_specific_methods):
    methods = [system.neg_log_dens, system.grad_neg_log_dens]
    if include_chmc_specific_methods:
        methods += [
            system.constr,
            system.jacob_constr_blocks,
            system.chol_gram_blocks,
            system.log_det_sqrt_gram,
            system.grad_log_det_sqrt_gram,
        ]
    return {
        f"{method.__name__}_calls": state._call_counts[_cache_key_func(system, method)]
        for method in methods
    }


def setup_hmc_mici_objects(args, model, rng, obs_interval, y_seq, dim_u, generate_σ):
    (
        neg_log_dens,
        grad_neg_log_dens,
    ) = sde.mici_extensions.conditioned_diffusion_neg_log_dens_and_grad(
        obs_interval,
        args.num_steps_per_obs,
        y_seq,
        dim_u,
        model.dim_v_0,
        model.dim_v,
        model.forward_func,
        model.generate_x_0,
        model.generate_z,
        generate_σ,
        model.obs_func,
        args.splitting == "gaussian",
    )

    if args.splitting == "gaussian":
        system = mici.systems.GaussianEuclideanMetricSystem(
            neg_log_dens=neg_log_dens, grad_neg_log_dens=grad_neg_log_dens
        )
    else:
        system = mici.systems.EuclideanMetricSystem(
            neg_log_dens=neg_log_dens, grad_neg_log_dens=grad_neg_log_dens
        )

    integrator = mici.integrators.LeapfrogIntegrator(system)

    sampler = mici.samplers.DynamicMultinomialHMC(
        system, integrator, rng, max_tree_depth=args.max_tree_depth
    )

    adapters = [
        mici.adapters.DualAveragingStepSizeAdapter(
            adapt_stat_target=args.step_size_adaptation_target,
            log_step_size_reg_coefficient=args.step_size_reg_coefficient,
        )
    ]

    if args.metric_type == "diagonal":
        adapters.append(mici.adapters.OnlineVarianceMetricAdapter())
    elif args.metric_type == "dense":
        adapters.append(mici.adapters.OnlineCovarianceMetricAdapter())
    elif args.metric_type == "block":
        adapters.append(
            sde.mici_extensions.OnlineBlockDiagonalMetricAdapter(dim_u + model.dim_v_0)
        )
    return system, integrator, sampler, adapters


def setup_chmc_mici_objects(args, model, rng, obs_interval, y_seq, dim_u, generate_σ):
    system = sde.mici_extensions.ConditionedDiffusionConstrainedSystem(
        obs_interval,
        args.num_steps_per_obs,
        args.num_obs_per_subseq,
        y_seq,
        dim_u,
        model.dim_x,
        model.dim_v,
        model.forward_func,
        model.generate_x_0,
        model.generate_z,
        model.obs_func,
        generate_σ=generate_σ,
        use_gaussian_splitting=args.splitting == "gaussian",
        dim_v_0=model.dim_v_0,
    )

    projection_solver = (
        sde.mici_extensions.jitted_solve_projection_onto_manifold_newton
        if args.projection_solver == "newton"
        else sde.mici_extensions.jitted_solve_projection_onto_manifold_quasi_newton
    )

    project_solver_kwargs = {
        "constraint_tol": args.projection_solver_constraint_tol,
        "position_tol": args.projection_solver_position_tol,
        "max_iters": args.projection_solver_max_iters,
    }

    integrator = mici.integrators.ConstrainedLeapfrogIntegrator(
        system,
        n_inner_step=args.num_inner_h2_step,
        projection_solver=projection_solver,
        reverse_check_tol=args.reverse_check_tol,
        projection_solver_kwargs=project_solver_kwargs,
    )

    sampler = mici.samplers.MarkovChainMonteCarloMethod(
        rng,
        transitions={
            "momentum": mici.transitions.IndependentMomentumTransition(system),
            "integration": mici.transitions.MultinomialDynamicIntegrationTransition(
                system, integrator
            ),
            "switch_partition": sde.mici_extensions.SwitchPartitionTransition(system),
        },
    )

    step_size_adapter = mici.adapters.DualAveragingStepSizeAdapter(
        adapt_stat_target=args.step_size_adaptation_target,
        log_step_size_reg_coefficient=args.step_size_reg_coefficient,
    )

    return system, integrator, sampler, [step_size_adapter]


def sample_hmc_chains(
    sampler,
    adapters,
    init_states,
    trace_func,
    output_dir,
    num_warm_up_iter,
    num_main_iter,
):
    # Ignore NumPy floating point overflow warnings
    # Prevents warning messages being produced while progress bars are being printed
    np.seterr(over="ignore")
    start_time = time.time()
    final_states, traces, stats = sampler.sample_chains_with_adaptive_warm_up(
        num_warm_up_iter,
        num_main_iter,
        init_states,
        trace_funcs=[trace_func],
        adapters=adapters,
        memmap_enabled=True,
        memmap_path=output_dir,
        monitor_stats=["accept_stat", "n_step"],
    )
    sampling_time = time.time() - start_time
    return final_states, traces, stats, sampling_time


def sample_chmc_chains(
    sampler,
    adapters,
    init_states,
    trace_func,
    output_dir,
    num_warm_up_iter,
    num_main_iter,
):
    # Ignore NumPy floating point overflow warnings
    # Prevents warning messages being produced while progress bars are being printed
    np.seterr(over="ignore")
    start_time = time.time()
    final_states, traces, stats = sampler.sample_chains_with_adaptive_warm_up(
        num_warm_up_iter,
        num_main_iter,
        init_states,
        trace_funcs=[trace_func],
        adapters={"integration": adapters},
        memmap_enabled=True,
        memmap_path=output_dir,
        monitor_stats=[
            ("integration", "accept_stat"),
            ("integration", "n_step"),
        ],
    )
    sampling_time = time.time() - start_time
    return final_states, traces, stats, sampling_time


def save_and_print_summary(output_dir, traces, summary_vars, sampling_time, integrator):
    summary = arviz.summary(traces, var_names=summary_vars)
    summary_dict = summary.to_dict()
    summary_dict["total_sampling_time"] = sampling_time
    summary_dict["final_integrator_step_size"] = integrator.step_size
    for key, value in traces.items():
        if key[-6:] == "_calls":
            summary_dict["total_" + key] = sum(int(v[-1]) for v in value)
    with open(os.path.join(output_dir, "summary.json"), mode="w") as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=2)
    print(f"Integrator step size = {integrator.step_size:.2g}")
    print(f"Total sampling time = {sampling_time:.0f} seconds")
    print(summary)
    return summary_dict


def add_experiment_grid_args(parser):
    parser.add_argument(
        "--default-num-obs-per-subseq",
        type=int,
        default=5,
        help="Default value for num. obs. times per subsequence when not grid variable",
    )
    parser.add_argument(
        "--default-num-steps-per-obs",
        type=int,
        default=25,
        help="Default value for number of steps per obs. time when not grid variable",
    )
    parser.add_argument(
        "--default-num-obs",
        type=int,
        default=100,
        help="Default value for number of observation times when not grid variable",
    )
    parser.add_argument(
        "--num-obs-per-subseq-grid",
        type=int,
        nargs="+",
        default=[2, 5, 10, 20, 50, 100],
        help="Values for number of obs. times per subsequence when grid variable",
    )
    parser.add_argument(
        "--num-steps-per-obs-grid",
        type=int,
        nargs="+",
        default=[25, 50, 100, 200, 400],
        help="Values for number of steps per observation time when grid variable",
    )
    parser.add_argument(
        "--num-obs-grid",
        type=int,
        nargs="+",
        default=[25, 50, 100, 200, 400],
        help="Values for number of observation times when grid variable",
    )


def add_plot_args(parser, experiment_subdirectories):
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default="experiments",
        help=(
            "Root directory containing the experiment output subdirectories: "
            + (
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


def check_experiment_dir_and_create_output_dir(args, experiment_subdirectories):
    for subdir in experiment_subdirectories:
        if not os.path.exists(os.path.join(args.experiment_dir, subdir)):
            raise ValueError(
                f"Specified experiment directory ({args.experiment_dir}) does not "
                f"contain required subdirectory ({subdir})"
            )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


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
