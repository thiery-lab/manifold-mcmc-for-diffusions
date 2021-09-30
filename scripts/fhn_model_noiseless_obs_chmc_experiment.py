"""Run experiment inferring FitzHugh-Nagumo model (noiseless) parameters with CHMC."""

import os
import argparse
import datetime
import sde
from sde.example_models import fhn
import numpy as np
from utils import (
    create_experiment_output_dir_and_save_args,
    get_call_counts,
    setup_chmc_mici_objects,
    setup_jax,
    setup_logger,
    add_chmc_experiment_args,
    add_common_experiment_args,
    sample_chmc_chains,
    save_and_print_summary,
)

# Ensure Jax configured to use double-precision and to run on CPU

setup_jax()

# Process command line arguments defining experiment parameters

parser = argparse.ArgumentParser(
    description="Run Fitzhugh-Nagumo model experiment (noiseless observations, CHMC)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
add_common_experiment_args(
    parser,
    default_num_steps_per_obs=25,
    default_num_warm_up_iter=250,
    default_num_main_iter=1000,
)
add_chmc_experiment_args(parser, default_num_obs_per_subseq=5)
parser.add_argument(
    "--num-obs", type=int, default=100, help="Number of observation times"
)
parser.add_argument(
    "--true-parameters",
    type=float,
    nargs=4,
    default=[0.3, 0.1, 1.5, 0.8],
    help="Values of true parameters [σ, ϵ, γ, β] used to generate observed data",
)
parser.add_argument(
    "--true-initial-state",
    type=float,
    nargs=2,
    default=[-0.5, 0.2],
    help="Value true initial state [x_0[0], x_0[1]] used to generate observed data",
)
parser.add_argument(
    "--obs-interval",
    type=float,
    default=0.2,
    help="Value of time interval between each observation time",
)
parser.add_argument(
    "--num_steps_per_obs_data",
    type=int,
    default=10000,
    help="Number of time steps per interobservation interval to use to generate data",
)
args = parser.parse_args()

# Set up output directory

timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
dir_name = (
    f"R_{args.num_obs_per_subseq}_S_{args.num_steps_per_obs}_T_{args.num_obs}_"
    f"H_{args.num_inner_h2_step}_{args.splitting}_splitting_{timestamp}"
)
output_dir = os.path.join(args.output_root_dir, "fhn_noiseless_chmc", dir_name)

create_experiment_output_dir_and_save_args(output_dir, args)

logger = setup_logger(output_dir)

# Generate simulated observed data

rng = np.random.default_rng(args.seed)
δ = args.obs_interval / args.num_steps_per_obs_data
z_true = np.array(args.true_parameters)  # [σ, ϵ, γ, β]
x_0_true = np.array(args.true_initial_state)
v_seq_true = rng.standard_normal(
    (args.num_obs * args.num_steps_per_obs_data, fhn.dim_v)
)
y_seq = np.asarray(
    fhn.generate_y_seq(z_true, x_0_true, v_seq_true, δ, args.num_steps_per_obs_data)
)

# Set up Mici objects

system, integrator, sampler, adapters = setup_chmc_mici_objects(
    args, fhn, rng, args.obs_interval, y_seq, fhn.dim_z, None,
)


def trace_func(state):
    q = state.pos
    u, v_0, v_seq = np.split(q, (fhn.dim_z, fhn.dim_z + fhn.dim_v_0,))
    v_seq = v_seq.reshape((-1, fhn.dim_v))
    z = fhn.generate_z(u)
    x_0 = fhn.generate_x_0(z, v_0)
    call_counts = get_call_counts(system, state, True)
    return {
        "σ": z[0],
        "ϵ": z[1],
        "γ": z[2],
        "β": z[3],
        "x_0": x_0,
        "hamiltonian": system.h(state),
        **call_counts,
    }


# Initialise chain states


def generate_x_obs_seq_init(rng):
    return np.concatenate((y_seq, rng.standard_normal(y_seq.shape) * 0.5), -1)


init_states = []
for c in range(args.num_chain):
    u = rng.standard_normal(fhn.dim_z)
    v_0 = rng.standard_normal(fhn.dim_v)
    state = sde.mici_extensions.find_initial_state_by_linear_interpolation(
        system, rng, generate_x_obs_seq_init, u=u, v_0=v_0
    )
    init_states.append(state)


# Sample chains

final_states, traces, stats, sampling_time = sample_chmc_chains(
    sampler,
    adapters,
    init_states,
    trace_func,
    output_dir,
    args.num_warm_up_iter,
    args.num_main_iter,
)
summary_vars = ["σ", "ϵ", "γ", "β", "x_0"]
summary_dict = save_and_print_summary(
    output_dir, traces, summary_vars, sampling_time, integrator
)
