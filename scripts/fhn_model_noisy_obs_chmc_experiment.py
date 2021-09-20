"""Run experiment inferring FitzHugh-Nagumo model (noisy obs.) parameters with CHMC."""

import os
import argparse
import datetime
import sde
from sde.example_models import fhn
import numpy as np
from utils import (
    create_experiment_output_dir_and_save_args,
    setup_chmc_mici_objects,
    setup_jax,
    setup_logger,
    add_chmc_experiment_args,
    add_common_experiment_args,
    add_observation_noise_std_arg,
    get_call_counts,
    sample_chmc_chains,
    save_and_print_summary,
)

# Ensure Jax configured to use double-precision and to run on CPU

setup_jax()

# Process command line arguments defining experiment parameters

parser = argparse.ArgumentParser(
    description="Run FitzHugh-Nagumo model experiment (noisy observations, CHMC)"
)
add_common_experiment_args(
    parser,
    default_num_steps_per_obs=40,
    default_num_warm_up_iter=500,
    default_num_main_iter=2500,
)
add_chmc_experiment_args(parser, default_num_obs_per_subseq=5)
add_observation_noise_std_arg(parser, default_val=0.1)
args = parser.parse_args()

# Set up output directory

variable_σ = args.observation_noise_std < 0
timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
dir_name = (
    "σ_variable_" if variable_σ else f"σ_{args.observation_noise_std:.2g}_"
) + f"H_{args.num_inner_h2_step}_{args.splitting}_splitting_{timestamp}"
output_dir = os.path.join(args.output_root_dir, "fhn_noisy_chmc", dir_name)

create_experiment_output_dir_and_save_args(output_dir, args)

logger = setup_logger(output_dir)

# Define model specific constants

dim_u = fhn.dim_z + 1 if variable_σ else fhn.dim_z

# Specify observed data

data = np.load(
    os.path.join(os.path.dirname(__file__), "fhn_model_noisy_obs_simulated_data.npz")
)
y_seq = (data["y_seq_mean"] + abs(args.observation_noise_std) * data["n_seq"])[:, None]
num_obs = y_seq.shape[0]

# Set up Mici objects

rng = np.random.default_rng(args.seed)

system, integrator, sampler, adapters = setup_chmc_mici_objects(
    args,
    fhn,
    rng,
    data["obs_interval"],
    y_seq,
    dim_u,
    fhn.generate_σ if variable_σ else args.observation_noise_std,
)


def trace_func(state):
    q = state.pos
    u, v_0, _ = sde.mici_extensions.split(q, (dim_u, fhn.dim_v_0,))
    z = fhn.generate_z(u)
    x_0 = fhn.generate_x_0(z, v_0)
    call_counts = get_call_counts(system, state, True)
    traced_vars = {
        "x_0": x_0,
        "σ": z[0],
        "ϵ": z[1],
        "γ": z[2],
        "β": z[3],
        "hamiltonian": system.h(state),
        **call_counts,
    }
    if variable_σ:
        traced_vars["σ_y"] = fhn.generate_σ(u)
    return traced_vars


# Initialise chain states


def generate_x_obs_seq_init(rng):
    return np.concatenate((y_seq, rng.standard_normal(y_seq.shape) * 0.5), -1)


init_states = []
for c in range(args.num_chain):
    u = rng.standard_normal(dim_u)
    v_0 = rng.standard_normal(fhn.dim_v)
    state = sde.mici_extensions.find_initial_state_by_linear_interpolation(
        system, rng, generate_x_obs_seq_init, u=u, v_0=v_0
    )
    assert abs(system.constr(state)).max() < args.projection_solver_convergence_tol
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
summary_vars = ["σ", "ϵ", "γ", "β", "x_0", "hamiltonian"]
if variable_σ:
    summary_vars.append("σ_y")
summary_dict = save_and_print_summary(
    output_dir, traces, summary_vars, sampling_time, integrator
)
