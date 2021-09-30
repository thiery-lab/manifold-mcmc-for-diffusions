"""Run experiment inferring SIR model parameters with HMC."""

import os
import argparse
import datetime
import sde
from sde.example_models import sir
import numpy as np
from utils import (
    create_experiment_output_dir_and_save_args,
    get_call_counts,
    setup_hmc_mici_objects,
    setup_jax,
    setup_logger,
    add_hmc_experiment_args,
    add_common_experiment_args,
    add_observation_noise_std_arg,
    sample_hmc_chains,
    save_and_print_summary,
)

# Ensure Jax configured to use double-precision and to run on CPU

setup_jax()

# Process command line arguments defining experiment parameters

parser = argparse.ArgumentParser(
    description="Run SIR, model experiment (HMC)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
add_common_experiment_args(
    parser,
    default_num_steps_per_obs=20,
    default_num_warm_up_iter=500,
    default_num_main_iter=2500,
)
add_hmc_experiment_args(parser)
add_observation_noise_std_arg(parser, default_val=1.0)
args = parser.parse_args()

# Set up output directory

variable_σ = args.observation_noise_std < 0
timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
dir_name = (
    "σ_variable_" if variable_σ else f"σ_{args.observation_noise_std:.2g}_"
) + f"{args.splitting}_splitting_{args.metric_type}_metric_{timestamp}"
output_dir = os.path.join(args.output_root_dir, "sir_hmc", dir_name)

create_experiment_output_dir_and_save_args(output_dir, args)

logger = setup_logger(output_dir)


# Define model specific constants

dim_u = (sir.dim_z + 1) if variable_σ else sir.dim_z


# Specify observed data

data = np.load(
    os.path.join(os.path.dirname(__file__), "sir_model_boarding_school_data.npz")
)
num_obs = data["y_seq"].shape[0]

# Set up Mici objects

rng = np.random.default_rng(args.seed)

system, integrator, sampler, adapters = setup_hmc_mici_objects(
    args,
    sir,
    rng,
    data["obs_interval"],
    data["y_seq"],
    dim_u,
    sir.generate_σ_y if variable_σ else args.observation_noise_std,
)


def trace_func(state):
    q = state.pos
    u, v_0, _ = sde.mici_extensions.split(q, (dim_u, sir.dim_v_0,))
    z = sir.generate_z(u)
    x_0 = sir.generate_x_0(z, v_0)
    call_counts = get_call_counts(system, state, False)
    traced_vars = {
        "α₀": x_0[-1],
        "β": z[0],
        "γ": z[1],
        "ζ": z[2],
        "ϵ": z[3],
        "hamiltonian": system.h(state),
        **call_counts,
    }
    if variable_σ:
        traced_vars["σ"] = sir.generate_σ_y(u)
    return traced_vars


# Initialise chain states

init_states = []
for c in range(args.num_chain):
    state = sde.mici_extensions.find_initial_state_by_gradient_descent_noisy_system(
        system,
        rng,
        max_num_tries=100,
        adam_step_size=1e-1,
        max_iters=5000,
        dim_u=dim_u,
        dim_v_0=sir.dim_v_0,
        dim_v=sir.dim_v,
        num_obs=num_obs,
        num_steps_per_obs=args.num_steps_per_obs,
        generate_z=sir.generate_z,
        generate_x_0=sir.generate_x_0,
        generate_σ=sir.generate_σ_y
        if variable_σ
        else lambda u: args.observation_noise_std,
        forward_func=sir.forward_func,
        obs_func=sir.obs_func,
        y_seq=data["y_seq"],
        δ=data["obs_interval"] / args.num_steps_per_obs,
    )
    init_states.append(state)


# Sample chains

final_states, traces, stats, sampling_time = sample_hmc_chains(
    sampler,
    adapters,
    init_states,
    trace_func,
    output_dir,
    args.num_warm_up_iter,
    args.num_main_iter,
)
summary_vars = ["α₀", "β", "γ", "ζ", "ϵ", "hamiltonian"]
if variable_σ:
    summary_vars.append("σ")
summary_dict = save_and_print_summary(
    output_dir, traces, summary_vars, sampling_time, integrator
)
