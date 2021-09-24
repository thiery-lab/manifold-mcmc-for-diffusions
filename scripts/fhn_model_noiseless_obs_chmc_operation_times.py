"""Compute FitzHugh-Nagumo model (noiseless) CHMC operation times."""

import os
import json
import argparse
import timeit
import time
import sde
import numpy as np
from statistics import median
from itertools import repeat, chain
from sde.example_models import fhn
from jax import jit, lax, numpy as jnp
from utils import setup_jax, add_experiment_grid_args


def block_until_ready_pytree(pytree):
    if isinstance(pytree, jnp.DeviceArray):
        return pytree.block_until_ready()
    elif isinstance(pytree, (float, int, complex, bool, type(None))):
        return pytree
    elif isinstance(pytree, (tuple, list)):
        return tuple(block_until_ready_pytree(subtree) for subtree in pytree)
    elif isinstance(pytree, dict):
        return {k: block_until_ready_pytree(v) for k, v in pytree.items()}
    else:
        raise ValueError(f"Unknown pytree node type {type(pytree)}")


def get_mapped_system_funcs(system):
    def map_jit_and_block_until_ready(func):
        def mapped_func(states):
            return lax.map(func, states)

        jitted_mapped_func = jit(mapped_func)

        def blocked_jitted_mapped_func(states):
            return block_until_ready_pytree(jitted_mapped_func(states))

        return blocked_jitted_mapped_func

    funcs = {
        "neg_log_dens": lambda state: system._neg_log_dens(state["q"]),
        "grad_neg_log_dens": lambda state: system._grad_neg_log_dens(state["q"]),
        "constr": lambda state: system._constr(state["q"], state["x_obs_seq"], 0),
        "jacob_constr_blocks": lambda state: system._jacob_constr_blocks(
            state["q"], state["x_obs_seq"], 0,
        ),
        "chol_gram_blocks": lambda state: system._chol_gram_blocks(
            *state["jacob_constr_blocks"]
        ),
        "lu_jacob_product_blocks": lambda state: system._lu_jacob_product_blocks(
            *state["jacob_constr_blocks"], *state["jacob_constr_blocks"]
        ),
        "log_det_sqrt_gram": lambda state: system._log_det_sqrt_gram_from_chol(
            *state["chol_gram_blocks"]
        ),
        "grad_log_det_sqrt_gram": lambda state: system._grad_log_det_sqrt_gram(
            state["q"], state["x_obs_seq"], 0
        )[1],
        "normal_space_component": lambda state: system._normal_space_component(
            state["q"], state["jacob_constr_blocks"], state["chol_gram_blocks"]
        ),
    }
    return {key: map_jit_and_block_until_ready(func) for key, func in funcs.items()}


# Ensure Jax configured to use double-precision and to run on CPU

setup_jax()

# Process command line arguments defining experiment parameters

parser = argparse.ArgumentParser(
    description="Compute Fitzhugh-Nagumo model (noiseless) CHMC operation times"
)
parser.add_argument(
    "--num-reps",
    type=int,
    default=10,
    help="Number independent repetitions to use when timing calls",
)
parser.add_argument(
    "--num-states",
    type=int,
    default=1000,
    help="Number of states to map each call across",
)
parser.add_argument(
    "--output-root-dir",
    default="experiments",
    help="Root directory to make experiment output subdirectory in",
)
add_experiment_grid_args(parser)
args = parser.parse_args()

output_dir = os.path.join(args.output_root_dir, "fhn_noiseless_chmc")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

rng = np.random.default_rng(20200710)
max_num_obs = max(args.num_obs_grid)
num_steps_per_obs_data = 10000
obs_interval = 0.2
δ = obs_interval / num_steps_per_obs_data
z = np.array([0.3, 0.1, 1.5, 0.8])  # [σ, ϵ, γ, β]
x_0 = np.array([-0.5, 0.2])
v_seq = rng.standard_normal((max_num_obs * num_steps_per_obs_data, fhn.dim_v))
y_seq = np.asarray(fhn.generate_y_seq(z, x_0, v_seq, δ, num_steps_per_obs_data))


all_call_times = []

for num_obs_per_subseq, num_steps_per_obs, num_obs in chain(
    zip(
        repeat(args.default_num_obs_per_subseq),
        repeat(args.default_num_steps_per_obs),
        args.num_obs_grid,
    ),
    zip(
        repeat(args.default_num_obs_per_subseq),
        args.num_obs_per_subseq_grid,
        repeat(args.default_num_obs),
    ),
    zip(
        args.num_obs_per_subseq_grid,
        repeat(args.default_num_steps_per_obs),
        repeat(args.default_num_obs),
    ),
):
    print(f"R = {num_obs_per_subseq} S = {num_steps_per_obs} T = {num_obs}")
    system = sde.mici_extensions.ConditionedDiffusionConstrainedSystem(
        obs_interval,
        num_steps_per_obs,
        num_obs_per_subseq,
        y_seq[:num_obs],
        fhn.dim_z,
        fhn.dim_x,
        fhn.dim_v,
        fhn.forward_func,
        fhn.generate_x_0,
        fhn.generate_z,
        fhn.obs_func,
        None,
        False,
        dim_v_0=fhn.dim_v_0,
    )
    mapped_funcs = get_mapped_system_funcs(system)
    dim_q = fhn.dim_z + fhn.dim_v_0 + num_steps_per_obs * num_obs * fhn.dim_v
    states = {
        "q": rng.standard_normal((args.num_states, dim_q)),
        "x_obs_seq": rng.standard_normal((args.num_states, num_obs, fhn.dim_x)),
    }
    call_times = {}
    for key, mapped_func in mapped_funcs.items():
        states[key] = mapped_func(states)
        times = [
            time_for_all_states / args.num_states
            for time_for_all_states in timeit.repeat(
                lambda: mapped_func(states), repeat=args.num_reps, number=1
            )
        ]
        print(
            f"{key:>25}: min={min(times):#.3g}s median={median(times):#.3g}s "
            f"max={max(times):#.3g}s"
        )
        call_times[key] = median(times)
    all_call_times.append(
        {
            "num_obs_per_subseq": num_obs_per_subseq,
            "num_steps_per_obs": num_steps_per_obs,
            "num_obs": num_obs,
            **call_times,
        }
    )

with open(os.path.join(output_dir, "fhn_noiseless_call_times.json"), "w") as f:
    json.dump(all_call_times, f)
