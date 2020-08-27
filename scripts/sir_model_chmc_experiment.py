import os
import logging
import argparse
import datetime
import json
import time
import mici
import sde
import symnum
import symnum.numpy as snp
import numpy as onp
import jax.config
import jax.numpy as jnp
import jax.lax as lax
import jax.api as api
import arviz

# Ensure Jax configured to use double-precision and to run on CPU

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Process command line arguments defining experiment parameters

parser = argparse.ArgumentParser(description="Run SIR diffusion model experiment")
parser.add_argument(
    "--output-root-dir",
    default="experiments",
    help="Root directory to make experiment output subdirectory in",
)
parser.add_argument(
    "--observation-noise-std",
    type=float,
    default=1.0,
    help=(
        "Standard deviation of observation noise. "
        "A negative value indicates instead treated as unknown variable to infer"
    ),
)
parser.add_argument(
    "--num-steps-per-obs",
    type=int,
    default=20,
    help="Number of time steps per interobservation interval to use in inference",
)
parser.add_argument(
    "--num-obs-per-subseq",
    type=int,
    default=14,
    help="Average number of observations per blocked subsequence",
)
parser.add_argument(
    "--splitting",
    choices=("standard", "gaussian"),
    default="standard",
    help=(
        "Hamiltonian splitting to use to define unconstrained integrator step"
        " used as basis for constrained integrator step"
    ),
)
parser.add_argument(
    "--num-inner-h2-step",
    type=int,
    default=1,
    help="Number of inner h2 flow steps in each constrained integrator step",
)
parser.add_argument(
    "--num-chain", type=int, default=4, help="Number of independent chains to sample"
)
parser.add_argument(
    "--num-warm-up-iter",
    type=int,
    default=500,
    help="Number of chain iterations in adaptive warm-up sampling stage",
)
parser.add_argument(
    "--num-main-iter",
    type=int,
    default=2500,
    help="Number of chain iterations in main sampling stage",
)
parser.add_argument(
    "--step-size-adaptation-target",
    type=float,
    default=0.8,
    help="Target acceptance statistic for step size adaptation",
)
parser.add_argument(
    "--seed", type=int, default=20200710, help="Seed for random number generator"
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
    "--projection-solver-convergence-tol",
    type=float,
    default=1e-9,
    help="Tolerance for norm of constraint function in projection solver",
)
parser.add_argument(
    "--projection-solver-position-tol",
    type=float,
    default=1e-7,
    help="Tolerance for norm of change in position in projection solver",
)
parser.add_argument(
    "--reverse-check-tol",
    type=float,
    default=2e-7,
    help="Tolerance for reversibility check on constrained integrator steps",
)
args = parser.parse_args()

# Set up output directory

variable_σ = args.observation_noise_std < 0
timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
dir_name = (
    "σ_variable_" if variable_σ else f"σ_{args.observation_noise_std:.2g}_"
) + f"H_{args.num_inner_h2_step}_{args.splitting}_splitting_{timestamp}"
output_dir = os.path.join(args.output_root_dir, "sir_chmc", dir_name)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, "args.json"), "w") as f:
    json.dump(vars(args), f, indent=2)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []
fh = logging.FileHandler(os.path.join(output_dir, "info.log"))
logger.addHandler(fh)


# Define model specific constants and functions

dim_x = 3
dim_y = 1
dim_w = 3
dim_z = 4
dim_u = (dim_z + 1) if variable_σ else dim_z
dim_v_0 = 1
dim_v = dim_w

N = 763  # total population size S + I + R


def drift_func(x, z):
    α = snp.exp(x[2])
    β, γ, ζ, ϵ = z
    return snp.array(
        [-α * x[0] * x[1] / N, α * x[0] * x[1] / N - β * x[1], γ * (ζ - x[2])]
    )


def diff_coeff(x, z):
    α = snp.exp(x[2])
    β, γ, ζ, ϵ = z
    return snp.array(
        [
            [snp.sqrt(α * x[0] * x[1] / N), 0, 0],
            [-snp.sqrt(α * x[0] * x[1] / N), snp.sqrt(β * x[1]), 0],
            [0, 0, ϵ],
        ]
    )


_forward_func = symnum.numpify_func(
    sde.integrators.euler_maruyama_step(
        *sde.integrators.transform_sde(
            lambda x: snp.array([snp.log(x[0]), snp.log(x[1]), x[2]]),
            lambda x: snp.array([snp.exp(x[0]), snp.exp(x[1]), x[2]]),
        )(drift_func, diff_coeff)
    ),
    (dim_z,),
    (dim_x,),
    (dim_v,),
    None,
    numpy_module=jnp,
)


def forward_func(z, x, v, δ):
    # Clip first two state components below at -500, in original domain corresponding to
    # exp(-500) ≈ 7 × 10^(-218) when updating state to prevent numerical NaN issues when
    # these state components tends to negative infinity. 500 was chosen as the cutoff to
    # avoid underflow / overflow as in double precision exp(-500) is non-zero and
    # exp(500) finite while for example exp(-1000) = 0 and exp(1000) = inf
    # We clip before and after _forward_func to avoid NaN gradients
    # https://github.com/tensorflow/probability/blob/master/discussion/where-nan.pdf
    x = x.at[:2].set(jnp.clip(x[:2], -500))
    x_ = _forward_func(z, x, v, δ)
    return jnp.array(
        [
            lax.select(x[0] > -500, x_[0], x[0]),
            lax.select(x[1] > -500, x_[1], x[1]),
            x_[2],
        ]
    )


def obs_func(x_seq):
    return jnp.exp(x_seq[..., 1:2])


def generate_z(u):
    return jnp.array(
        [
            jnp.exp(u[0]),  # β
            jnp.exp(u[1]),  # γ
            u[2],  # ζ
            jnp.exp(jnp.sqrt(0.75) * u[3] + 0.5 * u[1] - 3),  # ϵ
        ]
    )


def generate_x_0(z, v_0):
    return jnp.array([jnp.log(762.0), jnp.log(1.0), v_0[0]])


def generate_σ(u):
    return jnp.exp(u[dim_z])


# Specify observed data

y_seq_ref = onp.array(
    [3, 8, 28, 75, 221, 281, 255, 235, 190, 125, 70, 28, 12, 5], dtype=onp.float64
)[:, None]
num_obs = 14
num_steps = num_obs * args.num_steps_per_obs
obs_interval = 1.0

# Set up Mici objects

system = sde.mici_extensions.ConditionedDiffusionConstrainedSystem(
    obs_interval,
    args.num_steps_per_obs,
    args.num_obs_per_subseq,
    y_seq_ref,
    dim_u,
    dim_x,
    dim_v,
    forward_func,
    generate_x_0,
    generate_z,
    obs_func,
    generate_σ if variable_σ else args.observation_noise_std,
    use_gaussian_splitting=args.splitting == "gaussian",
    dim_v_0=dim_v_0,
)

projection_solver = (
    sde.mici_extensions.jitted_solve_projection_onto_manifold_newton
    if args.projection_solver == "newton"
    else sde.mici_extensions.jitted_solve_projection_onto_manifold_quasi_newton
)

project_solver_kwargs = {
    "convergence_tol": args.projection_solver_convergence_tol,
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

rng = onp.random.default_rng(args.seed)

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
    args.step_size_adaptation_target, log_step_size_reg_coefficient=0.1,
)


def trace_func(state):
    q = state.pos
    u, v_0, _ = sde.mici_extensions.split(q, (dim_u, dim_v_0,))
    z = generate_z(u)
    x_0 = generate_x_0(z, v_0)
    call_counts = {
        name.split(".")[-1] + "_calls": val
        for (name, _), val in state._call_counts.items()
    }
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
        traced_vars["σ"] = generate_σ(u)
    return traced_vars


# Initialise chain states

init_states = []
for c in range(args.num_chain):
    state = sde.mici_extensions.find_initial_state_by_gradient_descent_noisy_system(
        system, rng, max_num_tries=100, adam_step_size=1e-1, max_iters=5000
    )
    assert onp.allclose(system.constr(state), 0)
    init_states.append(state)


# Sample chains

# Ignore NumPy floating point overflow warnings
# Prevents warning messages being produced while progress bars are being printed

onp.seterr(over="ignore")

start_time = time.time()

final_states, traces, stats = sampler.sample_chains_with_adaptive_warm_up(
    args.num_warm_up_iter,
    args.num_main_iter,
    init_states,
    trace_funcs=[trace_func],
    adapters={"integration": [step_size_adapter]},
    memmap_enabled=True,
    memmap_path=output_dir,
    monitor_stats=[
        ("integration", "accept_stat"),
        ("integration", "convergence_error"),
        ("integration", "non_reversible_step"),
        ("integration", "n_step"),
    ],
)

sampling_time = time.time() - start_time
summary_vars = ["α₀", "β", "γ", "ζ", "ϵ", "hamiltonian"]
if variable_σ:
    summary_vars.append("σ")
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
