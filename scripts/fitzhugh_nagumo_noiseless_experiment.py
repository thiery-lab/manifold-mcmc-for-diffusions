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

parser = argparse.ArgumentParser(
    description="Run Fitzhugh-Nagumo model experiment (noiseless observations, CHMC)"
)
parser.add_argument(
    "--output-root-dir",
    default="experiments",
    help="Root directory to make experiment output subdirectory in",
)
parser.add_argument(
    "--num-steps-per-obs",
    type=int,
    default=25,
    help="Number of time steps per interobservation interval to use in inference",
)
parser.add_argument(
    "--num-obs", type=int, default=100, help="Number of observation times"
)
parser.add_argument(
    "--num-obs-per-subseq",
    type=int,
    default=5,
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
    "--num-warm_up-iter",
    type=int,
    default=250,
    help="Number of chain iterations in adaptive warm-up sampling stage",
)
parser.add_argument(
    "--num-main-iter",
    type=int,
    default=1000,
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
output_dir = os.path.join(
    args.output_root_dir, "fitzhugh-nagumo_noiseless_chmc", dir_name
)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, "args.json"), "w") as f:
    json.dump(vars(args), f, indent=2)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []
logger.addHandler(logging.FileHandler(os.path.join(output_dir, "info.log")))


# Define model specific constants and functions

dim_x = 2
dim_w = 1
dim_z = 4
dim_u = dim_z
dim_v_0 = dim_x
dim_v = 2 * dim_w


def drift_func(x, z):
    σ, ε, γ, β = z
    return snp.array([(x[0] - x[0] ** 3 - x[1]) / ε, γ * x[0] - x[1] + β])


def diff_coeff(x, z):
    σ, ε, γ, β = z
    return snp.array([[0], [σ]])


forward_func = symnum.numpify_func(
    sde.integrators.strong_order_1p5_step(drift_func, diff_coeff),
    (dim_z,),
    (dim_x,),
    (dim_v,),
    None,
    numpy_module=jnp,
)


def obs_func(x_seq):
    return x_seq[..., 0:1]


def generate_z(u):
    # [σ, ϵ, γ, β]
    return jnp.array([jnp.exp(u[0]), jnp.exp(u[1]), jnp.exp(u[2]), u[3]])


def generate_x_0(z, v_0):
    return v_0 - jnp.array([0, z[3]])


# Generate simulated observed data

rng = onp.random.default_rng(args.seed)
δ = args.obs_interval / args.num_steps_per_obs_data
z_true = onp.array(args.true_parameters)  # [σ, ϵ, γ, β]
x_0_true = onp.array(args.true_initial_state)
v_seq_true = rng.standard_normal((args.num_obs * args.num_steps_per_obs_data, dim_v))


def generate_from_model(z, x_0, v_seq, δ, num_steps_per_obs):
    def step_func(x, v):
        x_n = forward_func(z, x, v, δ)
        return x_n, x_n

    _, x_seq = lax.scan(step_func, x_0, v_seq)
    return obs_func(x_seq[num_steps_per_obs - 1 :: num_steps_per_obs])


y_seq = onp.asarray(
    generate_from_model(z_true, x_0_true, v_seq_true, δ, args.num_steps_per_obs_data)
)

# Set up Mici objects

system = sde.mici_extensions.ConditionedDiffusionConstrainedSystem(
    args.obs_interval,
    args.num_steps_per_obs,
    args.num_obs_per_subseq,
    y_seq,
    dim_u,
    dim_x,
    dim_v,
    forward_func,
    generate_x_0,
    generate_z,
    obs_func,
    use_gaussian_splitting=args.splitting == "gaussian",
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


def trace_func(state):
    q = state.pos
    u, v_0, v_seq = onp.split(q, (dim_z, dim_z + dim_v_0,))
    v_seq = v_seq.reshape((-1, dim_v))
    z = generate_z(u)
    x_0 = generate_x_0(z, v_0)
    call_counts = {
        name.split(".")[-1] + "_calls": val
        for (name, _), val in state._call_counts.items()
    }
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
    return onp.concatenate((y_seq, rng.standard_normal(y_seq.shape) * 0.5), -1)


init_states = []
for c in range(args.num_chain):
    u = rng.standard_normal(dim_u)
    v_0 = rng.standard_normal(dim_v)
    state = sde.mici_extensions.find_initial_state_by_linear_interpolation(
        system, rng, generate_x_obs_seq_init, u=u, v_0=v_0
    )
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
summary = arviz.summary(traces, var_names=["σ", "ϵ", "γ", "β", "x_0"])
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
