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
    description="Run FitzHugh-Nagumo model experiment (noisy observations, CHMC)"
)
parser.add_argument(
    "--output-root-dir",
    default="experiments",
    help="Root directory to make experiment output subdirectory in",
)
parser.add_argument(
    "--observation-noise-std",
    type=float,
    default=0.1,
    help=(
        "Standard deviation of observation noise. "
        "A negative value indicates instead treated as unknown variable to infer, "
        "with the absolute value indicating the true value used to generate the data."
    ),
)
parser.add_argument(
    "--num-steps-per-obs",
    type=int,
    default=40,
    help="Number of time steps per interobservation interval to use in inference",
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
    "--step-size-reg-coefficient",
    type=float,
    default=0.1,
    help="Regularisation coefficient for step size adaptation"
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
output_dir = os.path.join(args.output_root_dir, "fhn_noisy_chmc", dir_name)

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

dim_x = 2
dim_w = 1
dim_z = 4
dim_u = dim_z + 1 if variable_σ else dim_z
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
    # z = [σ, ε, γ, β]
    return jnp.array([jnp.exp(u[0]), jnp.exp(u[1]), jnp.exp(u[2]), u[3],])


def generate_x_0(z, v_0):
    return v_0 - jnp.array([0, z[3]])


def generate_σ(u):
    return jnp.exp(u[dim_z])


# Specify observed data

# fmt: off
y_seq_mean = onp.array(
    [
        -0.99525762, -0.89765401, -0.70217404, -0.38511491, +0.69229973,
        +0.98448125, +0.74801663, +0.18588055, -1.16875259, -1.08789095,
        -1.07112064, -0.99029622, -0.90727846, -0.80944544, -0.71684477,
        -0.46409533, -0.50837543, -0.85178688, -0.87366164, -0.72868445,
        -0.75532850, -0.81285379, -0.87768300, -0.80527302, -0.49252103,
        -0.12095192, +0.82912249, +0.63839301, -0.31717423, -1.17528358,
        -1.09933917, -0.99946683, -0.90756677, -0.93714368, -1.02324346,
        -0.94719715, -0.94498089, -1.03877556, -1.04006068, -0.99541242,
        -0.90816853, -0.91440991, -0.80980250, -0.83227142, -0.79982558,
        -0.71007423, -0.43845981, +0.22445879, +1.03583135, +0.94638622,
        +0.71602430, +0.16075345, -1.10183224, -1.03596633, -0.91360057,
        -0.88579604, -0.95018153, -0.85915278, -0.93419626, -0.90309440,
        -0.80571555, -0.78413866, -0.49454092, -0.28657136, -0.65150500,
        -0.97740954, -0.94559954, -0.88062381, -0.87374228, -0.90522535,
        -0.84492822, -0.72258583, -0.73637248, -0.73825501, -0.90574770,
        -0.79609687, -0.63163971, -0.60583385, -0.71869942, -0.66268013,
        -0.11994599, +1.07885266, +0.92035608, +0.53733620, -0.55526273,
        -1.14520496, -1.03682067, -0.96812972, -0.89632027, -0.85906482,
        -0.66729218, -0.09997162, +1.02479552, +0.89355522, +0.32848599,
        -1.19546240, -1.17743185, -1.08300409, -0.98123285, -0.92178637
    ]
)

n_seq = onp.array(
    [
        -0.23451280, -0.05148377, -1.21781216, +0.49377232, -1.28415855,
        +0.97832435, -0.73132606, -1.27082805, +0.19202742, +0.97002475,
        +0.28280723, -0.21364053, -0.60917770, +1.36431661, +0.56044353,
        +0.21693361, +0.39495255, +1.26354126, -0.27260488, +0.75396557,
        -0.66777258, -0.34878700, -0.70382125, -0.88300213, -0.37812808,
        -0.37322208, -0.45522517, +0.65404033, +0.74169440, -0.65699155,
        -0.31746624, -1.01268399, +2.06597828, +0.64149737, -2.22893520,
        -0.90631233, +0.13459057, -0.26372195, -0.50162076, +0.45646887,
        +1.42213169, +2.41722240, +0.40030146, -0.73951543, -0.12806942,
        +0.62655353, -0.46068362, +0.92126315, +1.45970978, +0.40565390,
        -0.67093057, -0.23192967, -1.03654250, +0.91066752, -1.07848054,
        +0.32306376, +0.16664054, +1.49015347, +0.01924661, -1.54073079,
        +0.05653097, +0.32953229, +1.46984889, -0.49474817, -0.83621444,
        -0.75038109, -1.90162507, +0.15207242, +1.93072014, +0.67188222,
        -1.55832031, +0.68398846, +0.70626719, -0.85510939, -0.19296086,
        -0.98774542, -0.55062115, +0.07389501, -0.71148913, -0.65844659,
        -0.88734157, -1.85138096, -0.31391251, -0.43393753, +2.24069399,
        +0.36251972, +0.80098667, +0.48307318, +0.86563988, -1.39600732,
        -0.59466357, -0.02245997, -0.63621862, -1.22193028, +0.67540931,
        -0.45025060, -0.85794509, -1.77185255, -0.39615538, -0.14458991
    ]
)
# fmt: on

y_seq = (y_seq_mean + abs(args.observation_noise_std) * n_seq)[:, None]
num_obs = y_seq.shape[0]
num_steps = num_obs * args.num_steps_per_obs
obs_interval = 0.2

# Set up Mici objects

system = sde.mici_extensions.ConditionedDiffusionConstrainedSystem(
    obs_interval,
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
    adapt_stat_target=args.step_size_adaptation_target,
    log_step_size_reg_coefficient=args.step_size_reg_coefficient,
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
        "x_0": x_0,
        "σ": z[0],
        "ϵ": z[1],
        "γ": z[2],
        "β": z[3],
        "hamiltonian": system.h(state),
        **call_counts,
    }
    if variable_σ:
        traced_vars["σ_y"] = generate_σ(u)
    return traced_vars


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
    assert abs(system.constr(state)).max() < args.projection_solver_convergence_tol
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
summary_vars = ["σ", "ϵ", "γ", "β", "x_0", "hamiltonian"]
if variable_σ:
    summary_vars.append("σ_y")
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
