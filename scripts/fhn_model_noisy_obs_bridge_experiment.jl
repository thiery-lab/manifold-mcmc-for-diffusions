using ArgParse
using JSON
using Printf
using Dates
using Bridge
using StaticArrays
using BridgeSDEInference
using BridgeSDEInference: indices
using Random, LinearAlgebra, Distributions
import Distributions.logpdf
using NPZ
using PyCall
const State = SArray{Tuple{2},T,1,2} where {T}


settings = ArgParseSettings(
    description="Run FitzHugh-Nagumo model experiment (noisy observations, BridgeSDEInference)"
)

@add_arg_table! settings begin
    "--output-root-dir"
        default = "experiments"
        arg_type = String
        help = "Root directory to make experiment output subdirectory in"
    "--observation-noise-std"
        help = "Standard deviation of observation noise"
        arg_type = Float64
        default = 0.01
    "--num-steps-per-obs"
        help = "Number of time steps per interobservation interval to use in inference"
        arg_type = Int
        default = 40
    "--seed-chains"
        help = "Seed for random number generator used to simulate chains"
        arg_type = Int
        default = 20200710
    "--seed-data"
        help = "Seed for random number generator used to simulate observed data"
        arg_type = Int
        default = 4
    "--num-chain"
        help = "Number of independent chains to sample"
        arg_type = Int
        default = 4
    "--num-warm-up-iter"
        help = "Number of chain iterations in adaptive warm-up sampling stage"
        arg_type = Int
        default = 100000
    "--num-main-iter"
        help = "Number of chain iterations in main sampling stage"
        arg_type = Int
        default = 500000
    "--num-iter-per-readjust"
        help = "Number of chain iterations between each adaptation (readjustment) of sampler parameters during warm-up sampling stage"
        arg_type = Int
        default = 100
    "--pcn-imputation-update-initial-persistence"
        help = "Initial value of persistence parameter (in [0, 1]) for pre-conditioned Crank-Nicoloson update to paths"
        arg_type = Float64
        default = 0.975
    "--rwm-parameter-update-initial-step-size"
        help = "Initial value of step size parameter (positive) for random-walk Metropolis update to parameters"
        arg_type = Float64
        default = 0.5
    "--num-iter-per-print-summary"
        help = "Number of chain iterations between printing a summary of chain progress"
        arg_type = Int
        default = 500
    "--num-iter-per-save-path"
        help = "Number of chain iterations between each record of diffusion path"
        arg_type = Int
        default = 500
end

args = parse_args(ARGS, settings)


function logpdf(p::Normal, coord_idx, θ::Array{Float64})
    θi = θ[[indices(coord_idx)...]]
    @assert length(θi) == 1
    logpdf.(p, θi[1])
end

function logpdf(p::LogNormal, coord_idx, θ::Array{Float64})
    θi = θ[[indices(coord_idx)...]]
    @assert length(θi) == 1
    logpdf.(p, θi[1])
end


# Output directory

dir_name = @sprintf("σ_%.2g_%s", args["observation-noise-std"], string(DateTime(Dates.now())))
output_dir = joinpath(args["output-root-dir"], "fhn_noisy_bridge", dir_name)

if !isdir(output_dir)
    mkpath(output_dir)
end

open(joinpath(output_dir, "args.json"), "w") do f
  JSON.print(f, args, 2)
end


# Reference 'true' values

θ_ref = [
    0.1, # ϵ
    -0.8, # s
    1.5, # γ
    0.0, # β
    0.3, # σ
]
x0_ref = State(-0.5, -0.6)

# Prior distributions
θ_priors = Dict(
    "ϵ" => LogNormal(0, 1),
    "s" => Normal(0, 1),
    "γ" => LogNormal(0, 1),
    "β" => Normal(0, 0),
    "σ" => LogNormal(0, 1),
)
x0_prior = MultivariateNormal([0, 0], I)

# Diffusion

param = :regular
P = FitzhughDiffusion(param, θ_ref...)

# MCMC update information for θ coords

θ_coord_names = ["ϵ", "s", "γ", "β", "σ"]
θ_name_to_coord_dict = Dict(name => coord for (coord, name) in enumerate(θ_coord_names))
θ_update_coords = [1, 2, 3, 5]
θ_is_nonnegative = Dict(name => minimum(θ_priors[name]) == 0 for name in θ_coord_names)

# Observation scheme

num_obs = 100
L = @SMatrix [1. 0.]
σ_y = args["observation-noise-std"]
Σ = @SMatrix [σ_y^2]

# Time grid

dt = 1/50000
T = 20.0
tt = 0.0:dt:T

# Imputation grid

dt_inference = T / (num_obs * args["num-steps-per-obs"])


# Generate simulated data

Random.seed!(args["seed-data"])
X, _ = simulate_segment(0.0, x0_ref, P, tt);
num_obs = 100
num_steps_per_obs = div(length(tt), num_obs)
obs_noise = [rand(Normal(0, 1)) for i in 1:num_obs]

obs = (
    times = X.tt[num_steps_per_obs+1:num_steps_per_obs:end],
    values = [
        ((L*x)[1] + σ_y * n)
        for (x, n) in zip(X.yy[num_steps_per_obs+1:num_steps_per_obs:end], obs_noise)
    ]
)

# Generate chain initializations

Random.seed!(args["seed-chains"])
n_chain = args["num-chain"]
θ_inits = Array{Float64, 2}(undef, n_chain, length(θ_ref))
x0_inits = Array{Float64, 2}(undef, n_chain, length(x0_ref))
for c in 1:n_chain
    θ_inits[c, :] = [rand(θ_priors[name]) for name in θ_coord_names]
    x0_inits[c, :] = rand(x0_prior)
end

## Run chains

println("Starting chains for σ_y = $σ_y")
println('='^80)
mcmc_outputs = Array{Any, 1}(undef, n_chain)
mcmc_times = Array{Float64}(undef, n_chain)
for c in 1:n_chain
    println("Starting sampling chain $c of $n_chain...")
    println('-'^80)
    x0_init = State(x0_inits[c, :]...)
    θ_init = θ_inits[c, :]
    P_trgt = FitzhughDiffusion(param, θ_init...)
    P_aux = [
        FitzhughDiffusionAux(param, θ_init..., t₀, u, T, v)
        for (t₀, T, u, v) in zip(obs.times[1:end-1], obs.times[2:end],
                                 obs.values[1:end-1], obs.values[2:end])
    ]
    model_setup = DiffusionSetup(P_trgt, P_aux, PartObs())
    set_imputation_grid!(model_setup, dt_inference)
    set_observations!(
        model_setup, [L for _ in P_aux], [Σ for _ in P_aux],
        obs.values, obs.times)
    set_x0_prior!(model_setup, GsnStartingPt(mean(x0_prior), cov(x0_prior)), x0_init)
    initialise!(eltype(x0_init), model_setup, Vern7(), false, NoChangePt(100))
    set_auxiliary!(model_setup; skip_for_save=1, adaptive_prop=NoAdaptation())
    θ_updates = [
        ParamUpdate(
            MetropolisHastingsUpdt(), i, θ_init,
            UniformRandomWalk(args["rwm-parameter-update-initial-step-size"], θ_is_nonnegative[name]),
            θ_priors[name],
            UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P_aux, i))
        )
        for (i, name) in zip(θ_update_coords, θ_coord_names[θ_update_coords])
    ]
    mcmc_setup = MCMCSetup(
        Imputation(NoBlocking(), args["pcn-imputation-update-initial-persistence"],
        Vern7()), θ_updates...
    )
    schedule = MCMCSchedule(
        args["num-warm-up-iter"] + args["num-main-iter"], [[1, 2, 3, 4, 5]],
        (save=args["num-iter-per-save-path"], verbose=args["num-iter-per-print-summary"], warm_up=0,
        readjust=(x->((x%args["num-iter-per-readjust"]==0) && (x < args["num-warm-up-iter"]))), fuse=(x->false))
    )
    GC.gc()
    mcmc_times[c] = @elapsed mcmc_outputs[c] = mcmc(mcmc_setup, schedule, model_setup)
    println("...finshed sampling chain $c of $n_chain, time taken: $(mcmc_times[c]) seconds.")
    println('-'^80)
end

num_path = size(mcmc_outputs[1][1].paths, 1)
paths = permutedims(
    cat([
            cat([
                    hcat([Array(x) for x in mcmc_outputs[c][1].paths[p]]...)
                    for p in 1:num_path
                ]..., dims=3)
            for c in 1:n_chain
        ]..., dims=4
    ), (4, 3, 2, 1)
)
path_times = vcat(mcmc_outputs[1][1].time...)
traces = Dict(
    name=>hcat([[c[i] for c in output[2].θ_chain] for output in mcmc_outputs]...)'
    for (i, name) in zip(θ_update_coords, θ_coord_names[θ_update_coords]))
npzwrite(
    joinpath(output_dir, "traces.npz"),
    merge(traces, Dict("chain_times" => mcmc_times, "paths" => paths, "path_times" => path_times))
)

arviz = pyimport("arviz")
summary = arviz.summary(traces)
summary_dict = summary.to_dict()
summary_dict["total_sampling_time"] = sum(mcmc_times)

open(joinpath(output_dir, "summary.json"), "w") do f
  JSON.print(f, summary_dict, 2)
end

show(summary)

println("\nSaved chain traces and summary to " * output_dir)
println('='^80)
