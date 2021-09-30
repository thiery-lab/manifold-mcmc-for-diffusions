#!/usr/bin/env bash
# Run FitzHugh-Nagumo model (noisy observations) experiments and generate plots

trap "echo Stopping; exit;" SIGINT SIGTERM

# Grid of observation noise standard deviations to run experiments for
OBS_NOISE_STD_GRID=(0.01 0.03162 0.1 0.3162)

# Number of independent chains to run per experiment
NUM_CHAIN=4

# Number of iterations in adaptive warm-up stage of chains (HMC + CHMC)
NUM_WARM_UP_ITER=500

# Number of iterations in main stage of chains (HMC + CHMC)
NUM_MAIN_ITER=2500

# Number of iterations in adaptive warm-up stage of chains (BridgeSDEInference)
NUM_WARM_UP_ITER_BRIDGE=100000

# Number of iterations in main stage of chains (BridgeSDEInference)
NUM_MAIN_ITER_BRIDGE=500000

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
EXPERIMENT_DIR="$(dirname ${SCRIPT_DIR})/experiments"
FIGURE_DIR="$(dirname ${SCRIPT_DIR})/figures"

echo "================================================================================"
echo "Running FitzHugh-Nagumo model noisy observation experiments"
echo "Saving results to ${EXPERIMENT_DIR}"
echo "Saving figures to ${FIGURE_DIR}"

echo "--------------------------------------------------------------------------------"
echo "Varying observation noise standard deviation with CHMC"
echo "--------------------------------------------------------------------------------"
for OBS_NOISE_STD in ${OBS_NOISE_STD_GRID[@]}; do
    echo "Observation noise standard deviation = ${OBS_NOISE_STD}"
    python ${SCRIPT_DIR}/fhn_model_noisy_obs_chmc_experiment.py \
        --observation-noise-std ${OBS_NOISE_STD} \
        --output-root-dir ${EXPERIMENT_DIR} \
        --num-chain ${NUM_CHAIN} \
        --num-warm-up-iter ${NUM_WARM_UP_ITER} \
        --num-main-iter ${NUM_MAIN_ITER}
done

echo "--------------------------------------------------------------------------------"
echo "Varying observation noise standard deviation with HMC"
echo "--------------------------------------------------------------------------------"
for OBS_NOISE_STD in ${OBS_NOISE_STD_GRID[@]}; do
    echo "Observation noise standard deviation = ${OBS_NOISE_STD}"
    python ${SCRIPT_DIR}/fhn_model_noisy_obs_hmc_experiment.py \
        --observation-noise-std ${OBS_NOISE_STD} \
        --output-root-dir ${EXPERIMENT_DIR} \
        --num-chain ${NUM_CHAIN} \
        --num-warm-up-iter ${NUM_WARM_UP_ITER} \
        --num-main-iter ${NUM_MAIN_ITER} \
        --metric-type diagonal
done

echo "--------------------------------------------------------------------------------"
echo "Varying observation noise standard deviation with BridgeSDEInference"
echo "--------------------------------------------------------------------------------"
for OBS_NOISE_STD in ${OBS_NOISE_STD_GRID[@]}; do
    echo "Observation noise standard deviation = ${OBS_NOISE_STD}"
    julia ${SCRIPT_DIR}/fhn_model_noisy_obs_bridge_experiment.jl \
        --observation-noise-std ${OBS_NOISE_STD} \
        --output-root-dir ${EXPERIMENT_DIR} \
        --num-chain ${NUM_CHAIN} \
        --num-warm-up-iter ${NUM_WARM_UP_ITER_BRIDGE} \
        --num-main-iter ${NUM_MAIN_ITER_BRIDGE}
done

echo "--------------------------------------------------------------------------------"
echo "Generating plots"
echo "--------------------------------------------------------------------------------"
python ${SCRIPT_DIR}/fhn_model_noisy_obs_generate_plots.py \
  --experiment-dir ${EXPERIMENT_DIR} \
  --output-dir ${FIGURE_DIR} \
  --obs-noise-std-grid ${OBS_NOISE_STD_GRID[@]}
