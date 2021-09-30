#!/usr/bin/env bash
# Run Susceptible-Recovered-Infected (SIR) model experiments and generate plots

trap "echo Stopping; exit;" SIGINT SIGTERM

# Grid of observation noise standard deviations to run experiments for
# -1 value indicates unknown observation noise standard deviation as variable to infer
OBS_NOISE_STD_GRID=(0.3162 1 3.162 10 -1)

# Number of independent chains to run per experiment
NUM_CHAIN=4

# Number of iterations in adaptive warm-up stage of chains
NUM_WARM_UP_ITER=500

# Number of iterations in main stage of chains
NUM_MAIN_ITER=2500

# Values of random seeds to run experiments for
SEED_VALS=(20200710)

# Values of Hamiltonian splittings to run experiments for (CHMC only)
SPLITTING_VALS=("standard" "gaussian")

# Values of metric (mass matrix) types to run experiments for (HMC only)
METRIC_TYPE_VALS=("identity" "diagonal")

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
EXPERIMENT_DIR="$(dirname ${SCRIPT_DIR})/experiments"
FIGURE_DIR="$(dirname ${SCRIPT_DIR})/figures"

echo "================================================================================"
echo "Running Susceptible-Infected-Recovered (SIR) model experiments"
echo "Saving results to ${EXPERIMENT_DIR}"
echo "Saving figures to ${FIGURE_DIR}"

echo "--------------------------------------------------------------------------------"
echo "Varying observation noise standard deviation with CHMC"
echo "--------------------------------------------------------------------------------"
for SEED in ${SEED_VALS[@]}; do
    for SPLITTING in ${SPLITTING_VALS[@]}; do
        for OBS_NOISE_STD in ${OBS_NOISE_STD_GRID[@]}; do
            echo "Seed ${SEED}, ${SPLITTING} splitting, obs. noise std ${OBS_NOISE_STD}"
            python ${SCRIPT_DIR}/sir_model_chmc_experiment.py \
              --observation-noise-std ${OBS_NOISE_STD} \
              --splitting ${SPLITTING} \
              --seed ${SEED} \
              --output-root-dir ${EXPERIMENT_DIR} \
              --num-chain ${NUM_CHAIN} \
              --num-warm-up-iter ${NUM_WARM_UP_ITER} \
              --num-main-iter ${NUM_MAIN_ITER}
        done
    done
done

echo "--------------------------------------------------------------------------------"
echo "Varying observation noise standard deviation with HMC"
echo "--------------------------------------------------------------------------------"
for SEED in ${SEED_VALS[@]}; do
    for METRIC_TYPE in ${METRIC_TYPE_VALS[@]}; do
        for OBS_NOISE_STD in ${OBS_NOISE_STD_GRID[@]}; do
            echo "Seed ${SEED}, ${METRIC_TYPE} metric, obs. noise std ${OBS_NOISE_STD}"
            python ${SCRIPT_DIR}/sir_model_hmc_experiment.py \
              --observation-noise-std ${OBS_NOISE_STD} \
              --metric-type ${METRIC_TYPE} \
              --seed ${SEED} \
              --output-root-dir ${EXPERIMENT_DIR} \
              --num-chain ${NUM_CHAIN} \
              --num-warm-up-iter ${NUM_WARM_UP_ITER} \
              --num-main-iter ${NUM_MAIN_ITER}
        done
    done
done

echo "--------------------------------------------------------------------------------"
echo "Generating plots"
echo "--------------------------------------------------------------------------------"
python ${SCRIPT_DIR}/sir_model_generate_plots.py \
  --experiment-dir ${EXPERIMENT_DIR} \
  --output-dir ${FIGURE_DIR} \
  --obs-noise-std-grid ${OBS_NOISE_STD_GRID[@]}
