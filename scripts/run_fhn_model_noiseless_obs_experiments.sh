#!/usr/bin/env bash
# Run FitzHugh-Nagumo model (noiseless observations) experiments and generate plots

trap "echo Stopping; exit;" SIGINT SIGTERM

# Default number of observation times when not grid variable
DEFAULT_NUM_OBS=100

# Default number of steps per observation time when not grid variable
DEFAULT_NUM_STEPS_PER_OBS=25

# Default number of observation times per subsequence when not grid variable
DEFAULT_NUM_OBS_PER_SUBSEQ=5

# Grid of number of observation times to run experiments for
NUM_OBS_GRID=(25 50 100 200 400)

# Grid of number of steps per observation time to run experiments for
NUM_STEPS_PER_OBS_GRID=(25 50 100 200 400)

# Grid of number of observation times per subsequence to run experiments for
NUM_OBS_PER_SUBSEQ_GRID=(2 5 10 20 50 100)

# Number of independent chains to run per experiment
NUM_CHAIN=4

# Number of iterations in adaptive warm-up stage of chains
NUM_WARM_UP_ITER=250

# Number of iterations in main stage of chains
NUM_MAIN_ITER=1000

# Values of Hamiltonian splittings to run experiments for
SPLITTING_VALS=("standard" "gaussian")

# Values of random seeds to run experiments for
SEED_VALS=(20200710 20200711 20200712)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
EXPERIMENT_DIR="$(dirname ${SCRIPT_DIR})/experiments"
FIGURE_DIR="$(dirname ${SCRIPT_DIR})/figures"

echo "================================================================================"
echo "Running FitzHugh-Nagumo model noiseless observation CHMC experiments"
echo "Saving results to ${EXPERIMENT_DIR}"
echo "Saving figures to ${FIGURE_DIR}"

echo "--------------------------------------------------------------------------------"
echo "Varying number of observation times"
echo "--------------------------------------------------------------------------------"
for SEED in ${SEED_VALS[@]}; do
    for SPLITTING in ${SPLITTING_VALS[@]}; do
        for NUM_OBS in ${NUM_OBS_GRID[@]}; do
            echo "Seed ${SEED}, ${SPLITTING} splitting, ${NUM_OBS} obs. times"
            python ${SCRIPT_DIR}/fhn_model_noiseless_obs_chmc_experiment.py \
              --num-obs-per-subseq ${DEFAULT_NUM_OBS_PER_SUBSEQ} \
              --num-steps-per-obs ${DEFAULT_NUM_STEPS_PER_OBS} \
              --num-obs ${NUM_OBS} \
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
echo "Varying number of time steps per observation time"
echo "--------------------------------------------------------------------------------"
for SEED in ${SEED_VALS[@]}; do
    for SPLITTING in ${SPLITTING_VALS[@]}; do
        for NUM_STEPS_PER_OBS in ${NUM_STEPS_PER_OBS_GRID[@]}; do
            echo "Seed ${SEED}, ${SPLITTING} splitting, ${NUM_STEPS_PER_OBS} steps per obs."
            python ${SCRIPT_DIR}/fhn_model_noiseless_obs_chmc_experiment.py \
              --num-obs-per-subseq ${DEFAULT_NUM_OBS_PER_SUBSEQ} \
              --num-steps-per-obs ${NUM_STEPS_PER_OBS} \
              --num-obs ${DEFAULT_NUM_OBS} \
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
echo "Varying number of observation times per subsequence"
echo "--------------------------------------------------------------------------------"
for SEED in ${SEED_VALS[@]}; do
    for SPLITTING in ${SPLITTING_VALS[@]}; do
        for NUM_OBS_PER_SUBSEQ in ${NUM_OBS_PER_SUBSEQ_GRID[@]}; do
            echo "Seed ${SEED}, ${SPLITTING} splitting, ${NUM_OBS_PER_SUBSEQ} obs. per subseq"
            python ${SCRIPT_DIR}/fhn_model_noiseless_obs_chmc_experiment.py \
              --num-obs-per-subseq ${NUM_OBS_PER_SUBSEQ} \
              --num-steps-per-obs ${DEFAULT_NUM_STEPS_PER_OBS} \
              --num-obs ${DEFAULT_NUM_OBS} \
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
echo "Computing average operation times"
echo "--------------------------------------------------------------------------------"
# Use taskset to run on one-core only to give fair reflection of serial timings
taskset -c 0 python ${SCRIPT_DIR}/fhn_model_noiseless_obs_chmc_operation_times.py \
  --output-root-dir ${EXPERIMENT_DIR} \
  --num-obs-per-subseq-grid ${NUM_OBS_PER_SUBSEQ_GRID[@]} \
  --num-steps-per-obs-grid ${NUM_STEPS_PER_OBS_GRID[@]} \
  --num-obs-grid ${NUM_OBS_GRID[@]} \
  --default-num-obs-per-subseq ${DEFAULT_NUM_OBS_PER_SUBSEQ} \
  --default-num-steps-per-obs ${DEFAULT_NUM_STEPS_PER_OBS} \
  --default-num-obs ${DEFAULT_NUM_OBS}

echo "Generating plots"
python ${SCRIPT_DIR}/fhn_model_noiseless_obs_generate_plots.py \
  --experiment-dir ${EXPERIMENT_DIR} \
  --output-dir ${FIGURE_DIR} \
  --num-obs-per-subseq-grid ${NUM_OBS_PER_SUBSEQ_GRID[@]} \
  --num-steps-per-obs-grid ${NUM_STEPS_PER_OBS_GRID[@]} \
  --num-obs-grid ${NUM_OBS_GRID[@]} \
  --default-num-obs-per-subseq ${DEFAULT_NUM_OBS_PER_SUBSEQ} \
  --default-num-steps-per-obs ${DEFAULT_NUM_STEPS_PER_OBS} \
  --default-num-obs ${DEFAULT_NUM_OBS}