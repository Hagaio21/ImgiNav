#!/bin/bash
#BSUB -J memorization_check
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/scripts/hpc_scripts/logs/memorization_check.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/scripts/hpc_scripts/logs/memorization_check.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 4:00
#BSUB -q gpuv100

set -euo pipefail

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
LOG_DIR="${BASE_DIR}/scripts/hpc_scripts/logs"
MEMORIZATION_SCRIPT="${BASE_DIR}/scripts/check_memorization.py"

mkdir -p "${LOG_DIR}"

# Parse arguments
CONFIG_FILE="${1}"
CHECKPOINT_PATH="${2}"
MANIFEST_PATH="${3}"
OUTPUT_DIR="${4}"
NUM_GENERATE="${5:-100}"
NUM_TRAINING="${6:-5000}"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}" >&2
    exit 1
fi

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Checkpoint file not found: ${CHECKPOINT_PATH}" >&2
    exit 1
fi

if [ ! -f "${MANIFEST_PATH}" ]; then
    echo "ERROR: Manifest file not found: ${MANIFEST_PATH}" >&2
    exit 1
fi

module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || conda activate scenefactor || exit 1
fi

cd "${BASE_DIR}"

echo "Running memorization check:"
echo "  Config: ${CONFIG_FILE}"
echo "  Checkpoint: ${CHECKPOINT_PATH}"
echo "  Manifest: ${MANIFEST_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Num generate: ${NUM_GENERATE}"
echo "  Num training: ${NUM_TRAINING}"

python "${MEMORIZATION_SCRIPT}" \
    --config "${CONFIG_FILE}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --manifest "${MANIFEST_PATH}" \
    --output "${OUTPUT_DIR}" \
    --num_generate "${NUM_GENERATE}" \
    --num_training "${NUM_TRAINING}" \
    --method ddpm

echo "Memorization check completed successfully"

