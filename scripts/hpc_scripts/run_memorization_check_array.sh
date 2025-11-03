#!/bin/bash
#BSUB -J memorization_check[1-6]
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/scripts/hpc_scripts/logs/memorization_check.%J.%I.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/scripts/hpc_scripts/logs/memorization_check.%J.%I.err
#BSUB -n 4
#BSUB -R "rusage[mem=64000]"
#BSUB -gpu "num=1"
#BSUB -W 8:00
#BSUB -q gpuv100

set -euo pipefail

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
LOG_DIR="${BASE_DIR}/scripts/hpc_scripts/logs"
MEMORIZATION_SCRIPT="${BASE_DIR}/scripts/check_memorization.py"
ABLATION_DIR="${BASE_DIR}/experiments/diffusion/ablation"
OUTPUT_BASE="/work3/s233249/ImgiNav/ImgiNav/analysis/memorization"
MANIFEST_PATH="/work3/s233249/ImgiNav/datasets/layouts_latents.csv"  # Use original pre-augmented dataset for memorization testing

mkdir -p "${LOG_DIR}"

# List of config files (matching array indices [1-6])
declare -a CONFIG_FILES=(
    "${ABLATION_DIR}/capacity_unet64_d4.yaml"
    "${ABLATION_DIR}/capacity_unet64_d5.yaml"
    "${ABLATION_DIR}/capacity_unet128_d3.yaml"
    "${ABLATION_DIR}/capacity_unet128_d4.yaml"
    "${ABLATION_DIR}/capacity_unet256_d4.yaml"
    "${ABLATION_DIR}/scheduler_linear.yaml"
)

JOB_INDEX=${LSB_JOBINDEX}
IDX=$((JOB_INDEX - 1))  # Convert to 0-based index

if [ ${IDX} -ge ${#CONFIG_FILES[@]} ]; then
    echo "Job index ${JOB_INDEX} exceeds number of configs (${#CONFIG_FILES[@]})"
    exit 0
fi

CONFIG_FILE="${CONFIG_FILES[${IDX}]}"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}" >&2
    exit 1
fi

# Extract experiment name
EXP_NAME=$(grep "^experiment:" -A 10 "${CONFIG_FILE}" | grep "name:" | head -1 | awk '{print $2}' | tr -d '"' | tr -d "'")
if [ -z "${EXP_NAME}" ]; then
    EXP_NAME=$(basename "${CONFIG_FILE}" .yaml)
fi

echo "========================================="
echo "Memorization Check - Job ${JOB_INDEX}/${#CONFIG_FILES[@]}"
echo "========================================="
echo "Config: ${CONFIG_FILE}"
echo "Experiment: ${EXP_NAME}"
echo ""

# Find checkpoint - ALWAYS use best checkpoint
SAVE_PATH=$(grep "^experiment:" -A 10 "${CONFIG_FILE}" | grep "save_path:" | head -1 | awk '{print $2}' | tr -d '"' | tr -d "'")

if [ -z "${SAVE_PATH}" ] || [ ! -d "${SAVE_PATH}" ]; then
    echo "ERROR: Save path not found or invalid: ${SAVE_PATH}" >&2
    exit 1
fi

# Always check for best checkpoint first
CHECKPOINT_PATH="${SAVE_PATH}/${EXP_NAME}_checkpoint_best.pt"

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Best checkpoint not found: ${CHECKPOINT_PATH}" >&2
    echo "Available checkpoints in ${SAVE_PATH}:" >&2
    ls -lh "${SAVE_PATH}"/*.pt 2>/dev/null | head -5 >&2 || echo "  (none found)" >&2
    exit 1
fi

echo "Checkpoint: ${CHECKPOINT_PATH}"

# Create output directory
OUTPUT_DIR="${OUTPUT_BASE}/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}"

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

echo "Running memorization check..."
echo "  Config: ${CONFIG_FILE}"
echo "  Checkpoint: ${CHECKPOINT_PATH}"
echo "  Manifest: ${MANIFEST_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

# Build command - omit --num_training to use entire dataset
python "${MEMORIZATION_SCRIPT}" \
    --config "${CONFIG_FILE}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --manifest "${MANIFEST_PATH}" \
    --output "${OUTPUT_DIR}" \
    --num_generate 1000 \
    --method ddpm

echo ""
echo "========================================="
echo "Memorization check completed successfully"
echo "Job ${JOB_INDEX}/${#CONFIG_FILES[@]} - ${EXP_NAME}"
echo "Results: ${OUTPUT_DIR}"
echo "========================================="

