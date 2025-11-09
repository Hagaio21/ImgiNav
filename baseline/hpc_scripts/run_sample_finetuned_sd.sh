#!/bin/bash
#BSUB -J sample_finetuned_sd
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/baseline/hpc_scripts/logs/sample_finetuned_sd.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/baseline/hpc_scripts/logs/sample_finetuned_sd.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 4:00
#BSUB -q gpua100

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
SCRIPT_PATH="${BASE_DIR}/baseline/sample_finetuned_sd.py"
LOG_DIR="${BASE_DIR}/baseline/hpc_scripts/logs"

# Base directory for fine-tuned models
FINETUNED_BASE_DIR="${BASE_DIR}/outputs/baseline_sd_finetuned"

# Output directory for samples
OUTPUT_DIR="${BASE_DIR}/outputs/baseline_sd_finetuned_samples"

# Sampling parameters
NUM_SAMPLES=64
NUM_STEPS=50
GUIDANCE_SCALE=1.0
SEED=42

# Auto-detect model directory (prefer best checkpoint, fallback to final)
if [ -d "${FINETUNED_BASE_DIR}/checkpoint-best/pipeline" ]; then
    MODEL_DIR="${FINETUNED_BASE_DIR}/checkpoint-best/pipeline"
    MODEL_TYPE="best checkpoint (recommended)"
elif [ -d "${FINETUNED_BASE_DIR}/pipeline" ]; then
    MODEL_DIR="${FINETUNED_BASE_DIR}/pipeline"
    MODEL_TYPE="final model"
else
    MODEL_DIR="${FINETUNED_BASE_DIR}/pipeline"
    MODEL_TYPE="not found"
fi

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "========================================="
echo "Sampling from Fine-tuned Stable Diffusion"
echo "========================================="
echo "Script: ${SCRIPT_PATH}"
echo "Model directory: ${MODEL_DIR}"
echo "Model type: ${MODEL_TYPE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Number of samples: ${NUM_SAMPLES}"
echo "DDIM steps: ${NUM_STEPS}"
echo ""

# Check required files
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "ERROR: Script not found: ${SCRIPT_PATH}" >&2
    exit 1
fi
if [ ! -d "${MODEL_DIR}" ]; then
    echo "ERROR: Model directory not found: ${MODEL_DIR}" >&2
    echo "Fine-tune SD first using: bsub < baseline/hpc_scripts/run_finetune_sd.sh" >&2
    echo ""
    echo "Available directories in ${FINETUNED_BASE_DIR}:"
    ls -la "${FINETUNED_BASE_DIR}" 2>/dev/null || echo "  (directory does not exist)"
    exit 1
fi

# ----------------------------------------------------------------------
# CUDA modules
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X

# ----------------------------------------------------------------------
# Conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || conda activate scenefactor || exit 1
fi

# ----------------------------------------------------------------------
# Run script
cd "${BASE_DIR}"
python "${SCRIPT_PATH}" \
    --model_dir "${MODEL_DIR}" \
    --num_samples "${NUM_SAMPLES}" \
    --num_steps "${NUM_STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "cuda"

echo ""
echo "========================================="
echo "Sampling Completed"
echo "========================================="
echo "Samples saved to: ${OUTPUT_DIR}"

