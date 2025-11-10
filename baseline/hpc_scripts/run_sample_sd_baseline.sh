#!/bin/bash
#BSUB -J sample_sd_baseline
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/baseline/hpc_scripts/logs/sample_sd_baseline.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/baseline/hpc_scripts/logs/sample_sd_baseline.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 4:00
#BSUB -q gpuv100

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
SCRIPT_PATH="${BASE_DIR}/baseline/sample_sd_baseline.py"
LOG_DIR="${BASE_DIR}/baseline/hpc_scripts/logs"

# Output directory for samples
OUTPUT_DIR="${BASE_DIR}/outputs/baseline_sd_unconditional"

# Sampling parameters
NUM_SAMPLES=64
NUM_STEPS=50
SEED=42

# Model ID
MODEL_ID="runwayml/stable-diffusion-v1-5"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "========================================="
echo "Sampling from Pretrained Stable Diffusion"
echo "========================================="
echo "Script: ${SCRIPT_PATH}"
echo "Model: ${MODEL_ID}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Number of samples: ${NUM_SAMPLES}"
echo "DDIM steps: ${NUM_STEPS}"
echo ""

# Check required files
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "ERROR: Script not found: ${SCRIPT_PATH}" >&2
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
# Set Hugging Face cache to /work3 (more space than home directory)
HF_CACHE_DIR="/work3/s233249/.cache/huggingface"
mkdir -p "${HF_CACHE_DIR}"
export HF_HOME="${HF_CACHE_DIR}"
export HF_HUB_CACHE="${HF_CACHE_DIR}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/transformers"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"
echo "Hugging Face cache directory: ${HF_CACHE_DIR}"

# ----------------------------------------------------------------------
# Run script
cd "${BASE_DIR}"
python "${SCRIPT_PATH}" \
    --model_id "${MODEL_ID}" \
    --num_samples "${NUM_SAMPLES}" \
    --num_steps "${NUM_STEPS}" \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "cuda" \
    --unconditional

echo ""
echo "========================================="
echo "Sampling Completed"
echo "========================================="
echo "Samples saved to: ${OUTPUT_DIR}"

