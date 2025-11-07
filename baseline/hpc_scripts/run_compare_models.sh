#!/bin/bash
#BSUB -J compare_models
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/baseline/hpc_scripts/logs/compare_models.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/baseline/hpc_scripts/logs/compare_models.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 4:00
#BSUB -q gpua100

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
SCRIPT_PATH="${BASE_DIR}/baseline/compare_models.py"
LOG_DIR="${BASE_DIR}/baseline/hpc_scripts/logs"

# Custom model checkpoint and config
# Update these paths for your specific checkpoint
CUSTOM_CHECKPOINT="${BASE_DIR}/experiments/diffusion/stage3/stage3_unet128_d4/checkpoints/diffusion_stage3_unet128_d4_checkpoint_best.pt"
CUSTOM_CONFIG="${BASE_DIR}/experiments/diffusion/stage3/stage3_unet128_d4.yaml"

# SD model (choose one):
# Option 1: Pretrained SD
SD_MODEL_ID="runwayml/stable-diffusion-v1-5"
SD_MODEL_DIR=""

# Option 2: Fine-tuned SD (uncomment to use)
# SD_MODEL_ID=""
# SD_MODEL_DIR="${BASE_DIR}/outputs/baseline_sd_finetuned/pipeline"

# Output directory
OUTPUT_DIR="${BASE_DIR}/outputs/comparison_stage3_vs_sd"

# Sampling parameters
NUM_SAMPLES=64
NUM_STEPS=50
SEED=42

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "========================================="
echo "Comparing Custom Model vs Stable Diffusion"
echo "========================================="
echo "Script: ${SCRIPT_PATH}"
echo "Custom checkpoint: ${CUSTOM_CHECKPOINT}"
echo "Custom config: ${CUSTOM_CONFIG}"
if [ -n "${SD_MODEL_ID}" ]; then
    echo "SD model: ${SD_MODEL_ID} (pretrained)"
else
    echo "SD model: ${SD_MODEL_DIR} (fine-tuned)"
fi
echo "Output directory: ${OUTPUT_DIR}"
echo "Number of samples: ${NUM_SAMPLES}"
echo "DDIM steps: ${NUM_STEPS}"
echo ""

# Check required files
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "ERROR: Script not found: ${SCRIPT_PATH}" >&2
    exit 1
fi
if [ ! -f "${CUSTOM_CHECKPOINT}" ]; then
    echo "ERROR: Custom checkpoint not found: ${CUSTOM_CHECKPOINT}" >&2
    exit 1
fi
if [ ! -f "${CUSTOM_CONFIG}" ]; then
    echo "ERROR: Custom config not found: ${CUSTOM_CONFIG}" >&2
    exit 1
fi

if [ -n "${SD_MODEL_DIR}" ] && [ ! -d "${SD_MODEL_DIR}" ]; then
    echo "ERROR: SD model directory not found: ${SD_MODEL_DIR}" >&2
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

if [ -n "${SD_MODEL_ID}" ]; then
    # Use pretrained SD
    python "${SCRIPT_PATH}" \
        --custom_checkpoint "${CUSTOM_CHECKPOINT}" \
        --custom_config "${CUSTOM_CONFIG}" \
        --sd_model_id "${SD_MODEL_ID}" \
        --num_samples "${NUM_SAMPLES}" \
        --num_steps "${NUM_STEPS}" \
        --seed "${SEED}" \
        --output_dir "${OUTPUT_DIR}" \
        --device "cuda"
else
    # Use fine-tuned SD
    python "${SCRIPT_PATH}" \
        --custom_checkpoint "${CUSTOM_CHECKPOINT}" \
        --custom_config "${CUSTOM_CONFIG}" \
        --sd_model_dir "${SD_MODEL_DIR}" \
        --num_samples "${NUM_SAMPLES}" \
        --num_steps "${NUM_STEPS}" \
        --seed "${SEED}" \
        --output_dir "${OUTPUT_DIR}" \
        --device "cuda"
fi

echo ""
echo "========================================="
echo "Comparison Complete"
echo "========================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo "  - custom_model/: Custom model samples"
echo "  - stable_diffusion/: SD samples"
echo "  - custom_model_grid.png: Custom model grid"
echo "  - sd_grid.png: SD grid"
echo "  - comparison_grid.png: Side-by-side comparison"

