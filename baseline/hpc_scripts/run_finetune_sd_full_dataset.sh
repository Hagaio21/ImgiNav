#!/bin/bash
#BSUB -J finetune_sd_full
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/baseline/hpc_scripts/logs/finetune_sd_full.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/baseline/hpc_scripts/logs/finetune_sd_full.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=32000]"
#BSUB -gpu "num=1"
#BSUB -W 48:00
#BSUB -q gpuv100

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
SCRIPT_PATH="${BASE_DIR}/baseline/finetune_sd.py"
LOG_DIR="${BASE_DIR}/baseline/hpc_scripts/logs"

# Input manifest path (use your full dataset manifest)
# Update this path to point to your main dataset manifest
# Examples:
#   - /work3/s233249/ImgiNav/datasets/layouts.csv (main layout manifest)
#   - /work3/s233249/ImgiNav/datasets/augmented/manifest_images.csv (augmented dataset)
MANIFEST_PATH="/work3/s233249/ImgiNav/datasets/augmented/manifest_images.csv"

# Output directory for fine-tuned model
OUTPUT_DIR="${BASE_DIR}/outputs/baseline_sd_finetuned_full"

# Training parameters
EPOCHS=50
BATCH_SIZE=4
LEARNING_RATE=5e-6  # Reduced from 1e-5 for stability (NaN loss prevention)
NUM_WORKERS=8

# Filtering parameters (same as diffusion training)
FILTER_EMPTY=true
WHITENESS_THRESHOLD=0.95  # Filter out images with whiteness_ratio >= 0.95

# Model ID
MODEL_ID="runwayml/stable-diffusion-v1-5"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "========================================="
echo "Fine-tuning Stable Diffusion on Full Dataset"
echo "========================================="
echo "Script: ${SCRIPT_PATH}"
echo "Manifest: ${MANIFEST_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Model: ${MODEL_ID}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Filter empty: ${FILTER_EMPTY}"
echo "Whiteness threshold: ${WHITENESS_THRESHOLD}"
echo ""
echo "Checkpoint strategy: Best checkpoint only (saves disk space)"
echo "  - Only the best checkpoint will be saved (no final model)"
echo "  - UNet will be saved in checkpoint-best/unet for ControlNet use"
echo ""

# Check required files
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "ERROR: Script not found: ${SCRIPT_PATH}" >&2
    exit 1
fi
if [ ! -f "${MANIFEST_PATH}" ]; then
    echo "ERROR: Manifest not found: ${MANIFEST_PATH}" >&2
    echo "Please update MANIFEST_PATH in this script to point to your dataset manifest." >&2
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

# Build command arguments
CMD_ARGS=(
    --manifest_path "${MANIFEST_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --model_id "${MODEL_ID}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --learning_rate "${LEARNING_RATE}"
    --num_workers "${NUM_WORKERS}"
    --device "cuda"
    --seed 42
    --whiteness_threshold "${WHITENESS_THRESHOLD}"
)

# Add filter_empty flag if enabled
if [ "${FILTER_EMPTY}" = "true" ]; then
    CMD_ARGS+=(--filter_empty)
fi

python "${SCRIPT_PATH}" "${CMD_ARGS[@]}"

echo ""
echo "========================================="
echo "Fine-tuning Completed"
echo "========================================="
echo "Fine-tuned model saved to: ${OUTPUT_DIR}"
echo ""

# Check what checkpoints are available
echo "Available checkpoints:"
if [ -d "${OUTPUT_DIR}/checkpoint-best" ]; then
    echo "  ✓ Best checkpoint: ${OUTPUT_DIR}/checkpoint-best"
    if [ -d "${OUTPUT_DIR}/checkpoint-best/pipeline" ]; then
        echo "    → Pipeline: ${OUTPUT_DIR}/checkpoint-best/pipeline (for sampling)"
    fi
    if [ -d "${OUTPUT_DIR}/checkpoint-best/unet" ]; then
        echo "    → UNet: ${OUTPUT_DIR}/checkpoint-best/unet (for ControlNet use)"
    fi
fi

# Check for old-style checkpoints (backward compatibility)
OLD_CHECKPOINTS=$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | grep -v "checkpoint-best" || true)
if [ -n "${OLD_CHECKPOINTS}" ]; then
    echo ""
    echo "  ⚠ Old-style checkpoints found (from previous runs):"
    echo "${OLD_CHECKPOINTS}" | while read -r checkpoint; do
        echo "    - ${checkpoint}"
    done
    echo "    These are from older versions. Consider cleaning them up to save space."
fi

echo ""
echo "========================================="
echo "Usage Instructions"
echo "========================================="
echo ""
echo "To sample from the fine-tuned model:"
echo "  python baseline/sample_finetuned_sd.py \\"
echo "    --model_dir ${OUTPUT_DIR}/checkpoint-best/pipeline \\"
echo "    --num_samples 64 \\"
echo "    --output_dir outputs/baseline_sd_finetuned_samples"
echo ""
echo "To use the fine-tuned UNet with ControlNet:"
echo "  python baseline/use_finetuned_unet_with_controlnet.py \\"
echo "    --checkpoint_dir ${OUTPUT_DIR}/checkpoint-best \\"
echo "    --num_samples 64 \\"
echo "    --output_dir outputs/controlnet_finetuned_samples"
echo ""
echo "Or load the UNet programmatically:"
echo "  from diffusers import UNet2DConditionModel, ControlNetModel"
echo "  unet = UNet2DConditionModel.from_pretrained('${OUTPUT_DIR}/checkpoint-best/unet')"
echo "  controlnet = ControlNetModel.from_config(unet.config)"
echo "  # ControlNet will share encoder weights with the fine-tuned UNet"
echo ""

