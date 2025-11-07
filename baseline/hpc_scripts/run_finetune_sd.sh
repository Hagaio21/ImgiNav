#!/bin/bash
#BSUB -J finetune_sd
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/baseline/hpc_scripts/logs/finetune_sd.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/baseline/hpc_scripts/logs/finetune_sd.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=32000]"
#BSUB -gpu "num=1"
#BSUB -W 24:00
#BSUB -q gpua100

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
SCRIPT_PATH="${BASE_DIR}/baseline/finetune_sd.py"
LOG_DIR="${BASE_DIR}/baseline/hpc_scripts/logs"

# Input dataset directory (from prepare_layout_dataset.sh)
DATASET_DIR="/work3/s233249/ImgiNav/datasets/sd_finetuning_images"

# Output directory for fine-tuned model
OUTPUT_DIR="${BASE_DIR}/outputs/baseline_sd_finetuned"

# Training parameters
EPOCHS=50
BATCH_SIZE=4
LEARNING_RATE=1e-5
NUM_WORKERS=8
SAVE_STEPS=500

# Model ID
MODEL_ID="runwayml/stable-diffusion-v1-5"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "========================================="
echo "Fine-tuning Stable Diffusion on Layouts"
echo "========================================="
echo "Script: ${SCRIPT_PATH}"
echo "Dataset directory: ${DATASET_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Model: ${MODEL_ID}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo ""

# Check required files
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "ERROR: Script not found: ${SCRIPT_PATH}" >&2
    exit 1
fi
if [ ! -d "${DATASET_DIR}" ]; then
    echo "ERROR: Dataset directory not found: ${DATASET_DIR}" >&2
    echo "Run prepare_layout_dataset.sh first!" >&2
    exit 1
fi

# Check if dataset has images
if [ -z "$(ls -A ${DATASET_DIR}/*.png ${DATASET_DIR}/*.jpg 2>/dev/null)" ]; then
    echo "ERROR: No images found in dataset directory: ${DATASET_DIR}" >&2
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
    --dataset_dir "${DATASET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_id "${MODEL_ID}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --num_workers "${NUM_WORKERS}" \
    --save_steps "${SAVE_STEPS}" \
    --device "cuda" \
    --seed 42

echo ""
echo "========================================="
echo "Fine-tuning Completed"
echo "========================================="
echo "Fine-tuned model saved to: ${OUTPUT_DIR}"
echo ""
echo "Next step: Sample from fine-tuned model using:"
echo "  python baseline/sample_finetuned_sd.py \\"
echo "    --model_dir ${OUTPUT_DIR}/pipeline \\"
echo "    --num_samples 64 \\"
echo "    --output_dir outputs/baseline_sd_finetuned_samples"

