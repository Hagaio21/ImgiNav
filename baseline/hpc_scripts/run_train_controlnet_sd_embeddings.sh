#!/bin/bash
#BSUB -J train_controlnet_sd
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/baseline/hpc_scripts/logs/train_controlnet_sd.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/baseline/hpc_scripts/logs/train_controlnet_sd.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=32000]"
#BSUB -gpu "num=1"
#BSUB -W 48:00
#BSUB -q gpuv100

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
SCRIPT_PATH="${BASE_DIR}/baseline/train_controlnet_sd_embeddings.py"
LOG_DIR="${BASE_DIR}/baseline/hpc_scripts/logs"

# Config file
CONFIG_PATH="${BASE_DIR}/experiments/controlnet/sd_finetuned_controlnet.yaml"

# Fine-tuned UNet path (update after SD fine-tuning completes)
FINETUNED_UNET_PATH="${BASE_DIR}/outputs/baseline_sd_finetuned_full/checkpoint-best/unet"

mkdir -p "${LOG_DIR}"

echo "========================================="
echo "Train ControlNet with Fine-tuned SD UNet"
echo "========================================="
echo "Script: ${SCRIPT_PATH}"
echo "Config: ${CONFIG_PATH}"
echo "Fine-tuned UNet: ${FINETUNED_UNET_PATH}"
echo ""

# Check required files
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "ERROR: Script not found: ${SCRIPT_PATH}" >&2
    exit 1
fi
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "ERROR: Config not found: ${CONFIG_PATH}" >&2
    exit 1
fi
if [ ! -d "${FINETUNED_UNET_PATH}" ]; then
    echo "ERROR: Fine-tuned UNet not found: ${FINETUNED_UNET_PATH}" >&2
    echo "Please run SD fine-tuning first!" >&2
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
# Set Hugging Face cache to /work3
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
    --config "${CONFIG_PATH}" \
    --finetuned_unet_path "${FINETUNED_UNET_PATH}" \
    --device "cuda"

echo ""
echo "========================================="
echo "Training Completed"
echo "========================================="
echo "Checkpoints saved to: experiments/controlnet/sd_finetuned/"
echo ""

