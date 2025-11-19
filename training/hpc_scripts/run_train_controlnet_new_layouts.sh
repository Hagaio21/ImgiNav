#!/bin/bash
#BSUB -J train_controlnet_new_layouts
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/train_controlnet_new_layouts.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/train_controlnet_new_layouts.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 48:00
#BSUB -q gpuv100

export MKL_INTERFACE_LAYER=LP64
# PyTorch memory management to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail

# ----------------------------------------------------------------------
# Create Log Directory
# ----------------------------------------------------------------------
LOG_DIR="/work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs"
mkdir -p "${LOG_DIR}"

# --- Log that the script has started ---
echo "[INFO] LSF Job $LSB_JOBID started on $(hostname)."
echo "[INFO] Log directory ${LOG_DIR} ensured."

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
echo "[INFO] Setting up paths..."

# --- Base Project Paths ---
PROJECT_ROOT="/work3/s233249/ImgiNav"
SCRIPT_DIR="${PROJECT_ROOT}/ImgiNav/training"
SCRIPT_PATH="${SCRIPT_DIR}/train_controlnet.py"
CONFIG_PATH="${PROJECT_ROOT}/ImgiNav/experiments/controlnet/new_layouts/controlnet_unet48_d4_new_layouts.yaml"
# Store for use in job chaining
export PROJECT_ROOT

# --- Check that required files exist ---
echo "[INFO] Checking for required files..."
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "[ERROR] Training script not found: ${SCRIPT_PATH}"
    exit 1
fi
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "[ERROR] Config file not found: ${CONFIG_PATH}"
    exit 1
fi
echo "[INFO] All required files found."

# ----------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X

# ----------------------------------------------------------------------
# Job Start
# ----------------------------------------------------------------------
echo "=============================================================="
echo " Training ControlNet Model"
echo "=============================================================="
echo " Config: ${CONFIG_PATH}"
echo " Script: ${SCRIPT_PATH}"
echo "=============================================================="

# ----------------------------------------------------------------------
# Conda environment
# ----------------------------------------------------------------------
echo "[INFO] Activating Conda environment..."
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || {
        echo "[ERROR] Failed to activate conda env 'imginav'" >&2
        exit 1
    }
    echo "[INFO] Conda environment 'imginav' activated."
    echo "[INFO] Python path: $(which python)"
else
    echo "[WARN] miniconda3 not found. Assuming environment is already active."
fi
echo "[CONDA] Environment: ${CONDA_DEFAULT_ENV:-none}"
echo ""

# ----------------------------------------------------------------------
# Run script
# ----------------------------------------------------------------------
echo "[INFO] Starting ControlNet training..."
cd "${PROJECT_ROOT}/ImgiNav"
# Use python -u for unbuffered output
python -u "${SCRIPT_PATH}" \
    --config "${CONFIG_PATH}"

# ----------------------------------------------------------------------
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "=============================================================="
    echo "[DONE] ControlNet training completed successfully"
    echo "=============================================================="
    echo " Config: ${CONFIG_PATH}"
    echo "=============================================================="
    echo ""
    echo "[INFO] ControlNet training pipeline complete!"
else
    echo "=============================================================="
    echo "[ERROR] ControlNet training failed with exit code: ${EXIT_CODE}"
    echo "=============================================================="
    exit $EXIT_CODE
fi

