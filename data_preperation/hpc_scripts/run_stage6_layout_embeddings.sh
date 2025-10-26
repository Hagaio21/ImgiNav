#!/bin/bash
#BSUB -J stage6
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs/stage6_scene_graphs.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs/stage6_scene_graphs.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=8000]"
#BSUB -W 06:00
#BSUB -q gpul40s

# ----------------------------------------------------------------------
# Create Log Directory
# ----------------------------------------------------------------------
# This path MUST EXACTLY match the paths in the #BSUB header above.
LOG_DIR="/work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs"
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
SCRIPT_DIR="${PROJECT_ROOT}/ImgiNav/data_preperation"
SCRIPT_PATH="${SCRIPT_DIR}/stage6_create_layout_embeddings.py"
DATA_ROOT="${PROJECT_ROOT}/datasets/scenes"

# --- Job Parameters ---
BATCH_SIZE=8
DEVICE="cuda"
export MKL_INTERFACE_LAYER=LP64

# ----------------------------------------------------------------------
# Argument Check
# ----------------------------------------------------------------------
# We check for the argument manually, *after* the first echo commands
if [ "$#" -ne 1 ]; then
    echo "[ERROR] Usage: $0 <EXPERIMENT_DIRECTORY>"
    echo "  (This script infers config and checkpoint paths from that dir)"
    echo
    echo "Example: $0 /work3/s233249/ImgiNav/experiments/VAE/VAE_512_32x32x4_SegLoss"
    exit 1
fi

# --- Inferred Experiment Paths (from 1st arg) ---
EXP_DIR="$1"
CONFIG="${EXP_DIR}/output/experiment_config.yaml"
CKPT="${EXP_DIR}/checkpoints/ae_latest.pt"

# --- Check that required files exist ---
echo "[INFO] Checking for required files..."
if [ ! -f "${CONFIG}" ]; then
    echo "[ERROR] Config file not found: ${CONFIG}"
    exit 1
fi
if [ ! -f "${CKPT}" ]; then
    echo "[ERROR] Checkpoint file not found: ${CKPT}"
    exit 1
fi
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "[ERROR] Python script not found: ${SCRIPT_PATH}"
    exit 1
fi
echo "[INFO] All required files found."

# ----------------------------------------------------------------------
# Job Start
# ----------------------------------------------------------------------
echo "=============================================================="
echo " Running AutoEncoder embedding job"
echo "=============================================================="
echo " Experiment Dir: ${EXP_DIR}"
echo " Config:         ${CONFIG}"
echo " Checkpoint:     ${CKPT}"
echo " Data root:      ${DATA_ROOT}"
echo " Device:         ${DEVICE}"
echo " Log Directory:  ${LOG_DIR}"
echo "=============================================================="

# ----------------------------------------------------------------------
# Conda environment
# ----------------------------------------------------------------------
echo "[INFO] Activating Conda environment..."
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav
    echo "[INFO] Conda environment 'imginav' activated."
    echo "[INFO] Python path: $(which python)"
else
    echo "[WARN] miniconda3 not found. Assuming environment is already active."
fi

# ----------------------------------------------------------------------
# Run embedding
# ----------------------------------------------------------------------
echo "[INFO] Starting Python script..."
# Use python -u for unbuffered output
python -u "${SCRIPT_PATH}" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --data_root "${DATA_ROOT}" \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}"

# ----------------------------------------------------------------------
echo "=============================================================="
echo "[DONE] AutoEncoder embedding completed successfully"
echo "=============================================================="