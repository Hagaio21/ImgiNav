#!/bin/bash
#BSUB -J copy_layouts_controlnet
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/copy_layouts_controlnet.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/copy_layouts_controlnet.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -W 04:00
#BSUB -q hpc

export MKL_INTERFACE_LAYER=LP64
set -euo pipefail

# ----------------------------------------------------------------------
# Create Log Directory
# ----------------------------------------------------------------------
LOG_DIR="/work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs"
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
SCRIPT_DIR="${PROJECT_ROOT}/ImgiNav/data_preparation"
SCRIPT_PATH="${SCRIPT_DIR}/copy_layouts_to_controlnet.py"

# --- Input/Output Paths ---
LAYOUTS_MANIFEST="/work3/s233249/ImgiNav/datasets/layouts_cleaned.csv"
DATASET_DIR="/work3/s233249/ImgiNav/datasets"

# --- Parameters ---
OVERWRITE=false  # Set to true to overwrite existing files

# --- Check that required files exist ---
echo "[INFO] Checking for required files..."
if [ ! -f "${LAYOUTS_MANIFEST}" ]; then
    echo "[ERROR] Layouts manifest not found: ${LAYOUTS_MANIFEST}"
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
echo " Copying Layouts to ControlNet Dataset"
echo "=============================================================="
echo " Layouts Manifest:        ${LAYOUTS_MANIFEST}"
echo " Dataset Directory:       ${DATASET_DIR}"
echo " Overwrite:               ${OVERWRITE}"
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
echo "[INFO] Starting layout copying..."
cd "${PROJECT_ROOT}/ImgiNav"

# Build command
CMD=(
    python -u "${SCRIPT_PATH}"
    --layouts-manifest "${LAYOUTS_MANIFEST}"
    --dataset-dir "${DATASET_DIR}"
)

# Add overwrite flag if enabled
if [ "${OVERWRITE}" = "true" ]; then
    CMD+=(--overwrite)
fi

# Execute command
"${CMD[@]}"

# ----------------------------------------------------------------------
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "=============================================================="
    echo "[DONE] Layout copying completed successfully"
    echo "=============================================================="
    echo " Dataset directory: ${DATASET_DIR}/controlnet"
    echo "=============================================================="
else
    echo "=============================================================="
    echo "[ERROR] Layout copying failed with exit code: ${EXIT_CODE}"
    echo "=============================================================="
    exit $EXIT_CODE
fi

