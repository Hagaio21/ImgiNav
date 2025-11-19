#!/bin/bash
#BSUB -J joint_manifest
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/joint_manifest.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/joint_manifest.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -W 06:00
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
SCRIPT_PATH="${SCRIPT_DIR}/create_joint_manifest.py"

# --- Input/Output Paths ---
LAYOUTS_MANIFEST="/work3/s233249/ImgiNav/datasets/layouts_cleaned.csv"
DATA_ROOT="/work3/s233249/ImgiNav/datasets/scenes"
OUTPUT_MANIFEST="/work3/s233249/ImgiNav/datasets/joint_manifest.csv"
OUTPUT_DIR="/work3/s233249/ImgiNav/datasets/collected"

# --- Check that required files exist ---
echo "[INFO] Checking for required files..."
if [ ! -f "${LAYOUTS_MANIFEST}" ]; then
    echo "[ERROR] Layouts manifest not found: ${LAYOUTS_MANIFEST}"
    exit 1
fi
if [ ! -d "${DATA_ROOT}" ]; then
    echo "[ERROR] Data root directory not found: ${DATA_ROOT}"
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
echo " Creating Joint Manifest"
echo "=============================================================="
echo " Layouts Manifest: ${LAYOUTS_MANIFEST}"
echo " Data Root:        ${DATA_ROOT}"
echo " Output Manifest:  ${OUTPUT_MANIFEST}"
echo " Output Dir:       ${OUTPUT_DIR}"
echo " Log Directory:    ${LOG_DIR}"
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
echo "[INFO] Starting Python script..."
# Use python -u for unbuffered output
python -u "${SCRIPT_PATH}" \
    --layouts-manifest "${LAYOUTS_MANIFEST}" \
    --data-root "${DATA_ROOT}" \
    --output "${OUTPUT_MANIFEST}" \
    --output-dir "${OUTPUT_DIR}"

# ----------------------------------------------------------------------
echo "=============================================================="
echo "[DONE] Joint manifest creation completed successfully"
echo "=============================================================="
echo " Output manifest: ${OUTPUT_MANIFEST}"
echo " Graphs copied to: ${OUTPUT_DIR}/graphs"
echo " Textured POVs copied to: ${OUTPUT_DIR}/povs/tex"
echo " Segmented POVs copied to: ${OUTPUT_DIR}/povs/seg"
echo "=============================================================="

