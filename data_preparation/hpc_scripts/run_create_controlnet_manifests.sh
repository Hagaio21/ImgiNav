#!/bin/bash
#BSUB -J create_controlnet_manifests
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/create_controlnet_manifests.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/create_controlnet_manifests.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -W 02:00
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
SCRIPT_PATH="${SCRIPT_DIR}/create_controlnet_manifests.py"

# --- Input/Output Paths ---
LAYOUTS_MANIFEST="/work3/s233249/ImgiNav/datasets/layouts_cleaned.csv"
GRAPHS_MANIFEST="/work3/s233249/ImgiNav/datasets/graphs.csv"
POVS_MANIFEST="/work3/s233249/ImgiNav/datasets/povs.csv"
DATASET_DIR="/work3/s233249/ImgiNav/datasets"
OUTPUT_TEX="/work3/s233249/ImgiNav/datasets/controlnet/manifest_tex.csv"
OUTPUT_SEG="/work3/s233249/ImgiNav/datasets/controlnet/manifest_seg.csv"

# --- Check that required files exist ---
echo "[INFO] Checking for required files..."
if [ ! -f "${LAYOUTS_MANIFEST}" ]; then
    echo "[ERROR] Layouts manifest not found: ${LAYOUTS_MANIFEST}"
    exit 1
fi
if [ ! -f "${GRAPHS_MANIFEST}" ]; then
    echo "[ERROR] Graphs manifest not found: ${GRAPHS_MANIFEST}"
    exit 1
fi
if [ ! -f "${POVS_MANIFEST}" ]; then
    echo "[ERROR] POVs manifest not found: ${POVS_MANIFEST}"
    exit 1
fi
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "[ERROR] Python script not found: ${SCRIPT_PATH}"
    exit 1
fi
if [ ! -d "${DATASET_DIR}/controlnet" ]; then
    echo "[ERROR] ControlNet dataset directory not found: ${DATASET_DIR}/controlnet"
    echo "[INFO] Make sure you've run the copy scripts first!"
    exit 1
fi
echo "[INFO] All required files found."

# ----------------------------------------------------------------------
# Job Start
# ----------------------------------------------------------------------
echo "=============================================================="
echo " Creating ControlNet Manifests (tex and seg)"
echo "=============================================================="
echo " Layouts Manifest:        ${LAYOUTS_MANIFEST}"
echo " Graphs Manifest:         ${GRAPHS_MANIFEST}"
echo " POVs Manifest:           ${POVS_MANIFEST}"
echo " Dataset Directory:       ${DATASET_DIR}"
echo " Output Tex Manifest:     ${OUTPUT_TEX}"
echo " Output Seg Manifest:     ${OUTPUT_SEG}"
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
echo "[INFO] Starting manifest creation..."
cd "${PROJECT_ROOT}/ImgiNav"

# Build command
CMD=(
    python -u "${SCRIPT_PATH}"
    --layouts-manifest "${LAYOUTS_MANIFEST}"
    --graphs-manifest "${GRAPHS_MANIFEST}"
    --povs-manifest "${POVS_MANIFEST}"
    --dataset-dir "${DATASET_DIR}"
    --output-tex "${OUTPUT_TEX}"
    --output-seg "${OUTPUT_SEG}"
)

# Execute command
"${CMD[@]}"

# ----------------------------------------------------------------------
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "=============================================================="
    echo "[DONE] ControlNet manifests creation completed successfully"
    echo "=============================================================="
    echo " Tex manifest: ${OUTPUT_TEX}"
    echo " Seg manifest: ${OUTPUT_SEG}"
    echo "=============================================================="
else
    echo "=============================================================="
    echo "[ERROR] Manifest creation failed with exit code: ${EXIT_CODE}"
    echo "=============================================================="
    exit $EXIT_CODE
fi

