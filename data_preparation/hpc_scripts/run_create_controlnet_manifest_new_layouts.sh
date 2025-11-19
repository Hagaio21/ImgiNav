#!/bin/bash
#BSUB -J controlnet_manifest
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/controlnet_manifest_new_layouts.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/controlnet_manifest_new_layouts.%J.err
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
SCRIPT_PATH="${SCRIPT_DIR}/create_controlnet_manifest_from_joint.py"
# Store for use in job chaining
export PROJECT_ROOT

# --- Input/Output Paths ---
JOINT_MANIFEST="/work3/s233249/ImgiNav/datasets/joint_manifest_with_embeddings.csv"
LAYOUTS_LATENT_MANIFEST="/work3/s233249/ImgiNav/datasets/layouts_cleaned_with_latents.csv"
OUTPUT_MANIFEST="/work3/s233249/ImgiNav/datasets/controlnet_training_manifest_new_layouts.csv"

# --- Parameters ---
# IMPORTANT: Scenes are NEVER skipped - they are always included
# "zero" uses zero POV embedding for scenes (recommended)
# "empty" uses empty string (dataset must handle)
HANDLE_SCENES="zero"  # Options: zero (recommended), empty

# --- Check that required files exist ---
echo "[INFO] Checking for required files..."
if [ ! -f "${JOINT_MANIFEST}" ]; then
    echo "[ERROR] Joint manifest not found: ${JOINT_MANIFEST}"
    exit 1
fi
if [ ! -f "${LAYOUTS_LATENT_MANIFEST}" ]; then
    echo "[ERROR] Layouts latent manifest not found: ${LAYOUTS_LATENT_MANIFEST}"
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
echo " Creating ControlNet Training Manifest"
echo "=============================================================="
echo " Joint Manifest:        ${JOINT_MANIFEST}"
echo " Layouts Latent:        ${LAYOUTS_LATENT_MANIFEST}"
echo " Output Manifest:       ${OUTPUT_MANIFEST}"
echo " Handle Scenes w/o POV: ${HANDLE_SCENES}"
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
    --joint-manifest "${JOINT_MANIFEST}" \
    --layouts-latent-manifest "${LAYOUTS_LATENT_MANIFEST}" \
    --output "${OUTPUT_MANIFEST}" \
    --handle-scenes-without-pov "${HANDLE_SCENES}"

# ----------------------------------------------------------------------
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "=============================================================="
    echo "[DONE] ControlNet manifest creation completed successfully"
    echo "=============================================================="
    echo " Output manifest: ${OUTPUT_MANIFEST}"
    echo "=============================================================="
    echo ""
    echo "[INFO] Pipeline complete! ControlNet training manifest is ready."
    echo "       Next step: Train ControlNet using the manifest"
else
    echo "=============================================================="
    echo "[ERROR] ControlNet manifest creation failed with exit code: ${EXIT_CODE}"
    echo "=============================================================="
    exit $EXIT_CODE
fi

