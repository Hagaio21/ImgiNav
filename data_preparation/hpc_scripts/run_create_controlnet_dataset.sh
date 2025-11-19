#!/bin/bash
#BSUB -J create_controlnet_dataset
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/create_controlnet_dataset.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/create_controlnet_dataset.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 12:00
#BSUB -q gpul40s

export MKL_INTERFACE_LAYER=LP64
# PyTorch memory management to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
SCRIPT_PATH="${SCRIPT_DIR}/create_controlnet_dataset.py"
# Store for use in job chaining
export PROJECT_ROOT

# --- Input/Output Paths ---
LAYOUTS_MANIFEST="/work3/s233249/ImgiNav/datasets/layouts_cleaned.csv"
AUTOENCODER_CHECKPOINT="/work3/s233249/ImgiNav/experiments/new_layouts/new_layouts_VAE_64x64_structural_256/new_layouts_VAE_64x64_structural_256_checkpoint_best.pt"
TAXONOMY="/work3/s233249/ImgiNav/ImgiNav/config/taxonomy.json"
OUTPUT_MANIFEST="/work3/s233249/ImgiNav/datasets/controlnet_training_manifest.csv"

# --- Parameters ---
POV_BATCH_SIZE=32
GRAPH_MODEL="all-MiniLM-L6-v2"
LAYOUT_BATCH_SIZE=32
NUM_WORKERS=8
HANDLE_SCENES="zero"  # Options: zero (recommended), empty

# --- Check that required files exist ---
echo "[INFO] Checking for required files..."
if [ ! -f "${LAYOUTS_MANIFEST}" ]; then
    echo "[ERROR] Layouts manifest not found: ${LAYOUTS_MANIFEST}"
    exit 1
fi
if [ ! -f "${AUTOENCODER_CHECKPOINT}" ]; then
    echo "[ERROR] Autoencoder checkpoint not found: ${AUTOENCODER_CHECKPOINT}"
    exit 1
fi
if [ ! -f "${TAXONOMY}" ]; then
    echo "[ERROR] Taxonomy file not found: ${TAXONOMY}"
    exit 1
fi
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "[ERROR] Python script not found: ${SCRIPT_PATH}"
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
echo " Creating ControlNet Training Dataset"
echo "=============================================================="
echo " Layouts Manifest:     ${LAYOUTS_MANIFEST}"
echo " Autoencoder Checkpoint: ${AUTOENCODER_CHECKPOINT}"
echo " Taxonomy:              ${TAXONOMY}"
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
echo "[INFO] Starting unified ControlNet dataset creation..."
cd "${PROJECT_ROOT}/ImgiNav"
# Use python -u for unbuffered output
python -u "${SCRIPT_PATH}" \
    --layouts-manifest "${LAYOUTS_MANIFEST}" \
    --autoencoder-checkpoint "${AUTOENCODER_CHECKPOINT}" \
    --taxonomy "${TAXONOMY}" \
    --output "${OUTPUT_MANIFEST}" \
    --pov-batch-size "${POV_BATCH_SIZE}" \
    --graph-model "${GRAPH_MODEL}" \
    --layout-batch-size "${LAYOUT_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --device "cuda" \
    --handle-scenes-without-pov "${HANDLE_SCENES}"

# ----------------------------------------------------------------------
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "=============================================================="
    echo "[DONE] ControlNet dataset creation completed successfully"
    echo "=============================================================="
    echo " Output manifest: ${OUTPUT_MANIFEST}"
    echo "=============================================================="
    
    # Submit next job: Train ControlNet
    echo ""
    echo "[INFO] Submitting next job: Train ControlNet..."
    # Use PROJECT_ROOT that's already defined in this script
    NEXT_SCRIPT="${PROJECT_ROOT}/ImgiNav/training/hpc_scripts/run_train_controlnet_new_layouts.sh"
    if [ -f "${NEXT_SCRIPT}" ]; then
        bsub < "${NEXT_SCRIPT}"
        echo "[INFO] Next job submitted successfully"
    else
        echo "[WARN] Next job script not found: ${NEXT_SCRIPT}"
    fi
else
    echo "=============================================================="
    echo "[ERROR] ControlNet dataset creation failed with exit code: ${EXIT_CODE}"
    echo "=============================================================="
    exit $EXIT_CODE
fi

