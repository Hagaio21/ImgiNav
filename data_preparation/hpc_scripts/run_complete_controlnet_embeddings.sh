#!/bin/bash
#BSUB -J complete_controlnet_embeddings
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/complete_controlnet_embeddings.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/complete_controlnet_embeddings.%J.err
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
SCRIPT_PATH="${SCRIPT_DIR}/complete_controlnet_embeddings.py"
# Store for use in job chaining
export PROJECT_ROOT

# --- Input/Output Paths ---
INPUT_MANIFEST="/work3/s233249/ImgiNav/datasets/controlnet_training_manifest.csv"
OUTPUT_MANIFEST="/work3/s233249/ImgiNav/datasets/controlnet_training_manifest_complete.csv"
TAXONOMY="/work3/s233249/ImgiNav/ImgiNav/config/taxonomy.json"

# Autoencoder checkpoint (required for layout embeddings)
AUTOENCODER_CHECKPOINT="/work3/s233249/ImgiNav/experiments/new_layouts/new_layouts_VAE_64x64_structural_256/new_layouts_VAE_64x64_structural_256_checkpoint_best.pt"
AUTOENCODER_CONFIG=""  # Optional - will try to find from checkpoint path if not provided

# --- Parameters ---
BATCH_SIZE=32
NUM_WORKERS=8
DEVICE="cuda"
OVERWRITE=false  # Set to true to overwrite existing embedding files

# --- Check that required files exist ---
echo "[INFO] Checking for required files..."
if [ ! -f "${INPUT_MANIFEST}" ]; then
    echo "[ERROR] Input manifest not found: ${INPUT_MANIFEST}"
    exit 1
fi
if [ ! -f "${TAXONOMY}" ]; then
    echo "[ERROR] Taxonomy file not found: ${TAXONOMY}"
    exit 1
fi
if [ ! -f "${AUTOENCODER_CHECKPOINT}" ]; then
    echo "[ERROR] Autoencoder checkpoint not found: ${AUTOENCODER_CHECKPOINT}"
    exit 1
fi
if [ -n "${AUTOENCODER_CONFIG}" ] && [ ! -f "${AUTOENCODER_CONFIG}" ]; then
    echo "[WARNING] Autoencoder config not found: ${AUTOENCODER_CONFIG}"
    echo "[INFO] Will try to find config from checkpoint path (config may be embedded)"
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
echo " Completing Missing ControlNet Embeddings"
echo "=============================================================="
echo " Input Manifest:           ${INPUT_MANIFEST}"
echo " Output Manifest:          ${OUTPUT_MANIFEST}"
echo " Taxonomy:                 ${TAXONOMY}"
echo " Autoencoder Checkpoint:   ${AUTOENCODER_CHECKPOINT}"
if [ -n "${AUTOENCODER_CONFIG}" ]; then
    echo " Autoencoder Config:       ${AUTOENCODER_CONFIG}"
else
    echo " Autoencoder Config:       (will auto-detect from checkpoint path)"
fi
echo " Batch Size:               ${BATCH_SIZE}"
echo " Num Workers:              ${NUM_WORKERS}"
echo " Device:                   ${DEVICE}"
echo " Overwrite:                ${OVERWRITE}"
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
echo "[INFO] Starting embedding completion..."
cd "${PROJECT_ROOT}/ImgiNav"
# Use python -u for unbuffered output

# Build command
CMD=(
    python -u "${SCRIPT_PATH}"
    --manifest "${INPUT_MANIFEST}"
    --output "${OUTPUT_MANIFEST}"
    --taxonomy "${TAXONOMY}"
    --autoencoder-checkpoint "${AUTOENCODER_CHECKPOINT}"
)

# Add autoencoder config if provided
if [ -n "${AUTOENCODER_CONFIG}" ]; then
    CMD+=(--autoencoder-config "${AUTOENCODER_CONFIG}")
fi

# Add remaining arguments
CMD+=(
    --device "${DEVICE}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
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
    echo "[DONE] Embedding completion finished successfully"
    echo "=============================================================="
    echo " Output manifest: ${OUTPUT_MANIFEST}"
    echo "=============================================================="
    
    # Optional: Submit next job (uncomment if needed)
    # echo ""
    # echo "[INFO] Submitting next job: Train ControlNet..."
    # NEXT_SCRIPT="${PROJECT_ROOT}/ImgiNav/training/hpc_scripts/run_train_controlnet_new_layouts.sh"
    # if [ -f "${NEXT_SCRIPT}" ]; then
    #     bsub < "${NEXT_SCRIPT}"
    #     echo "[INFO] Next job submitted successfully"
    # else
    #     echo "[WARN] Next job script not found: ${NEXT_SCRIPT}"
    # fi
else
    echo "=============================================================="
    echo "[ERROR] Embedding completion failed with exit code: ${EXIT_CODE}"
    echo "=============================================================="
    exit $EXIT_CODE
fi

