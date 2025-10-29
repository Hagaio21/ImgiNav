#!/bin/bash
#BSUB -J fix_existing_latents
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/fix_existing_latents.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/fix_existing_latents.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 2:00
#BSUB -q gpuv100

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/ImgiNav/training/fix_embeddings.py"

# Manifest with existing embeddings
MANIFEST_PATH="/work3/s233249/ImgiNav/datasets/layouts_with_embeddings.csv"

# Backup directory for original embeddings (optional but recommended)
BACKUP_DIR="/work3/s233249/ImgiNav/datasets/layout_embeddings_backup"

# Create log directory if it doesn't exist
LOG_DIR="${BASE_DIR}/ImgiNav/training/hpc_scripts/logs"
mkdir -p "${LOG_DIR}"

# =============================================================================
# MODULES
# =============================================================================
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# CONDA ENV
# =============================================================================
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || {
    echo "Failed to activate conda environment 'imginav'" >&2
    conda activate scenefactor || {
      echo "Failed to activate any conda environment" >&2
      exit 1
    }
  }
fi

# =============================================================================
# RUN
# =============================================================================
echo "=========================================="
echo "Fixing existing embeddings (removing batch dimension)..."
echo "Manifest: ${MANIFEST_PATH}"
echo "Backup:   ${BACKUP_DIR}"
echo "Start: $(date)"
echo "=========================================="

# First, copy the fix script to the correct location if it doesn't exist
SCRIPT_SOURCE="/mnt/user-data/outputs/fix_existing_embeddings.py"
if [ -f "${SCRIPT_SOURCE}" ]; then
    cp "${SCRIPT_SOURCE}" "${PYTHON_SCRIPT}"
    echo "Copied fix script from ${SCRIPT_SOURCE}"
fi

# First verify the current state
echo "Verifying current embedding shapes..."
python "${PYTHON_SCRIPT}" \
  --manifest "${MANIFEST_PATH}" \
  --verify_only

# Ask for confirmation (in HPC, we'll skip this and proceed)
echo "Proceeding with fix..."

# Run the fix with backup
python "${PYTHON_SCRIPT}" \
  --manifest "${MANIFEST_PATH}" \
  --backup_dir "${BACKUP_DIR}"

echo "=========================================="
echo "Completed at: $(date)"
echo "Original embeddings backed up to: ${BACKUP_DIR}"
echo "Fixed embeddings remain in original locations"
echo "=========================================="

# Update the diffusion config to use the same manifest
echo ""
echo "IMPORTANT: Your embeddings are now fixed in-place."
echo "Continue using the same manifest: ${MANIFEST_PATH}"
echo "No need to change your diffusion training config."