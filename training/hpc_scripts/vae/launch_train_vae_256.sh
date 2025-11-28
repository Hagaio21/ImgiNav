#!/bin/bash
# =============================================================================
# Launch VAE Training Jobs - 256x256
# =============================================================================
# This script submits both VAE training jobs (seg and tex) to the HPC queue
# Usage:
#   ./launch_train_vae_256.sh              # Submit both jobs
#   ./launch_train_vae_256.sh seg          # Submit only segmented VAE
#   ./launch_train_vae_256.sh tex          # Submit only textured VAE
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Job scripts
SEG_SCRIPT="${SCRIPT_DIR}/run_train_vae_seg_256.sh"
TEX_SCRIPT="${SCRIPT_DIR}/run_train_vae_tex_256.sh"

# Check which variant to run (default: both)
VARIANT="${1:-both}"

echo "=========================================="
echo "Launching VAE Training Jobs (256x256)"
echo "=========================================="
echo "Base directory: ${BASE_DIR}"
echo "Variant: ${VARIANT}"
echo "=========================================="
echo ""

# Validate scripts exist
if [ ! -f "${SEG_SCRIPT}" ]; then
  echo "ERROR: Segmented VAE script not found: ${SEG_SCRIPT}" >&2
  exit 1
fi

if [ ! -f "${TEX_SCRIPT}" ]; then
  echo "ERROR: Textured VAE script not found: ${TEX_SCRIPT}" >&2
  exit 1
fi

# Make scripts executable
chmod +x "${SEG_SCRIPT}"
chmod +x "${TEX_SCRIPT}"

# Submit jobs
case "${VARIANT}" in
  seg)
    echo "Submitting segmented VAE training job..."
    bsub < "${SEG_SCRIPT}"
    echo "Job submitted! Check status with: bjobs"
    ;;
  tex)
    echo "Submitting textured VAE training job..."
    bsub < "${TEX_SCRIPT}"
    echo "Job submitted! Check status with: bjobs"
    ;;
  both)
    echo "Submitting segmented VAE training job..."
    bsub < "${SEG_SCRIPT}"
    SEG_JOB_ID=$?
    
    echo ""
    echo "Submitting textured VAE training job..."
    bsub < "${TEX_SCRIPT}"
    TEX_JOB_ID=$?
    
    echo ""
    echo "=========================================="
    echo "Both jobs submitted!"
    echo "=========================================="
    echo "Check job status with: bjobs"
    echo "Monitor logs in: ${BASE_DIR}/training/hpc_scripts/logs/"
    echo "=========================================="
    ;;
  *)
    echo "ERROR: Invalid variant '${VARIANT}'. Use: seg, tex, or both" >&2
    exit 1
    ;;
esac

echo ""
echo "Done!"

