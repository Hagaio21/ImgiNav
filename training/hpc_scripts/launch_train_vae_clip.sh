#!/bin/bash
# Launch script for VAE Training with CLIP Loss
# This script submits the VAE training job to the HPC cluster

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# =============================================================================
# CONFIGURATION
# =============================================================================
TRAIN_SCRIPT="${SCRIPT_DIR}/run_train_vae_clip.sh"

# =============================================================================
# VALIDATION
# =============================================================================
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "ERROR: Training script not found: ${TRAIN_SCRIPT}" >&2
    exit 1
fi

# Make script executable
chmod +x "${TRAIN_SCRIPT}"

# Verify the script is actually executable
if [ ! -x "${TRAIN_SCRIPT}" ]; then
    echo "ERROR: Training script is not executable: ${TRAIN_SCRIPT}" >&2
    exit 1
fi

# =============================================================================
# MAIN
# =============================================================================
echo "=============================================================================="
echo "Launching VAE Training with CLIP Loss"
echo "=============================================================================="
echo ""
echo "Script directory: ${SCRIPT_DIR}"
echo "Base directory: ${BASE_DIR}"
echo ""
echo "This will train a VAE with CLIP loss to create a joint embedding space."
echo "The trained VAE will have better alignment with text/POV embeddings."
echo ""
echo "Config: experiments/autoencoders/new_layouts/new_layouts_VAE_32x32_structural_256_clip.yaml"
echo ""
echo "=============================================================================="
echo ""

# Prompt for confirmation
read -p "Submit VAE training job? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# =============================================================================
# SUBMIT JOB
# =============================================================================
echo ""
echo "Submitting VAE training job..."
echo "=============================================================================="

JOB_OUTPUT=$(bsub -J "train_vae_clip" \
    -o "${BASE_DIR}/training/hpc_scripts/logs/train_vae_clip.%J.out" \
    -e "${BASE_DIR}/training/hpc_scripts/logs/train_vae_clip.%J.err" \
    -n 8 \
    -R "rusage[mem=16000]" \
    -gpu "num=1" \
    -W 48:00 \
    -q gpuv100 \
    bash "${TRAIN_SCRIPT}" 2>&1)

BSUB_EXIT_CODE=$?
if [ $BSUB_EXIT_CODE -eq 0 ]; then
    # Extract job ID from bsub output
    JOB_ID=$(echo "${JOB_OUTPUT}" | grep -oP 'Job <\K[0-9]+(?=>)' || echo "")
    if [ -n "${JOB_ID}" ]; then
        echo "SUCCESS - Job submitted (Job ID: ${JOB_ID})"
        echo ""
        echo "Monitor job with: bjobs ${JOB_ID}"
        echo "Check logs in: ${BASE_DIR}/training/hpc_scripts/logs/"
        echo ""
        echo "Once complete, the VAE checkpoint will be saved and can be used for diffusion training."
    else
        if echo "${JOB_OUTPUT}" | grep -qi "submitted"; then
            echo "SUBMITTED (could not extract job ID from: ${JOB_OUTPUT})"
        else
            echo "FAILED - bsub output: ${JOB_OUTPUT}"
            exit 1
        fi
    fi
else
    echo "FAILED (bsub exit code: ${BSUB_EXIT_CODE})"
    echo "bsub output: ${JOB_OUTPUT}"
    exit 1
fi

echo "=============================================================================="

