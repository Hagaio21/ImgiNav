#!/bin/bash
# Launch script for ControlNet Dataset Embedding
# This script submits the embedding job to the HPC cluster
#
# The embedding creates a shared manifest that all experiments can use:
# - latent_path: Layout embeddings from VAE
# - pov_embedding_path: POV embeddings from ResNet18
# - graph_embedding_path: Graph embeddings from SentenceTransformer

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# =============================================================================
# CONFIGURATION
# =============================================================================
EMBEDDING_SCRIPT="${SCRIPT_DIR}/run_embed_controlnet_dataset.sh"

# =============================================================================
# VALIDATION
# =============================================================================
if [ ! -f "${EMBEDDING_SCRIPT}" ]; then
    echo "ERROR: Embedding script not found: ${EMBEDDING_SCRIPT}" >&2
    exit 1
fi

# Make script executable
chmod +x "${EMBEDDING_SCRIPT}"

# Verify the script is actually executable
if [ ! -x "${EMBEDDING_SCRIPT}" ]; then
    echo "ERROR: Embedding script is not executable: ${EMBEDDING_SCRIPT}" >&2
    exit 1
fi

# =============================================================================
# MAIN
# =============================================================================
echo "=============================================================================="
echo "Launching ControlNet Dataset Embedding"
echo "=============================================================================="
echo ""
echo "Script directory: ${SCRIPT_DIR}"
echo "Base directory: ${BASE_DIR}"
echo ""
echo "This will create a shared embedding manifest that all experiments can use."
echo "Output location: /work3/s233249/ImgiNav/experiments/shared_embeddings/manifest_with_embeddings.csv"
echo ""
echo "=============================================================================="
echo ""

# Prompt for confirmation
read -p "Submit embedding job? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# =============================================================================
# SUBMIT JOB
# =============================================================================
echo ""
echo "Submitting embedding job..."
echo "=============================================================================="

JOB_OUTPUT=$(bsub -J "embed_controlnet_dataset" \
    -o "${BASE_DIR}/training/hpc_scripts/logs/embed_controlnet_dataset.%J.out" \
    -e "${BASE_DIR}/training/hpc_scripts/logs/embed_controlnet_dataset.%J.err" \
    -n 8 \
    -R "rusage[mem=16000]" \
    -gpu "num=1" \
    -W 24:00 \
    -q gpuv100 \
    bash "${EMBEDDING_SCRIPT}" 2>&1)

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
        echo "Once complete, all experiments can use the shared embedding manifest."
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

