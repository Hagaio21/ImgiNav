#!/bin/bash
# Launch script for debugging diffusion models (all 3 sizes)
# Submits debug job to HPC queue

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
DEBUG_SCRIPT="${SCRIPT_DIR}/debug_diffusion.sh"

# Create logs directory if it doesn't exist
mkdir -p "${BASE_DIR}/training/hpc_scripts/logs"

echo "=============================================================================="
echo "Submitting Debug Job for Diffusion Models"
echo "=============================================================================="
echo "This will run debug tests for:"
echo "  - Small bottleneck model"
echo "  - Medium bottleneck model"
echo "  - Large bottleneck model"
echo ""
echo "Debug tests include:"
echo "  1. VAE Round Trip (encode/decode test)"
echo "  2. Noise Schedule (forward process visualization)"
echo "  3. Overfit Test (single batch training)"
echo "=============================================================================="

# Submit job to HPC queue
# Using gpul40s queue with 1 GPU, moderate resources for debugging
bsub -J "debug_diffusion" \
    -o "${BASE_DIR}/training/hpc_scripts/logs/debug_diffusion.%J.out" \
    -e "${BASE_DIR}/training/hpc_scripts/logs/debug_diffusion.%J.err" \
    -n 2 \
    -R "rusage[mem=16000]" \
    -gpu "num=1" \
    -W 4:00 \
    -q gpul40s \
    bash "${DEBUG_SCRIPT}"

echo ""
echo "Job submitted! Check logs at:"
echo "  Output: ${BASE_DIR}/training/hpc_scripts/logs/debug_diffusion.%J.out"
echo "  Error:  ${BASE_DIR}/training/hpc_scripts/logs/debug_diffusion.%J.err"
echo ""
echo "Debug results will be saved to: ${BASE_DIR}/debug_outputs/"

