#!/bin/bash
# Evaluation script for Medium CLIP Diffusion - Scenes
# Uses DDPM sampling from best checkpoint

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
CONFIG="${BASE_DIR}/experiments/diffusion/clip/regular_scenes/medium_bottleneck.yaml"

cd "${BASE_DIR}"

echo "=============================================================================="
echo "Evaluating Medium CLIP Diffusion - Scenes"
echo "=============================================================================="
echo "Config: ${CONFIG}"
echo "Date: $(date)"
echo "=============================================================================="

python scripts/evaluate_diffusion.py \
    "${CONFIG}" \
    --num_samples 64 \
    --batch_size 16 \
    --unconditional_samples 32 \
    --seed 42

echo "=============================================================================="
echo "Evaluation complete!"
echo "=============================================================================="

