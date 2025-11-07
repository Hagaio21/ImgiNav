#!/bin/bash
# Launch script for creating discriminator dataset
# This script creates real and bad latents for discriminator training

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Launching Discriminator Dataset Creation"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Select 5000 real (non-augmented) images from manifest"
echo "  2. Encode them to get real latents"
echo "  3. Generate 5000 'bad' latents from Stage 2 diffusion model"
echo "  4. Create a manifest with good/bad labels"
echo ""
echo "Submitting job to V100 queue..."
echo ""

bsub < "${SCRIPT_DIR}/create_discriminator_dataset.sh"

echo ""
echo "=========================================="
echo "Discriminator dataset creation job submitted!"
echo "=========================================="
echo ""
echo "Monitor with: bjobs"
echo "Check logs: ${SCRIPT_DIR}/logs/create_discriminator_dataset.*.out"
echo ""
echo "After completion, train discriminator:"
echo "  python training/train_discriminator.py \\"
echo "    --real_latents /work3/s233249/ImgiNav/datasets/discriminator_dataset/real_latents_all.pt \\"
echo "    --fake_latents /work3/s233249/ImgiNav/datasets/discriminator_dataset/bad_latents_all.pt \\"
echo "    --output_dir /work3/s233249/ImgiNav/experiments/discriminator/discriminator_unet128_d4"

