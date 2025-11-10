#!/bin/bash
# Launch script for creating discriminator dataset with images
# This script creates real and generated images for discriminator training

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Launching Discriminator Images Dataset Creation"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Select real (non-empty, non-augmented) images from manifest"
echo "  2. Copy them to 'real/' directory"
echo "  3. Generate images using diffusion model"
echo "  4. Save generated images to 'generated/' directory"
echo "  5. Create a manifest with valid/invalid labels"
echo ""
echo "Submitting job to V100 queue..."
echo ""

bsub < "${SCRIPT_DIR}/create_discriminator_images.sh"

echo ""
echo "=========================================="
echo "Discriminator images dataset creation job submitted!"
echo "=========================================="
echo ""
echo "Monitor with: bjobs"
echo "Check logs: ${SCRIPT_DIR}/logs/create_discriminator_images.*.out"
echo ""
echo "After completion, the dataset will be at:"
echo "  /work3/s233249/ImgiNav/datasets/discriminator/"
echo ""
echo "Directory structure:"
echo "  discriminator/real/          (real images)"
echo "  discriminator/generated/     (generated images)"
echo "  discriminator/manifest.csv   (manifest with labels)"

