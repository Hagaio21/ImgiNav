#!/bin/bash
#BSUB -J create_augmentations
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/create_augmentations.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/create_augmentations.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -W 12:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# ----------------------------------------------------------------------
# Configuration
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
SCRIPT_PATH="${BASE_DIR}/data_preparation/create_augmentations.py"
LOG_DIR="${BASE_DIR}/data_preparation/hpc_scripts/logs"

# Input manifest (original dataset)
INPUT_MANIFEST="/work3/s233249/ImgiNav/datasets/layouts_latents.csv"

# Output directory for augmented dataset
OUTPUT_DIR="/work3/s233249/ImgiNav/datasets/augmented"

# Augmentation parameters
USE_MIRROR_ROTATION=true  # Use mirror + 90-degree rotations (7 variants per sample)

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "========================================="
echo "Creating Augmentations (CPU Job)"
echo "========================================="
echo "Script: ${SCRIPT_PATH}"
echo "Input manifest: ${INPUT_MANIFEST}"
echo "Output directory: ${OUTPUT_DIR}"
if [ "${USE_MIRROR_ROTATION}" = "true" ]; then
    echo "Augmentation method: Mirror + 90-degree rotations (7 variants per sample)"
    echo "  - Original + 3 rotations (90°, 180°, 270°)"
    echo "  - Mirror + 3 rotations (mirrored 90°, 180°, 270°)"
else
    echo "Augmentation method: Legacy random augmentation (3 variants per sample, deprecated)"
fi
echo ""

# Check required files
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "ERROR: Script not found: ${SCRIPT_PATH}" >&2
    exit 1
fi

if [ ! -f "${INPUT_MANIFEST}" ]; then
    echo "ERROR: Input manifest not found: ${INPUT_MANIFEST}" >&2
    exit 1
fi

# ----------------------------------------------------------------------
# Conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || conda activate scenefactor || exit 1
fi

# ----------------------------------------------------------------------
# Run augmentation (CPU job, no GPU needed)
cd "${BASE_DIR}"

echo "Starting augmentation process..."
echo "Note: This will filter out white images (>95% whitish pixels) and empty images"
echo ""

# Build command with optional --use-mirror-rotation flag
CMD="python \"${SCRIPT_PATH}\" \
    --dataset-manifest \"${INPUT_MANIFEST}\" \
    --output-dir \"${OUTPUT_DIR}\""

if [ "${USE_MIRROR_ROTATION}" = "true" ]; then
    CMD="${CMD} --use-mirror-rotation"
fi

if [ "${OVERWRITE:-false}" = "true" ]; then
    CMD="${CMD} --overwrite"
fi

eval ${CMD}

echo ""
echo "========================================="
echo "Augmentation creation completed"
echo "========================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "Images manifest: ${OUTPUT_DIR}/manifest_images.csv"
echo ""
echo "Next step: Run GPU job to embed images:"
echo "  bsub < data_preparation/hpc_scripts/embed_images.sh"

