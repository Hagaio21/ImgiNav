#!/bin/bash
#BSUB -J embed_images
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/embed_images.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/embed_images.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 12:00
#BSUB -q gpul40s

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# ----------------------------------------------------------------------
# Configuration
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
SCRIPT_PATH="${BASE_DIR}/data_preparation/preembed_latents.py"
LOG_DIR="${BASE_DIR}/data_preparation/hpc_scripts/logs"

# Autoencoder config and checkpoint (needed to encode images to latents)
AUTOENCODER_CONFIG="/work3/s233249/ImgiNav/ImgiNav/experiments/autoencoders/phase1/phase1_5_AE_final.yaml"
AUTOENCODER_CHECKPOINT="/work3/s233249/ImgiNav/experiments/phase1/phase1_5_AE_final/phase1_5_AE_final_checkpoint_best.pt"

# Images manifest (from augmentation script)
IMAGES_MANIFEST="/work3/s233249/ImgiNav/datasets/augmented/manifest_images.csv"

# Output manifest (final manifest with latent paths)
OUTPUT_MANIFEST="/work3/s233249/ImgiNav/datasets/augmented/manifest.csv"

mkdir -p "${LOG_DIR}"

echo "========================================="
echo "Embedding Images to Latents (GPU Job)"
echo "========================================="
echo "Script: ${SCRIPT_PATH}"
echo "Autoencoder config: ${AUTOENCODER_CONFIG}"
echo "Autoencoder checkpoint: ${AUTOENCODER_CHECKPOINT}"
echo "Images manifest: ${IMAGES_MANIFEST}"
echo "Output manifest: ${OUTPUT_MANIFEST}"
echo ""

# Check required files
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "ERROR: Script not found: ${SCRIPT_PATH}" >&2
    exit 1
fi

if [ ! -f "${AUTOENCODER_CONFIG}" ]; then
    echo "ERROR: Autoencoder config not found: ${AUTOENCODER_CONFIG}" >&2
    exit 1
fi

if [ ! -f "${AUTOENCODER_CHECKPOINT}" ]; then
    echo "ERROR: Autoencoder checkpoint not found: ${AUTOENCODER_CHECKPOINT}" >&2
    exit 1
fi

if [ ! -f "${IMAGES_MANIFEST}" ]; then
    echo "ERROR: Images manifest not found: ${IMAGES_MANIFEST}" >&2
    echo "Hint: Run create_augmentations.sh first to create the images manifest" >&2
    exit 1
fi

# ----------------------------------------------------------------------
# Conda environment
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || conda activate scenefactor || exit 1
fi

# ----------------------------------------------------------------------
# Run embedding (GPU job)
cd "${BASE_DIR}"

echo "Starting embedding process..."
echo "Note: Latents will be saved in datasets/augmented/latents/"
echo ""

python "${SCRIPT_PATH}" \
    --autoencoder-config "${AUTOENCODER_CONFIG}" \
    --autoencoder-checkpoint "${AUTOENCODER_CHECKPOINT}" \
    --dataset-manifest "${IMAGES_MANIFEST}" \
    --output-manifest "${OUTPUT_MANIFEST}" \
    --batch-size 32 \
    --num-workers 8 \
    --device cuda \
    ${OVERWRITE:+"--overwrite"}

echo ""
echo "========================================="
echo "Embedding completed"
echo "========================================="
echo "Output manifest: ${OUTPUT_MANIFEST}"
echo ""
echo "You can now update your training configs to use:"
echo "  manifest: ${OUTPUT_MANIFEST}"

