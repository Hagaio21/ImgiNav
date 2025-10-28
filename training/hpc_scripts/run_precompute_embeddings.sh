#!/bin/bash
#BSUB -J precompute_latents
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/precompute_latents.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/precompute_latents.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=24000]"
#BSUB -gpu "num=1"
#BSUB -W 10:00
#BSUB -q gpuv100

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/ImgiNav/training/precompute_embeddings.py"

MANIFEST_PATH="/work3/s233249/ImgiNav/datasets/layouts.csv"
OUTPUT_LATENT_DIR="/work3/s233249/ImgiNav/datasets/layout_embeddings"
NEW_MANIFEST_PATH="/work3/s233249/ImgiNav/datasets/layouts_with_embeddings.csv"

AE_CONFIG="${BASE_DIR}/experiments/AEVAE_sweep/AE_large_latent_seg/output/autoencoder_config.yaml"
AE_CKPT="${BASE_DIR}/experiments/AEVAE_sweep/AE_large_latent_seg/checkpoints/ae_latest.pt"

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
echo "Precomputing layout embeddings..."
echo "Manifest: ${MANIFEST_PATH}"
echo "Output:   ${OUTPUT_LATENT_DIR}"
echo "Config:   ${AE_CONFIG}"
echo "Checkpoint: ${AE_CKPT}"
echo "Start: $(date)"
echo "=========================================="

python "${PYTHON_SCRIPT}" \
  --manifest "${MANIFEST_PATH}" \
  --autoencoder_config "${AE_CONFIG}" \
  --autoencoder_ckpt "${AE_CKPT}" \
  --output_latent_dir "${OUTPUT_LATENT_DIR}" \
  --new_manifest "${NEW_MANIFEST_PATH}"

echo "=================
