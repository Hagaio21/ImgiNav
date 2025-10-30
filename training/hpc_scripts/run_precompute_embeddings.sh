#!/bin/bash
#BSUB -J recompute_latents_robust
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/recompute_latents_robust.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/recompute_latents_robust.%J.err
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
PYTHON_SCRIPT="${BASE_DIR}/ImgiNav/data_preperation/create_embeddings.py"

# Input/Output paths
MANIFEST_PATH="/work3/s233249/ImgiNav/datasets/layouts.csv"
OUTPUT_LATENT_DIR="/work3/s233249/ImgiNav/datasets/layout_embeddings_fixed"
NEW_MANIFEST_PATH="/work3/s233249/ImgiNav/datasets/layouts_with_embeddings_fixed.csv"

# AutoEncoder paths
AE_CONFIG="${BASE_DIR}/experiments/AEVAE_sweep/AE_large_latent_seg/output/autoencoder_config.yaml"
AE_CKPT="${BASE_DIR}/experiments/AEVAE_sweep/AE_large_latent_seg/checkpoints/ae_latest.pt"

# Create log directory if it doesn't exist
LOG_DIR="${BASE_DIR}/ImgiNav/training/hpc_scripts/logs"
mkdir -p "${LOG_DIR}"

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
echo "Recomputing layout embeddings (robust version)..."
echo "Manifest: ${MANIFEST_PATH}"
echo "Output:   ${OUTPUT_LATENT_DIR}"
echo "Config:   ${AE_CONFIG}"
echo "Checkpoint: ${AE_CKPT}"
echo "Start: $(date)"
echo "=========================================="

# Create output directory
mkdir -p "${OUTPUT_LATENT_DIR}"

# Run the embedding creation using data preparation script
python "${PYTHON_SCRIPT}" \
  --type layout \
  --config "${AE_CONFIG}" \
  --ckpt "${AE_CKPT}" \
  --data_root "/work3/s233249/ImgiNav/datasets" \
  --output "${OUTPUT_LATENT_DIR}" \
  --manifest_out "layouts_with_embeddings_fixed.csv" \
  --batch_size 32 \
  --format pt \
  --overwrite

echo "=========================================="
echo "Completed at: $(date)"
echo "New manifest saved to: ${NEW_MANIFEST_PATH}"
echo "Embeddings saved to: ${OUTPUT_LATENT_DIR}"
echo "Check encoding log at: ${OUTPUT_LATENT_DIR}/encoding_log.txt"
echo "=========================================="