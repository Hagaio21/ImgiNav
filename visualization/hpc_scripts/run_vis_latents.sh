#!/bin/bash
#BSUB -J visualize_latents
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/visualization/hpc_scripts/logs/vis_latents.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/visualization/hpc_scripts/logs/vis_latents.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -W 4:00
#BSUB -q gpuv100

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# ----------------------------------------------------------------------
# Configuration
# --- Path to the visualization script ---
SCRIPT_PATH="/work3/s233249/ImgiNav/ImgiNav/visualization/visualize_latents.py"
# --- Path to the project root (for imports) ---
PROJECT_ROOT="/work3/s233249/ImgiNav/ImgiNav"

# --- Dataset configuration ---
MANIFEST="/work3/s233249/ImgiNav/datasets/layouts.csv"
LABEL_COL="room_id"       # <-- UPDATED based on your manifest
CATEGORY_COL="type"       # <-- UPDATED based on your manifest
OUTPUT_DIR="/work3/s233249/ImgiNav/ImgiNav/visualization/latent_visuals"

# --- Model A (Baseline: MSE-Only) ---
# !!! UPDATE THESE PATHS to point to your trained MSE-only model !!!
MODEL_A_CONFIG="/work3/s233249/ImgiNav/experiments/VAE/mse_loss/VAE_512_64x64x4/experiment_config.yaml"
MODEL_A_CKPT="/work3/s233249/ImgiNav/experiments/VAE/mse_loss/VAE_512_64x64x4/checkpoints/ae_latest.pt"

# --- Model B (New: Segmentation Loss) ---
MODEL_B_CONFIG="/work3/s233249/ImgiNav/experiments/VAE/cross_enth_loss/VAE_512_64x64x4/output/experiment_config.yaml"
MODEL_B_CKPT="/work3/s233249/ImgiNav/experiments/VAE/cross_enth_loss/VAE_512_64x64x4/checkpoints/ae_epoch_40.pt"

echo "Running visualize_latents job"
echo "Comparing Model A: ${MODEL_A_CONFIG}"
echo "With Model B: ${MODEL_B_CONFIG}"
echo "Dataset: ${MANIFEST}"
echo "Outputting plots to: ${OUTPUT_DIR}"

# ----------------------------------------------------------------------
# Conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || true
fi

# ----------------------------------------------------------------------
# Run visualization
# --no_filter_empty is NOT set, so empty layouts will be filtered out by default.
python "${SCRIPT_PATH}" \
  --project_root "${PROJECT_ROOT}" \
  --manifest "${MANIFEST}" \
  --label_col "${LABEL_COL}" \
  --category_col "${CATEGORY_COL}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_a_config "${MODEL_A_CONFIG}" \
  --model_a_ckpt "${MODEL_A_CKPT}" \
  --model_b_config "${MODEL_B_CONFIG}" \
  --model_b_ckpt "${MODEL_B_CKPT}"

echo "Visualization completed successfully"