#!/bin/bash
#BSUB -J generate_embeddings
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/visualization/hpc_scripts/logs/gen_embeddings.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/visualization/hpc_scripts/logs/gen_embeddings.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -W 6:00
#BSUB -q gpuv100

# ----------------------------------------------------------------------
# Configuration
# --- Paths ---
SCRIPT_PATH="/work3/s233249/ImgiNav/ImgiNav/visualization/generate_embeddings.py"
PROJECT_ROOT="/work3/s233249/ImgiNav/ImgiNav"
MANIFEST="/work3/s233249/ImgiNav/datasets/layouts.csv"
OUTPUT_DIR="/work3/s233249/ImgiNav/ImgiNav/visualization/saved_embeddings"

# --- Metadata ---
LABEL_COL="room_id"
CATEGORY_COL="type"

# --- Sampling ---
NUM_POINTS="27897"

# --- Models to Compare (LISTS in specific order) ---
# 1. MSE 64x64x4 (Baseline)
# 2. SegLoss 64x64x4
# 3. SegLoss 32x32x4
# 4. SegLoss 32x32x4 HighSeg
# 5. SegLoss 32x32x2
CONFIGS=(
    "/work3/s233249/ImgiNav/experiments/VAE/VAE_512_64x64x4_MSE/output/experiment_config.yaml" # Model 0
    "/work3/s233249/ImgiNav/experiments/VAE/VAE_512_64x64x4_SegLoss/output/experiment_config.yaml"  # Model 1
    "/work3/s233249/ImgiNav/experiments/VAE/VAE_512_32x32x4_SegLoss/output/experiment_config.yaml"  # Model 2
    "/work3/s233249/ImgiNav/experiments/VAE/VAE_512_32x32x4_SegLoss_HighSeg/output/experiment_config.yaml" # Model 3
    "/work3/s233249/ImgiNav/experiments/VAE/VAE_512_32x32x2_SegLoss/output/experiment_config.yaml"  # Model 4
)

CHECKPOINTS=(
    "/work3/s233249/ImgiNav/experiments/VAE/VAE_512_64x64x4_MSE/checkpoints/ae_latest.pt"  # Model 0
    "/work3/s233249/ImgiNav/experiments/VAE/VAE_512_64x64x4_SegLoss/checkpoints/ae_latest.pt" # Model 1
    "/work3/s233249/ImgiNav/experiments/VAE/VAE_512_32x32x4_SegLoss/checkpoints/ae_latest.pt" # Model 2
    "/work3/s233249/ImgiNav/experiments/VAE/VAE_512_32x32x4_SegLoss_HighSeg/checkpoints/ae_latest.pt" # Model 3
    "/work3/s233249/ImgiNav/experiments/VAE/VAE_512_32x32x2_SegLoss/checkpoints/ae_latest.pt" # Model 4
)

echo "Generating embeddings for ${#CONFIGS[@]} models"
echo "Sampling ${NUM_POINTS} points for embedding"
echo "Outputting embeddings to: ${OUTPUT_DIR}"

# ----------------------------------------------------------------------
# Conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || true
fi

# ----------------------------------------------------------------------
# Create output directory
mkdir -p "${OUTPUT_DIR}"

# ----------------------------------------------------------------------
# Run embedding generation

# Build options array
OPTS=(
    --project_root "${PROJECT_ROOT}"
    --manifest "${MANIFEST}"
    --label_col "${LABEL_COL}"
    --category_col "${CATEGORY_COL}"
    --output_dir "${OUTPUT_DIR}"
    --num_points "${NUM_POINTS}"
    --umap_neighbors 15
    --umap_min_dist 0.1
    --configs "${CONFIGS[@]}"
    --checkpoints "${CHECKPOINTS[@]}"
)

# Run the script
python "${SCRIPT_PATH}" "${OPTS[@]}"

echo "Embedding generation completed successfully"