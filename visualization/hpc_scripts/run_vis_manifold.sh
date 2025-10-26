#!/bin/bash
#BSUB -J visualize_latents
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/visualization/hpc_scripts/logs/vis_latents.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/visualization/hpc_scripts/logs/vis_latents.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -W 4:00
#BSUB -q gpuv100

# ----------------------------------------------------------------------
# Configuration
# --- Paths ---
SCRIPT_PATH="/work3/s233249/ImgiNav/ImgiNav/visualization/viz_latent_manifolds.py"
PROJECT_ROOT="/work3/s233249/ImgiNav/ImgiNav"
MANIFEST="/work3/s233249/ImgiNav/datasets/layouts.csv"
OUTPUT_DIR="/work3/s233249/ImgiNav/ImgiNav/visualization/latent_visuals_split" # New output dir

# --- Metadata ---
LABEL_COL="room_id"
CATEGORY_COL="type"

# --- NEW: Sampling ---
# Set to a number (e.g., 1000) to sample.
# Set to "" (empty string) to use all points.
NUM_POINTS="27897"

# --- Models to Compare (LISTS in specific order) ---
# 1. MSE 64x64x4 (Baseline)
# 2. SegLoss 64x64x4
# 3. SegLoss 32x32x4
# 4. SegLoss 32x32x4 HighSeg
# 5. SegLoss 32x32x2
CONFIGS=(
    "/work3/s233249/ImgiNav/experiments/VAE/VAE_512_64x64x4_MSE/experiment_config.yaml" # Model 0
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

echo "Running visualize_latents job for ${#CONFIGS[@]} models, splitting plots"
if [ -n "${NUM_POINTS}" ]; then
    echo "Sampling ${NUM_POINTS} points for embedding"
fi
echo "Outputting plots to: ${OUTPUT_DIR}"

# ----------------------------------------------------------------------
# Conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || true
fi

# ----------------------------------------------------------------------
# Run visualization

# Build options array
OPTS=(
    --project_root "${PROJECT_ROOT}"
    --manifest "${MANIFEST}"
    --label_col "${LABEL_COL}"
    --category_col "${CATEGORY_COL}"
    --output_dir "${OUTPUT_DIR}"
    --configs "${CONFIGS[@]}"
    --checkpoints "${CHECKPOINTS[@]}"
)

# Add sampling argument ONLY if NUM_POINTS is set and non-empty
if [ -n "${NUM_POINTS}" ]; then
    OPTS+=(--num_points "${NUM_POINTS}")
    echo "Using --num_points ${NUM_POINTS}"
fi

# Pass the options array to the python script
python "${SCRIPT_PATH}" "${OPTS[@]}"

echo "Visualization completed successfully"