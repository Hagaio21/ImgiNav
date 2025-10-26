#!/bin/bash
#BSUB -J analyze_embeddings
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/visualization/hpc_scripts/logs/analyze_embeddings.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/visualization/hpc_scripts/logs/analyze_embeddings.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -W 2:00
#BSUB -q hpc

# ----------------------------------------------------------------------
# Configuration
# --- Paths ---
SCRIPT_PATH="/work3/s233249/ImgiNav/ImgiNav/visualization/analyze_embeddings.py"
EMBEDDINGS_DIR="/work3/s233249/ImgiNav/ImgiNav/visualization/saved_embeddings"
OUTPUT_DIR="/work3/s233249/ImgiNav/ImgiNav/visualization/latent_analysis_results"

# Optional: specify models to analyze (leave empty to analyze all)
# MODELS=("VAE_512_64x64x4_MSE" "VAE_512_64x64x4_SegLoss")
MODELS=()

echo "Analyzing embeddings from: ${EMBEDDINGS_DIR}"
echo "Outputting results to: ${OUTPUT_DIR}"

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
# Run analysis

# Build options array
OPTS=(
    --embeddings_dir "${EMBEDDINGS_DIR}"
    --output_dir "${OUTPUT_DIR}"
)

# Add models if specified
if [ ${#MODELS[@]} -gt 0 ]; then
    OPTS+=(--models "${MODELS[@]}")
fi

# Run the script
python "${SCRIPT_PATH}" "${OPTS[@]}"

echo "Analysis completed successfully"