#!/bin/bash
#BSUB -J preembed_latents
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/preembed_latents.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/preembed_latents.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 12:00  # 12 hours should be enough for encoding full dataset
#BSUB -q gpul40s

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/data_preparation/create_embeddings.py"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Autoencoder config and checkpoint
AUTOENCODER_CONFIG="${BASE_DIR}/experiments/autoencoders/phase1/phase1_5_AE_final.yaml"
AUTOENCODER_CHECKPOINT="/work3/s233249/ImgiNav/experiments/phase1/phase1_5_AE_final/phase1_5_AE_final_checkpoint_best.pt"

# Dataset paths
INPUT_MANIFEST="/work3/s233249/ImgiNav/datasets/layouts.csv"
OUTPUT_MANIFEST="/work3/s233249/ImgiNav/datasets/layouts_latents.csv"

# Pre-embedding parameters
BATCH_SIZE=32
NUM_WORKERS=8

# =============================================================================
# VALIDATION
# =============================================================================
if [ ! -f "${AUTOENCODER_CONFIG}" ]; then
  echo "ERROR: Autoencoder config not found: ${AUTOENCODER_CONFIG}" >&2
  exit 1
fi

if [ ! -f "${AUTOENCODER_CHECKPOINT}" ]; then
  echo "WARNING: Best checkpoint not found, trying latest..." >&2
  AUTOENCODER_CHECKPOINT="${BASE_DIR}/experiments/phase1/phase1_5_AE_final/phase1_5_AE_final_checkpoint_latest.pt"
  if [ ! -f "${AUTOENCODER_CHECKPOINT}" ]; then
    echo "ERROR: Autoencoder checkpoint not found at best or latest path" >&2
    echo "  Expected: ${BASE_DIR}/experiments/phase1/phase1_5_AE_final/phase1_5_AE_final_checkpoint_best.pt" >&2
    echo "  Or: ${BASE_DIR}/experiments/phase1/phase1_5_AE_final/phase1_5_AE_final_checkpoint_latest.pt" >&2
    exit 1
  fi
fi

if [ ! -f "${INPUT_MANIFEST}" ]; then
  echo "ERROR: Input manifest not found: ${INPUT_MANIFEST}" >&2
  exit 1
fi

if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
  exit 1
fi

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
echo "Pre-embedding Layout Dataset"
echo "Autoencoder Config: ${AUTOENCODER_CONFIG}"
echo "Autoencoder Checkpoint: ${AUTOENCODER_CHECKPOINT}"
echo "Input Manifest: ${INPUT_MANIFEST}"
echo "Output Manifest: ${OUTPUT_MANIFEST}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Num Workers: ${NUM_WORKERS}"
echo "Working directory: ${BASE_DIR}"
echo "Start: $(date)"
echo "=========================================="

cd "${BASE_DIR}"

# Run pre-embedding
python "${PYTHON_SCRIPT}" \
  --type layout \
  --manifest "${INPUT_MANIFEST}" \
  --output-manifest "${OUTPUT_MANIFEST}" \
  --autoencoder-config "${AUTOENCODER_CONFIG}" \
  --autoencoder-checkpoint "${AUTOENCODER_CHECKPOINT}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device cuda

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "=========================================="
  echo "Pre-embedding COMPLETE - SUCCESS"
  echo "Output manifest: ${OUTPUT_MANIFEST}"
  echo "End: $(date)"
  echo "=========================================="
  echo ""
  echo "Next step: Run diffusion ablation experiments"
  echo "The manifest ${OUTPUT_MANIFEST} is ready to use."
else
  echo "=========================================="
  echo "Pre-embedding FAILED with exit code: ${EXIT_CODE}"
  echo "End: $(date)"
  echo "=========================================="
  exit $EXIT_CODE
fi

