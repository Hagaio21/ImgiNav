#!/bin/bash
#BSUB -J phase3_1_pipeline_unet48_256_conditional
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/phase3_1_pipeline_unet48_256_conditional.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/phase3_1_pipeline_unet48_256_conditional.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"  # Reduced from 32GB - 64x64x4 latents are much smaller than 512x512 images
#BSUB -gpu "num=1"
#BSUB -W 24:00  # Fine-tuning typically needs less time than full training
#BSUB -q gpuv100

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train_pipeline_phase3.py"
DIFFUSION_CONFIG="${BASE_DIR}/experiments/diffusion/phase3/phase3_1_diffusion_64x64_bottleneck_attn_unet48_256_conditional.yaml"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Validate config files exist
if [ ! -f "${DIFFUSION_CONFIG}" ]; then
  echo "ERROR: Diffusion config file not found: ${DIFFUSION_CONFIG}" >&2
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
echo "Phase 3.1: Fine-tuning Pipeline (UNet48, 256×256, Conditional)"
echo "Fine-tuning Unconditional → Conditional"
echo "=========================================="
echo "Diffusion config: ${DIFFUSION_CONFIG}"
echo "Working directory: ${BASE_DIR}"
echo "Start: $(date)"
echo "=========================================="
echo ""
echo "Pipeline Overview:"
echo "  1. Load unconditional model checkpoint"
echo "  2. Fine-tune with conditioning enabled (cfg_dropout_rate=0.1)"
echo "  3. Train for 200 epochs with lower learning rate (0.0001)"
echo "=========================================="

cd "${BASE_DIR}"

# Validate Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
  exit 1
fi

# Run phase 3 pipeline
# The script will auto-detect the unconditional checkpoint from the config
python "${PYTHON_SCRIPT}" \
  --diffusion-config "${DIFFUSION_CONFIG}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "Pipeline COMPLETE - SUCCESS"
  echo "Diffusion: $(basename ${DIFFUSION_CONFIG})"
  echo "End: $(date)"
  echo "=========================================="
else
  echo ""
  echo "=========================================="
  echo "Pipeline FAILED with exit code: ${EXIT_CODE}"
  echo "Diffusion: $(basename ${DIFFUSION_CONFIG})"
  echo "End: $(date)"
  echo "=========================================="
  exit $EXIT_CODE
fi

