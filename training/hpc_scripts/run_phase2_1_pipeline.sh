#!/bin/bash
#BSUB -J phase2_1_pipeline
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/phase2_1_pipeline.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/phase2_1_pipeline.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=32000]"
#BSUB -gpu "num=1"
#BSUB -W 48:00  # Extended wall time for both autoencoder and diffusion training
#BSUB -q gpul40s

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train_pipeline_phase2.py"
AE_CONFIG="${BASE_DIR}/experiments/autoencoders/phase2/phase2_1_AE_64x64_structural.yaml"
DIFFUSION_CONFIG="${BASE_DIR}/experiments/diffusion/phase2/phase2_1_diffusion_64x64_bottleneck_attn.yaml"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Validate config files exist
if [ ! -f "${AE_CONFIG}" ]; then
  echo "ERROR: Autoencoder config file not found: ${AE_CONFIG}" >&2
  exit 1
fi

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
echo "Phase 2.1: Training Pipeline"
echo "Autoencoder + Diffusion Training"
echo "=========================================="
echo "Autoencoder config: ${AE_CONFIG}"
echo "Diffusion config: ${DIFFUSION_CONFIG}"
echo "Working directory: ${BASE_DIR}"
echo "Start: $(date)"
echo "=========================================="
echo ""
echo "Pipeline Overview:"
echo "  1. Train autoencoder with structural constraints"
echo "  2. Update diffusion config with autoencoder checkpoint"
echo "  3. Train diffusion model with bottleneck-only attention"
echo "=========================================="

cd "${BASE_DIR}"

# Validate Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
  exit 1
fi

# Run pipeline
python "${PYTHON_SCRIPT}" \
  --ae-config "${AE_CONFIG}" \
  --diffusion-config "${DIFFUSION_CONFIG}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "Pipeline COMPLETE - SUCCESS"
  echo "Autoencoder: $(basename ${AE_CONFIG})"
  echo "Diffusion: $(basename ${DIFFUSION_CONFIG})"
  echo "End: $(date)"
  echo "=========================================="
else
  echo ""
  echo "=========================================="
  echo "Pipeline FAILED with exit code: ${EXIT_CODE}"
  echo "Autoencoder: $(basename ${AE_CONFIG})"
  echo "Diffusion: $(basename ${DIFFUSION_CONFIG})"
  echo "End: $(date)"
  echo "=========================================="
  exit $EXIT_CODE
fi

