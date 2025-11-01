#!/bin/bash
#BSUB -J phase1_2_analysis
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/analysis/hpc_scripts/logs/phase1_2_analysis.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/analysis/hpc_scripts/logs/phase1_2_analysis.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -gpu "num=1"
#BSUB -W 02:00
#BSUB -q gpuv100

# Ensure this runs on GPU queue
export CUDA_VISIBLE_DEVICES=0

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/analysis/phase1_2_analysis.py"
LOG_DIR="${BASE_DIR}/analysis/hpc_scripts/logs"

# Phase-specific paths
PHASE_DIR="${BASE_DIR}/outputs/phase1_2_spatial_resolution"
EXPERIMENTS_DIR="${BASE_DIR}/../experiments/phase1"
DATASET_MANIFEST="/work3/s233249/ImgiNav/datasets/layouts.csv"

# Ensure log directory exists
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
echo "Phase 1.2: Spatial Resolution Test Analysis"
echo "=========================================="
echo "Phase directory: ${PHASE_DIR}"
echo "Experiments directory: ${EXPERIMENTS_DIR}"
echo "Dataset manifest: ${DATASET_MANIFEST}"
echo "Working directory: ${BASE_DIR}"
echo "Start: $(date)"
echo "=========================================="

cd "${BASE_DIR}"

# Validate Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
  exit 1
fi

# Validate phase directory exists
if [ ! -d "${PHASE_DIR}" ]; then
  echo "ERROR: Phase directory not found: ${PHASE_DIR}" >&2
  echo "Please ensure Phase 1.2 experiments have completed." >&2
  exit 1
fi

# Run analysis with GPU
python "${PYTHON_SCRIPT}" \
  --phase-dir "${PHASE_DIR}" \
  --experiments-dir "${EXPERIMENTS_DIR}" \
  --dataset-manifest "${DATASET_MANIFEST}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "=========================================="
  echo "Analysis COMPLETE - SUCCESS"
  echo "Results saved to: ${PHASE_DIR}/analysis"
  echo "End: $(date)"
  echo "=========================================="
else
  echo "=========================================="
  echo "Analysis FAILED with exit code: ${EXIT_CODE}"
  echo "End: $(date)"
  echo "=========================================="
  exit $EXIT_CODE
fi

