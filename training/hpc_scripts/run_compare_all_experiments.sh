#!/bin/bash
#BSUB -J compare_experiments
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/compare_experiments.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/compare_experiments.%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=4000]"
#BSUB -W 2:00
#BSUB -q gpuv100

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/scripts/compare_all_experiments.py"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# =============================================================================
# MODULES
# =============================================================================
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
echo "Comparing All Experiments"
echo "=========================================="
echo "Start: $(date)"
echo "=========================================="

cd "${BASE_DIR}"

# Validate Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
  exit 1
fi

# Run comparison script
python "${PYTHON_SCRIPT}" \
    --base-dir /work3/s233249/ImgiNav/experiments/clip \
    --output-dir /work3/s233249/ImgiNav/experiments/clip/comparison_summary

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "Comparison COMPLETE - SUCCESS"
  echo "End: $(date)"
  echo "=========================================="
  exit 0
else
  echo ""
  echo "=========================================="
  echo "Comparison FAILED with exit code: ${EXIT_CODE}"
  echo "End: $(date)"
  echo "=========================================="
  exit $EXIT_CODE
fi

