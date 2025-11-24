#!/bin/bash
#BSUB -J debug_all_arch
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/debug_all_arch.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/debug_all_arch.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 12:00
#BSUB -q gpuv100

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/run_debug_all_architectures_job.py"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"
OUTPUT_DIR="${BASE_DIR}/debug_results"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

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
echo "Running debug on all architectures"
echo "=========================================="
echo "Base directory: ${BASE_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Python script: ${PYTHON_SCRIPT}"
echo "=========================================="

cd "${BASE_DIR}"

python3 "${PYTHON_SCRIPT}" \
    --output-base-dir "${OUTPUT_DIR}" \
    --skip-existing

echo "=========================================="
echo "Debug job completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="

