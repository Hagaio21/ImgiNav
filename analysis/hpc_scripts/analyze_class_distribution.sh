#!/bin/bash
#BSUB -J analyze_class_dist
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/analysis/hpc_scripts/logs/analyze_class_dist.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/analysis/hpc_scripts/logs/analyze_class_dist.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -W 02:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/analysis/analyze_class_distribution.py"
MANIFEST="/work3/s233249/ImgiNav/datasets/augmented/manifest.csv"
OUTPUT_DIR="${BASE_DIR}/analysis/class_distribution_results"
LOG_DIR="${BASE_DIR}/analysis/hpc_scripts/logs"
RARE_THRESHOLD_PERCENTILE=10.0

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

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
# ENVIRONMENT VARIABLES
# =============================================================================
export MPLBACKEND=Agg
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# VALIDATION
# =============================================================================
echo "=========================================="
echo "Analyze Class Distribution"
echo "=========================================="
echo "Manifest: ${MANIFEST}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Rare threshold percentile: ${RARE_THRESHOLD_PERCENTILE}"
echo "Working directory: ${BASE_DIR}"
echo "Start: $(date)"
echo "=========================================="

cd "${BASE_DIR}"

# Validate Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
  exit 1
fi

# Validate manifest exists
if [ ! -f "${MANIFEST}" ]; then
  echo "ERROR: Manifest file not found: ${MANIFEST}" >&2
  exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# =============================================================================
# RUN ANALYSIS
# =============================================================================
python "${PYTHON_SCRIPT}" \
  --manifest "${MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --rare_threshold_percentile "${RARE_THRESHOLD_PERCENTILE}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "=========================================="
  echo "Class distribution analysis COMPLETE - SUCCESS"
  echo "Results saved to: ${OUTPUT_DIR}"
  echo "End: $(date)"
  echo "=========================================="
else
  echo "=========================================="
  echo "Class distribution analysis FAILED with exit code: ${EXIT_CODE}"
  echo "End: $(date)"
  echo "=========================================="
  exit $EXIT_CODE
fi

