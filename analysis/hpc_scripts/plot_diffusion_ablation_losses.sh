#!/bin/bash
#BSUB -J plot_diffusion_ablation
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/analysis/hpc_scripts/logs/plot_diffusion_ablation.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/analysis/hpc_scripts/logs/plot_diffusion_ablation.%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=4000]"
#BSUB -W 00:30
#BSUB -q hpc

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/analysis/plot_diffusion_ablation_losses.py"
LOG_DIR="${BASE_DIR}/analysis/hpc_scripts/logs"

# Ablation experiments directory
ABLATION_DIR="/work3/s233249/ImgiNav/experiments/diffusion/ablation"
OUTPUT_DIR="${ABLATION_DIR}/analysis"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# =============================================================================
# MODULES (if needed)
# =============================================================================
# No GPU needed for plotting, but we might need matplotlib
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
echo "Plot Diffusion Ablation Loss Curves"
echo "=========================================="
echo "Ablation directory: ${ABLATION_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Working directory: ${BASE_DIR}"
echo "Start: $(date)"
echo "=========================================="

cd "${BASE_DIR}"

# Validate Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
  exit 1
fi

# Validate ablation directory exists
if [ ! -d "${ABLATION_DIR}" ]; then
  echo "ERROR: Ablation directory not found: ${ABLATION_DIR}" >&2
  exit 1
fi

# Run plotting script
python "${PYTHON_SCRIPT}" \
  --ablation-dir "${ABLATION_DIR}" \
  --output-dir "${OUTPUT_DIR}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "=========================================="
  echo "Plotting COMPLETE - SUCCESS"
  echo "Results saved to: ${OUTPUT_DIR}"
  echo "End: $(date)"
  echo "=========================================="
else
  echo "=========================================="
  echo "Plotting FAILED with exit code: ${EXIT_CODE}"
  echo "End: $(date)"
  echo "=========================================="
  exit $EXIT_CODE
fi

