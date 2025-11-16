#!/bin/bash
#BSUB -J filter_overly_white_gray
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/filter_overly_white_gray.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/filter_overly_white_gray.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=2000]"
#BSUB -W 02:00
#BSUB -q hpc

set -euo pipefail

# Configuration
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
LAYOUT_DIR="/work3/s233249/ImgiNav/datasets/layout_new"
TO_REVIEW_DIR="/work3/s233249/ImgiNav/datasets/layout_new/to_review"
PYTHON_SCRIPT="${BASE_DIR}/data_preparation/filter_overly_white_gray_layouts.py"
LOG_DIR="${BASE_DIR}/data_preparation/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

echo "Filtering overly white or gray layout images"
echo "Layout directory: ${LAYOUT_DIR}"
echo "Review directory: ${TO_REVIEW_DIR}"

# Verify layout directory exists
if [ ! -d "${LAYOUT_DIR}" ]; then
  echo "ERROR: Layout directory not found: ${LAYOUT_DIR}"
  exit 1
fi

# Verify Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}"
  exit 1
fi

# Conda activation
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

# Change to base directory and set PYTHONPATH
cd "${BASE_DIR}"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"

# Run filtering script
python "${PYTHON_SCRIPT}" \
  --layout_dir "${LAYOUT_DIR}" \
  --to_review_dir "${TO_REVIEW_DIR}" \
  --max_whiteness 0.90 \
  --max_grayness 0.30 \
  --max_combined 0.95

echo "Filtering completed successfully"
echo "Filtered images moved to: ${TO_REVIEW_DIR}"

