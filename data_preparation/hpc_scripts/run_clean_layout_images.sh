#!/bin/bash
#BSUB -J clean_layout_images
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/clean_layout_images.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/clean_layout_images.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=2000]"
#BSUB -W 04:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# Configuration
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
LAYOUT_DIR="/work3/s233249/ImgiNav/datasets/layout_new"
TAXONOMY_FILE="${BASE_DIR}/config/taxonomy.json"
PYTHON_SCRIPT="${BASE_DIR}/data_preparation/clean_layout_images.py"
FAILED_DIR="${LAYOUT_DIR}/failed"
CLEANED_DIR="${LAYOUT_DIR}/cleaned"
LOG_DIR="${BASE_DIR}/data_preparation/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

echo "Cleaning layout images by detecting floor color and calculating density"
echo "Layout directory: ${LAYOUT_DIR}"
echo "Failed images will be moved to: ${FAILED_DIR}"
echo "Cleaned images will be saved to: ${CLEANED_DIR}"

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

# Verify taxonomy file exists
if [ ! -f "${TAXONOMY_FILE}" ]; then
  echo "ERROR: Taxonomy file not found: ${TAXONOMY_FILE}"
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

# Run cleaning script
python "${PYTHON_SCRIPT}" \
  --input-dir "${LAYOUT_DIR}" \
  --taxonomy "${TAXONOMY_FILE}" \
  --min-density 0.2 \
  --tolerance 0 \
  --failed-dir "${FAILED_DIR}" \
  --clean-dir "${CLEANED_DIR}" \
  --extensions png jpg jpeg

echo "Layout image cleaning completed successfully"
echo "Failed images moved to: ${FAILED_DIR}"
echo "Cleaned images saved to: ${CLEANED_DIR}"

