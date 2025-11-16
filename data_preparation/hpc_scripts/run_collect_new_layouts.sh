#!/bin/bash
#BSUB -J collect_new_layouts
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/collect_new_layouts.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/collect_new_layouts.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=2000]"
#BSUB -W 02:00
#BSUB -q hpc

set -euo pipefail

# Configuration
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
LAYOUT_DIR="/work3/s233249/ImgiNav/datasets/layout_new"
OUTPUT_CSV="/work3/s233249/ImgiNav/datasets/layouts_new.csv"
DATA_ROOT="/work3/s233249/ImgiNav/datasets"
PYTHON_SCRIPT="${BASE_DIR}/data_preparation/collect_new_layouts.py"
LOG_DIR="${BASE_DIR}/data_preparation/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

echo "Collecting all layout images from target folder into manifest"
echo "Layout directory: ${LAYOUT_DIR}"
echo "Output manifest: ${OUTPUT_CSV}"

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

# Count files in directory for info
FILE_COUNT=$(find "${LAYOUT_DIR}" -maxdepth 1 -type f | wc -l)
echo "Found ${FILE_COUNT} files in ${LAYOUT_DIR}"

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

# Run collection script - collects all files in the target folder
python "${PYTHON_SCRIPT}" \
  --layout_dir "${LAYOUT_DIR}" \
  --output "${OUTPUT_CSV}" \
  --data_root "${DATA_ROOT}" \
  --min-colors 4

echo "Layout collection completed successfully"
echo "Manifest saved to: ${OUTPUT_CSV}"

