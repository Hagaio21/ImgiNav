#!/bin/bash
#BSUB -J create_category_zmaps
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/create_category_zmaps.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/create_category_zmaps.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -W 10:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
SCENES_ROOT="/work3/s233249/ImgiNav/datasets/scenes"
TAXONOMY_FILE="${BASE_DIR}/config/taxonomy.json"
PYTHON_SCRIPT="${BASE_DIR}/data_preparation/create_category_zmaps.py"
OUTPUT_DIR="${BASE_DIR}/config"
ROOM_MANIFEST="/work3/s233249/ImgiNav/datasets/room_list.csv"
LOG_DIR="${BASE_DIR}/data_preparation/hpc_scripts/logs"

# =============================================================================
# SETUP
# =============================================================================
# Ensure log directory exists
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Running Category Z-Map Creation"
echo "Taxonomy: ${TAXONOMY_FILE}"
echo "Source: ${SCENES_ROOT}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Room List: ${ROOM_MANIFEST}"
echo "Start: $(date)"
echo "=========================================="

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

# Verify room manifest exists (optional but recommended)
if [ ! -s "${ROOM_MANIFEST}" ]; then
  echo "WARNING: Room manifest not found: ${ROOM_MANIFEST}"
  echo "Will discover room files from filesystem"
fi

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
# Change to base directory and set PYTHONPATH
cd "${BASE_DIR}"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"

# Run zmap creation with room_list if available
if [ -s "${ROOM_MANIFEST}" ]; then
  python "${PYTHON_SCRIPT}" \
    --parquet-root "${SCENES_ROOT}" \
    --taxonomy "${TAXONOMY_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --room-list "${ROOM_MANIFEST}" \
    --pattern "rooms/*/*.parquet"
else
  python "${PYTHON_SCRIPT}" \
    --parquet-root "${SCENES_ROOT}" \
    --taxonomy "${TAXONOMY_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --pattern "rooms/*/*.parquet"
fi

echo "=========================================="
echo "Category Z-Map creation COMPLETE"
echo "Output files:"
echo "  - Scene zmap: ${OUTPUT_DIR}/scene_zmap.json"
echo "  - Room zmap: ${OUTPUT_DIR}/room_zmap.json"
echo "End: $(date)"
echo "=========================================="

