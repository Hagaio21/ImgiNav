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
TAXONOMY_FILE="${BASE_DIR}/config/taxonomy.json"
PYTHON_SCRIPT="${BASE_DIR}/data_preparation/create_category_zmaps.py"
OUTPUT_DIR="${BASE_DIR}/config"
SCENE_MANIFEST="/work3/s233249/ImgiNav/datasets/scene_list.csv"
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
echo "Output Dir: ${OUTPUT_DIR}"
echo "Scene List: ${SCENE_MANIFEST}"
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

# Verify manifests exist
if [ ! -s "${SCENE_MANIFEST}" ]; then
  echo "ERROR: Scene manifest not found: ${SCENE_MANIFEST}"
  exit 1
fi

if [ ! -s "${ROOM_MANIFEST}" ]; then
  echo "ERROR: Room manifest not found: ${ROOM_MANIFEST}"
  exit 1
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

# Run zmap creation with both manifests
python "${PYTHON_SCRIPT}" \
  --taxonomy "${TAXONOMY_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --scene-list "${SCENE_MANIFEST}" \
  --room-list "${ROOM_MANIFEST}"

echo "=========================================="
echo "Category Z-Map creation COMPLETE"
echo "Output files:"
echo "  - Scene zmap: ${OUTPUT_DIR}/zmap_scenes.json"
echo "  - Room zmap: ${OUTPUT_DIR}/zmap_rooms.json"
echo "End: $(date)"
echo "=========================================="

