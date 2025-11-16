#!/bin/bash
#BSUB -J create_new_layouts
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/create_new_layouts.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/create_new_layouts.%J.err
#BSUB -n 10
#BSUB -R "rusage[mem=4000]"
#BSUB -W 10:00
#BSUB -q hpc

set -euo pipefail

# Configuration
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
SCENES_ROOT="/work3/s233249/ImgiNav/datasets/scenes"
TAXONOMY_FILE="${BASE_DIR}/config/taxonomy.json"
PYTHON_SCRIPT="${BASE_DIR}/data_preparation/create_new_layouts.py"
OUTPUT_DIR="/work3/s233249/ImgiNav/datasets"
ROOM_MANIFEST="/work3/s233249/ImgiNav/datasets/room_list.csv"
LOG_DIR="${BASE_DIR}/data_preparation/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

echo "Running new layout creation for all rooms"
echo "Job ID: ${LSB_JOBID:-unknown}"

# Verify Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}"
  echo "Please verify the script exists at this path"
  exit 1
fi

# Verify taxonomy file exists
if [ ! -f "${TAXONOMY_FILE}" ]; then
  echo "ERROR: Taxonomy file not found: ${TAXONOMY_FILE}"
  exit 1
fi

# Verify manifest exists
if [ ! -s "${ROOM_MANIFEST}" ]; then
  echo "ERROR: Room manifest not found: ${ROOM_MANIFEST}"
  exit 1
fi

echo "Using manifest: ${ROOM_MANIFEST}"
echo "Manifest exists and has $(wc -l < "${ROOM_MANIFEST}") lines"
echo "Sample entries:"
head -3 "${ROOM_MANIFEST}" || echo "Could not read manifest"

# Conda activation
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate scenefactor || true
fi

# Run create_new_layouts with manifest
python "${PYTHON_SCRIPT}" \
  --in_root "${SCENES_ROOT}" \
  --taxonomy "${TAXONOMY_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --manifest "${ROOM_MANIFEST}" \
  --mode "room" \
  --res 512 \
  --hmin -1.0 \
  --hmax 1.8 \
  --point-size 5 \
  --color-mode "category"

echo "New layout creation completed successfully"

