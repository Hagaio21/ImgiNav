#!/bin/bash
#BSUB -J create_new_layouts[1-10]
#BSUB -o /zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/logs/create_new_layouts.%I.%J.out
#BSUB -e /zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/logs/create_new_layouts.%I.%J.err
#BSUB -n 10
#BSUB -R "rusage[mem=4000]"
#BSUB -W 10:00
#BSUB -q hpc

set -euo pipefail

# Configuration
SCENES_ROOT="/work3/s233249/ImgiNav/datasets/scenes"
TAXONOMY_FILE="/zhome/62/5/203350/ws/ImgiNav/config/taxonomy.json"
PYTHON_SCRIPT="/zhome/62/5/203350/ws/ImgiNav/data_preperation/create_new_layouts.py"
OUTPUT_DIR="/work3/s233249/ImgiNav/datasets"
MANIFEST_DIR="/zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/manifests/shards"
LOG_DIR="/zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Job array indexing
IDX=$((LSB_JOBINDEX - 1))
ROOM_MANIFEST="${MANIFEST_DIR}/room_manifest_shard$(printf "%03d" ${IDX}).csv"

echo "=========================================="
echo "Running new layout creation task ${LSB_JOBINDEX}/10 → shard ${IDX}"
echo "Job ID: ${LSB_JOBID}"
echo "Job Index: ${LSB_JOBINDEX}"
echo "=========================================="

# Verify paths exist
if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}"
  exit 1
fi

if [ ! -f "${TAXONOMY_FILE}" ]; then
  echo "ERROR: Taxonomy file not found: ${TAXONOMY_FILE}"
  exit 1
fi

# Verify manifest exists
if [ ! -s "${ROOM_MANIFEST}" ]; then
  echo "ERROR: Room manifest not found: ${ROOM_MANIFEST}"
  echo "Available manifests in ${MANIFEST_DIR}:"
  ls -la "${MANIFEST_DIR}"/*.csv 2>/dev/null | head -5 || echo "No CSV files found"
  exit 1
fi

echo "Using manifest: ${ROOM_MANIFEST}"
echo "Manifest size: $(wc -l < "${ROOM_MANIFEST}") lines"
echo "Sample entries:"
head -3 "${ROOM_MANIFEST}"

# Conda activation
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate scenefactor || true
fi

# Run create_new_layouts with manifest
echo "Starting Python script..."
echo "Command: python ${PYTHON_SCRIPT} --in_root ${SCENES_ROOT} --taxonomy ${TAXONOMY_FILE} --output_dir ${OUTPUT_DIR} --manifest ${ROOM_MANIFEST} --mode room --res 512 --hmin -1.0 --hmax 1.8 --point-size 5 --color-mode category"

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

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
  echo "✓ New layout creation task ${LSB_JOBINDEX} completed successfully"
else
  echo "✗ New layout creation task ${LSB_JOBINDEX} failed with exit code ${EXIT_CODE}"
  exit ${EXIT_CODE}
fi

