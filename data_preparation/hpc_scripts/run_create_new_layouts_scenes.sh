#!/bin/bash
#BSUB -J create_new_layouts_scenes[1-10]
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/create_new_layouts_scenes.%J.%I.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/create_new_layouts_scenes.%J.%I.err
#BSUB -n 10
#BSUB -R "rusage[mem=4000]"
#BSUB -W 10:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# Configuration
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
SCENES_ROOT="/work3/s233249/ImgiNav/datasets/scenes"
TAXONOMY_FILE="${BASE_DIR}/config/taxonomy.json"
PYTHON_SCRIPT="${BASE_DIR}/data_preparation/create_new_layouts.py"
OUTPUT_DIR="/work3/s233249/ImgiNav/datasets"
SCENE_MANIFEST="/work3/s233249/ImgiNav/datasets/scene_list.csv"
LOG_DIR="${BASE_DIR}/data_preparation/hpc_scripts/logs"
MANIFEST_DIR="${BASE_DIR}/data_preparation/hpc_scripts/manifests/shards"

# Job array indexing
if [ -z "${LSB_JOBINDEX:-}" ]; then
  echo "ERROR: LSB_JOBINDEX is not set. This script must be run as a job array."
  exit 1
fi

NUM_JOBS=10
JOB_ID=${LSB_JOBINDEX}
IDX=$((JOB_ID - 1))

# Ensure directories exist
mkdir -p "${LOG_DIR}"
mkdir -p "${MANIFEST_DIR}"

echo "Running new scene layout creation task ${JOB_ID}/10"
echo "Job ID: ${LSB_JOBID:-unknown}"
echo "Job Index: ${JOB_ID}"

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

# Verify manifest exists
if [ ! -s "${SCENE_MANIFEST}" ]; then
  echo "ERROR: Scene manifest not found: ${SCENE_MANIFEST}"
  exit 1
fi

# Create chunk manifest for this job
CHUNK_FILE="${MANIFEST_DIR}/scene_manifest_shard$(printf "%03d" ${IDX}).csv"

# Count total lines (excluding header)
TOTAL_LINES=$(tail -n +2 "${SCENE_MANIFEST}" | wc -l)
LINES_PER_JOB=$(( (TOTAL_LINES + NUM_JOBS - 1) / NUM_JOBS ))

START_LINE=$(( IDX * LINES_PER_JOB + 2 ))  # +2 because we skip header and awk is 1-indexed
END_LINE=$(( (IDX + 1) * LINES_PER_JOB + 1 ))

echo "Processing scenes ${START_LINE} to ${END_LINE} (total: ${TOTAL_LINES})"

# Extract header
head -n 1 "${SCENE_MANIFEST}" > "${CHUNK_FILE}"

# Extract chunk
sed -n "${START_LINE},${END_LINE}p" "${SCENE_MANIFEST}" >> "${CHUNK_FILE}"

CHUNK_LINES=$(tail -n +2 "${CHUNK_FILE}" | wc -l)
echo "Chunk contains ${CHUNK_LINES} scenes"

if [ ${CHUNK_LINES} -eq 0 ]; then
  echo "No scenes in this chunk, skipping"
  exit 0
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

# Change to base directory and set PYTHONPATH so Python can find common module
cd "${BASE_DIR}"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"

# Run create_new_layouts with chunk manifest for scenes
python "${PYTHON_SCRIPT}" \
  --in_root "${SCENES_ROOT}" \
  --taxonomy "${TAXONOMY_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --manifest "${CHUNK_FILE}" \
  --mode "scene" \
  --res 512 \
  --hmin 0.1 \
  --hmax 1.8 \
  --point-size 5 \
  --scene-point-size 3 \
  --object-point-size-multiplier 1.5 \
  --scene-object-point-size-multiplier 1.0 \
  --min-colors 4 \
  --max-whiteness 0.95 \
  --color-mode "category"

echo "New scene layout creation task ${JOB_ID} completed successfully"
