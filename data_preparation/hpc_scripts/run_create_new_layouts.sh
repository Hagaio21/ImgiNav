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

# Job array indexing
IDX=$((LSB_JOBINDEX - 1))
ROOM_MANIFEST="${MANIFEST_DIR}/room_manifest_shard$(printf "%03d" ${IDX}).csv"

echo "Running new layout creation task ${LSB_JOBINDEX}/10 → shard ${IDX}"

# Verify manifest exists
if [ ! -s "${ROOM_MANIFEST}" ]; then
  echo "ERROR: Room manifest not found: ${ROOM_MANIFEST}"
  exit 1
fi

echo "Using manifest: ${ROOM_MANIFEST}"
echo "Sample entries:"
head -3 "${ROOM_MANIFEST}"

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

echo "New layout creation task ${LSB_JOBINDEX} completed successfully"

