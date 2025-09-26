#!/bin/bash
#BSUB -J stage3_scenes[1-10]
#BSUB -o /zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/logs/stage3_scenes.%I.%J.out
#BSUB -e /zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/logs/stage3_scenes.%I.%J.err
#BSUB -n 10
#BSUB -R "rusage[mem=4000]"
#BSUB -W 06:00
#BSUB -q hpc

set -euo pipefail

# Configuration
SCENES_ROOT="/work3/s233249/ImgiNav/datasets/scenes"
TAXONOMY_FILE="/zhome/62/5/203350/ws/ImgiNav/config/taxonomy.json"
PYTHON_SCRIPT="/zhome/62/5/203350/ws/ImgiNav/data_preperation/stage3_create_room_scenes_layouts.py"
MANIFEST_DIR="/zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/manifests/shards"

# Job array indexing
IDX=$((LSB_JOBINDEX - 1))
SCENE_MANIFEST="${MANIFEST_DIR}/scene_manifest_shard$(printf "%03d" ${IDX}).csv"

echo "Running room layout task ${LSB_JOBINDEX}/10 → shard ${IDX}"

# Verify manifest exists
if [ ! -s "${SCENE_MANIFEST}" ]; then
  echo "ERROR: Room manifest not found: ${SCENE_MANIFEST}"
  exit 1
fi

echo "Using manifest: ${SCENE_MANIFEST}"
echo "Sample entries:"
head -3 "${SCENE_MANIFEST}"

# Conda activation
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate scenefactor || true
fi

# Run Stage 3 for rooms with manifest
python "${PYTHON_SCRIPT}" \
  --in_root "${SCENES_ROOT}" \
  --taxonomy "${TAXONOMY_FILE}" \
  --manifest "${SCENE_MANIFEST}" \
  --mode "scene" \
  --res 512 \
  --hmin 0.1 \
  --hmax 1.8 \
  --point-size 5 \
  --color-mode "super"

echo "Scene layout task ${LSB_JOBINDEX} completed successfully"