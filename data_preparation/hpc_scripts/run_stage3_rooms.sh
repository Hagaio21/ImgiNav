#!/bin/bash
#BSUB -J stage3_rooms[1-10]
#BSUB -o ${BASE_DIR}/data_preparation/hpc_scripts/logs/stage3_rooms.%I.%J.out
#BSUB -e ${BASE_DIR}/data_preparation/hpc_scripts/logs/stage3_rooms.%I.%J.err
#BSUB -n 10
#BSUB -R "rusage[mem=4000]"
#BSUB -W 10:00
#BSUB -q hpc

set -euo pipefail

# Source environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../env_config.sh" 2>/dev/null || {
    # Fallback if env_config.sh not found
    BASE_DIR="${IMGINAV_ROOT:-/work3/s233249/ImgiNav}"
    export BASE_DIR
}

# Configuration
SCENES_ROOT="${BASE_DIR}/datasets/scenes"
TAXONOMY_FILE="${BASE_DIR}/config/taxonomy.json"
PYTHON_SCRIPT="${BASE_DIR}/data_preparation/create_new_layouts.py"
MANIFEST_DIR="${BASE_DIR}/data_preparation/hpc_scripts/manifests/shards"

# Job array indexing
IDX=$((LSB_JOBINDEX - 1))
ROOM_MANIFEST="${MANIFEST_DIR}/room_manifest_shard$(printf "%03d" ${IDX}).csv"

echo "Running room layout task ${LSB_JOBINDEX}/10 → shard ${IDX}"

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

# Run Stage 3 for rooms with manifest
python "${PYTHON_SCRIPT}" \
  --in_root "${SCENES_ROOT}" \
  --taxonomy "${TAXONOMY_FILE}" \
  --output_dir "${SCENES_ROOT}" \
  --manifest "${ROOM_MANIFEST}" \
  --mode "room" \
  --res 512 \
  --hmin 0.1 \
  --hmax 1.8 \
  --point-size 10 \
  --color-mode "super"

echo "Room layout task ${LSB_JOBINDEX} completed successfully"