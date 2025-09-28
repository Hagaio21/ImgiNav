#!/bin/bash
#BSUB -J stage4_povs_gpu[1-10]              # job array, adjust count
#BSUB -o /zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/logs/stage4_povs.%I.%J.out
#BSUB -e /zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/logs/stage4_povs.%I.%J.err
#BSUB -n 4                                  # CPU cores for GPU job
#BSUB -R "rusage[mem=4000]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -q gpuv100

set -euo pipefail

# --- paths ---
SCENES_ROOT="/work3/s233249/ImgiNav/datasets/scenes"
TAXONOMY_FILE="/zhome/62/5/203350/ws/ImgiNav/config/taxonomy.json"
PYTHON_SCRIPT="/zhome/62/5/203350/ws/ImgiNav/data_preperation/stage4_create_room_povs.py"
MANIFEST_DIR="/zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/manifests/shards"
OUT_DIR="/work3/s233249/ImgiNav/outputs/povs"

# --- shard selection ---
IDX=$((LSB_JOBINDEX - 1))
ROOM_MANIFEST="${MANIFEST_DIR}/room_manifest_shard$(printf "%03d" ${IDX}).csv"

if [ ! -s "${ROOM_MANIFEST}" ]; then
  echo "ERROR: missing shard manifest: ${ROOM_MANIFEST}" >&2
  exit 2
fi

echo "Running POV generation (GPU) for shard ${IDX} → ${ROOM_MANIFEST}"

# --- conda setup ---
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate scenefactor || true
elif [ -x "$HOME/miniconda3/bin/conda" ]; then
  eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
  conda activate scenefactor || true
fi

# --- runtime fixes ---
export XDG_RUNTIME_DIR=/tmp/$USER
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# --- run POV generation ---
python "${PYTHON_SCRIPT}" \
  --dataset-root "${SCENES_ROOT}" \
  --manifest "${ROOM_MANIFEST}" \
  --taxonomy "${TAXONOMY_FILE}" \
  --width 512 \
  --height 512 \
  --fov-deg 70 \
  --eye-height 1.6 \
  --point-size 3 \
  --num-views 6 \
  --seed 1 \
  --hpc
