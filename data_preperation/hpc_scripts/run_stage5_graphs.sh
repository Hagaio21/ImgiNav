#!/bin/bash
#BSUB -J stage5_graphs
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs/stage5_graphs.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs/stage5_graphs.%J.err
#BSUB -n 10
#BSUB -R "rusage[mem=8000]"
#BSUB -W 08:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# ----------------------------------------------------------------------
# Configuration
DATA_ROOT="/work3/s233249/ImgiNav/datasets"
TAXONOMY_FILE="/work3/s233249/ImgiNav/ImgiNav/config/taxonomy.json"
STAGE5_ROOM_SCRIPT="/work3/s233249/ImgiNav/ImgiNav/data_preperation/stage5_1_build_room_graphs.py"
STAGE5_SCENE_SCRIPT="/work3/s233249/ImgiNav/ImgiNav/data_preperation/stage5_2_build_scene_graphs.py"
LAYOUT_MANIFEST="/work3/s233249/ImgiNav/datasets/layouts.csv"
SCENE_MANIFEST="/work3/s233249/ImgiNav/datasets/scene_list.csv"

echo "Running Stage 5 pipeline"

# ----------------------------------------------------------------------
# Conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || true
fi

# ----------------------------------------------------------------------
# Stage 5.1 – room-level graphs (uses layouts.csv)
python "${STAGE5_ROOM_SCRIPT}" \
  --taxonomy "${TAXONOMY_FILE}" \
  --manifest "${LAYOUT_MANIFEST}"

# ----------------------------------------------------------------------
# Stage 5.2 – scene-level graphs (uses scene_list.csv)
python "${STAGE5_SCENE_SCRIPT}" \
  --in_dir "${DATA_ROOT}" \
  --taxonomy "${TAXONOMY_FILE}" \
  --manifest "${SCENE_MANIFEST}"

echo "Stage 5 completed successfully"
