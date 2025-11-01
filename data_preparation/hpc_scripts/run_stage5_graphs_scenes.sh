#!/bin/bash
#BSUB -J stage5_scene_graphs
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs/stage5_scene_graphs.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs/stage5_scene_graphs.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=8000]"
#BSUB -W 06:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

DATA_ROOT="/work3/s233249/ImgiNav/datasets"
TAXONOMY_FILE="/work3/s233249/ImgiNav/ImgiNav/config/taxonomy.json"
STAGE5_SCENE_SCRIPT="/work3/s233249/ImgiNav/ImgiNav/data_preperation/stage5_2_build_scene_graphs.py"
SCENE_MANIFEST="/work3/s233249/ImgiNav/datasets/scene_list.csv"

echo "Running Stage 5.2 - Scene-level graphs"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || true
fi

python "${STAGE5_SCENE_SCRIPT}" \
  --in_dir "${DATA_ROOT}" \
  --taxonomy "${TAXONOMY_FILE}" \
  --manifest "${SCENE_MANIFEST}"

echo "Stage 5.2 completed successfully"