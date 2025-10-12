#!/bin/bash
#BSUB -J stage5_room_graphs
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs/stage5_room_graphs.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs/stage5_room_graphs.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=8000]"
#BSUB -W 10:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

TAXONOMY_FILE="/work3/s233249/ImgiNav/ImgiNav/config/taxonomy.json"
STAGE5_ROOM_SCRIPT="/work3/s233249/ImgiNav/ImgiNav/data_preperation/stage5_1_build_room_graphs.py"
LAYOUT_MANIFEST="/work3/s233249/ImgiNav/datasets/layouts.csv"

echo "Running Stage 5.1 - Room-level graphs"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || true
fi

python "${STAGE5_ROOM_SCRIPT}" \
  --taxonomy "${TAXONOMY_FILE}" \
  --manifest "${LAYOUT_MANIFEST}"

echo "Stage 5.1 completed successfully"