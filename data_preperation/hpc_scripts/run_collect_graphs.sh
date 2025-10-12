#!/bin/bash
#BSUB -J graph_manifest
#BSUB -o /zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/logs/graph_manifest.%J.out
#BSUB -e /zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/logs/graph_manifest.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=2000]"
#BSUB -W 02:00
#BSUB -q hpc

set -euo pipefail

# Paths
LAYOUT_MANIFEST="/work3/s233249/ImgiNav/datasets/layouts.csv"
SCRIPT_PATH="/zhome/62/5/203350/ws/ImgiNav/data_preperation/create_graph_manifest.py"
OUT_CSV="/work3/s233249/ImgiNav/datasets/graphs.csv"

# Conda activation (if you use it)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate scenefactor || true
fi

echo "Creating graph manifest from $LAYOUT_MANIFEST"
python "$SCRIPT_PATH" \
  --layout_manifest "$LAYOUT_MANIFEST" \
  --output "$OUT_CSV"

echo "Finished. Output manifest: $OUT_CSV"