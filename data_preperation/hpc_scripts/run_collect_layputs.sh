#!/bin/bash
#BSUB -J layout_manifest
#BSUB -o /zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/logs/layout_manifest.%J.out
#BSUB -e /zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/logs/layout_manifest.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=2000]"
#BSUB -W 02:00
#BSUB -q hpc

set -euo pipefail

# Paths
ROOT_DIR="/work3/s233249/ImgiNav/datasets/scenes"
SCRIPT_PATH="/zhome/62/5/203350/ws/ImgiNav/data_preperation/layout_collection.py"
OUT_CSV="/work3/s233249/ImgiNav/datasets/layouts.csv"

# Conda activation (if you use it)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate scenefactor || true
fi

echo "Running layout collection on $ROOT_DIR"
python "$SCRIPT_PATH" --root "$ROOT_DIR" --out "$OUT_CSV"
echo "Finished. Output manifest: $OUT_CSV"
