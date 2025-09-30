#!/bin/bash
#BSUB -J pov_data_collect
#BSUB -o /zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/logs/pov_data_collect.%J.out
#BSUB -e /zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/logs/pov_data_collect.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=4000]"
#BSUB -W 04:00
#BSUB -q hpc

set -euo pipefail

# Paths
ROOT_DIR="/work3/s233249/ImgiNav/datasets/scenes"
SCRIPT_PATH="/zhome/62/5/203350/ws/ImgiNav/data_preperation/collect_dataset.py"
OUT_DIR="/work3/s233249/ImgiNav/datasets/manifests"
LAYOUTS_CSV="/zhome/62/5/203350/ws/ImgiNav/datasets/layouts.csv"

# Conda activation (if used)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate scenefactor || true
fi

echo "Running POV and dataset collection on $ROOT_DIR"
python "$SCRIPT_PATH" \
  --root "$ROOT_DIR" \
  --out "$OUT_DIR" \
  --layouts "$LAYOUTS_CSV"
echo "Finished. Output CSVs: $OUT_DIR/povs.csv , $OUT_DIR/data.csv"
