#!/bin/bash
#BSUB -J graph_manifest
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs/graph_manifest.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs/graph_manifest.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=2000]"
#BSUB -W 04:00
#BSUB -q hpc

export MKL_INTERFACE_LAYER=LP64
set -euo pipefail

# Paths
ROOT_DIR="/work3/s233249/ImgiNav/datasets/scenes"
SCRIPT_PATH="/work3/s233249/ImgiNav/ImgiNav/data_preperation/collect_graph_manifest.py"
OUT_CSV="/work3/s233249/ImgiNav/datasets/graphs.csv"

# Conda activation
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || {
    echo "Failed to activate conda env 'imginav'" >&2
    exit 1
  }
fi
echo "[CONDA] Environment: ${CONDA_DEFAULT_ENV:-none}"
echo ""

echo "Collecting graph manifest from $ROOT_DIR"
python "$SCRIPT_PATH" \
  --root "$ROOT_DIR" \
  --output "$OUT_CSV"

echo "Finished. Graph manifest: $OUT_CSV"