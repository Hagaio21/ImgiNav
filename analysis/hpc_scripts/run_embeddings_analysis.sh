#!/bin/bash
#BSUB -J embedding_analysis
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/analysis/logs/embedding_analysis.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/analysis/logs/embedding_analysis.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -W 04:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# Paths
SCRIPT_PATH="/work3/s233249/ImgiNav/ImgiNav/analysis/embeddings_analysis.py"
ROOM_MANIFEST="/work3/s233249/ImgiNav/datasets/room_dataset_with_emb.csv"
SCENE_MANIFEST="/work3/s233249/ImgiNav/datasets/scene_dataset_with_emb.csv"
OUT_DIR="/work3/s233249/ImgiNav/analysis/embedding_analysis_out"

# Conda environment setup
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || true
fi

echo "Running embedding diversity analysis..."
python "$SCRIPT_PATH" \
  --room_manifest "$ROOM_MANIFEST" \
  --scene_manifest "$SCENE_MANIFEST" \
  --out_dir "$OUT_DIR" \
  --subsample 5000

echo "Analysis complete. Results saved to $OUT_DIR"
