#!/bin/bash
#BSUB -J graph_embed
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs/graph_embed.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs/graph_embed.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=32000]"
#BSUB -gpu "num=1"
#BSUB -W 10:00
#BSUB -q gpul40s

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# ----------------------------------------------------------------------
# Configuration
SCRIPT_PATH="/work3/s233249/ImgiNav/ImgiNav/data_preperation/stage7_create_graph_embeddings.py"
TAXONOMY="/work3/s233249/ImgiNav/config/taxonomy.json"
MANIFEST="/work3/s233249/ImgiNav/datasets/graphs.csv"
OUT_MANIFEST="/work3/s233249/ImgiNav/datasets/graphs_with_embeddings.csv"
MODEL="all-MiniLM-L6-v2"
FORMAT="pt"

echo "Running Graph SentenceTransformer embedding job"

# ----------------------------------------------------------------------
# Conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || true
fi

# ----------------------------------------------------------------------
# Run embedding
python "${SCRIPT_PATH}" \
  --manifest "${MANIFEST}" \
  --taxonomy "${TAXONOMY}" \
  --output "${OUT_MANIFEST}" \
  --model "${MODEL}" \
  --format "${FORMAT}"

echo "Graph embedding completed successfully"