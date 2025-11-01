#!/bin/bash
#BSUB -J graph_text
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs/graph_text.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs/graph_text.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -W 10:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64
# ----------------------------------------------------------------------
# Configuration
SCRIPT_PATH="/work3/s233249/ImgiNav/ImgiNav/data_preperation/create_graph_text.py"
TAXONOMY="/work3/s233249/ImgiNav/ImgiNav/config/taxonomy.json"
MANIFEST="/work3/s233249/ImgiNav/datasets/graphs.csv"

echo "Running Graph to Text conversion job"

# ----------------------------------------------------------------------
# Conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || true
fi

# ----------------------------------------------------------------------
# Run conversion
python "${SCRIPT_PATH}" \
  --manifest "${MANIFEST}" \
  --taxonomy "${TAXONOMY}"

echo "Graph text conversion completed successfully"