#!/bin/bash
#BSUB -J pov_embed
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs/pov_embed.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs/pov_embed.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=32000]"
#BSUB -gpu "num=1"
#BSUB -W 10:00
#BSUB -q gpul40s

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# ----------------------------------------------------------------------
# Configuration
SCRIPT_PATH="/work3/s233249/ImgiNav/ImgiNav/data_preperation/stage7_create_pov_embeddings.py"
MANIFEST="/work3/s233249/ImgiNav/datasets/povs.csv"
OUT_MANIFEST="/work3/s233249/ImgiNav/datasets/povs_with_embeddings.csv"
FORMAT="pt"

echo "Running POV ResNet embedding job"

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
  --output "${OUT_MANIFEST}" \
  --format "${FORMAT}"

echo "POV embedding completed successfully"