#!/bin/bash
#BSUB -J collect_all
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs/collect_all.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs/collect_all.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -W 2:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# ----------------------------------------------------------------------
# Configuration
SCRIPT_PATH="/work3/s233249/ImgiNav/ImgiNav/data_preperation/collect_all.py"
DATA_ROOT="/work3/s233249/ImgiNav/datasets/scenes"
POV_MANIFEST="/work3/s233249/ImgiNav/datasets/povs.csv"
OUTPUT_MANIFEST="/work3/s233249/ImgiNav/datasets/training_manifest.csv"

echo "Running collect_all job"
echo "Data root: ${DATA_ROOT}"
echo "POV manifest: ${POV_MANIFEST}"
echo "Output: ${OUTPUT_MANIFEST}"

# ----------------------------------------------------------------------
# Conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || true
fi

# ----------------------------------------------------------------------
# Run collection
python "${SCRIPT_PATH}" \
  --data-root "${DATA_ROOT}" \
  --pov-manifest "${POV_MANIFEST}" \
  --output "${OUTPUT_MANIFEST}"

echo "Collection completed successfully"