#!/bin/bash
#BSUB -J analyze_dataset
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/analysis/hpc_scripts/logs/analyze_dataset.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/analysis/hpc_scripts/logs/analyze_dataset.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=32000]"
#BSUB -W 02:00
#BSUB -q hpc

SCRIPT_PATH="/work3/s233249/ImgiNav/ImgiNav/analysis/analyze_layout_dataset.py"
MANIFEST="/work3/s233249/ImgiNav/datasets/layouts.csv"
OUTPUT_DIR="/work3/s233249/ImgiNav/ImgiNav/analysis/dataset_analysis"

echo "Analyzing dataset: ${MANIFEST}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate imginav || true

export MPLBACKEND=Agg
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC

mkdir -p "${OUTPUT_DIR}"

python "${SCRIPT_PATH}" \
  --manifest "${MANIFEST}" \
  --output_dir "${OUTPUT_DIR}"

echo "Dataset analysis completed successfully"
