#!/bin/bash
#BSUB -J analyze_saved_latents
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/analysis/hpc_scripts/logs/analyze_saved_latents.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/analysis/hpc_scripts/logs/analyze_saved_latents.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=32000]"
#BSUB -W 04:00
#BSUB -q hpc

SCRIPT_PATH="/work3/s233249/ImgiNav/ImgiNav/analysis/analyze_latents.py"
LATENT_ROOT="/work3/s233249/ImgiNav/ImgiNav/analysis/latent_analysis"
OUTPUT_DIR="/work3/s233249/ImgiNav/ImgiNav/analysis/latent_analysis_results"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate imginav || true

export MPLBACKEND=Agg
export PYTHONHASHSEED=0

python "${SCRIPT_PATH}" \
  --latent_root "${LATENT_ROOT}" \
  --output_dir "${OUTPUT_DIR}"

echo "Latent analysis completed successfully"
