#!/bin/bash
#BSUB -J analyze_latents
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/analysis/hpc_scripts/logs/analyze_latents.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/analysis/hpc_scripts/logs/analyze_latents.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=32000]"
#BSUB -W 4:00
#BSUB -q hpc

# ----------------------------------------------------------------------
# Configuration
SCRIPT_PATH="/work3/s233249/ImgiNav/ImgiNav/analysis/analyze_latents.py"
LATENT_ROOT="/work3/s233249/ImgiNav/ImgiNav/analysis/latent_analysis"
EXP_ROOT="/work3/s233249/ImgiNav/experiments/AEVAE_sweep"
OUTPUT_DIR="/work3/s233249/ImgiNav/ImgiNav/analysis/latent_analysis_results"

echo "Analyzing saved latents:"
echo "  Latent root: ${LATENT_ROOT}"
echo "  Experiment root: ${EXP_ROOT}"
echo "  Output directory: ${OUTPUT_DIR}"

# ----------------------------------------------------------------------
# Environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || true
fi

export MPLBACKEND=Agg
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC

# ----------------------------------------------------------------------
# Run analysis
python "${SCRIPT_PATH}" \
  --latent_root "${LATENT_ROOT}" \
  --exp_root "${EXP_ROOT}" \
  --output_root "${OUTPUT_DIR}"

echo "Latent analysis completed successfully"
