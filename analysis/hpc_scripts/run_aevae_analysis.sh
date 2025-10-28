#!/bin/bash
#BSUB -J analyze_latents
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/analysis/hpc_scripts/logs/analyze_latents.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/analysis/hpc_scripts/logs/analyze_latents.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -W 8:00
#BSUB -q gpuv100

# ----------------------------------------------------------------------
# Configuration
export TORCHVISION_DISABLE_NMS_WARNING=1

SCRIPT_PATH="/work3/s233249/ImgiNav/ImgiNav/analysis/analyze_latents.py"
MANIFEST="/work3/s233249/ImgiNav/datasets/layouts.csv"
OUTPUT_DIR="/work3/s233249/ImgiNav/latent_analysis"

BATCH_SIZE=32
NUM_SAMPLES=1024

echo "Starting latent analysis job"
echo "Manifest: ${MANIFEST}"
echo "Output directory: ${OUTPUT_DIR}"

# ----------------------------------------------------------------------
# Conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || true
fi

# ----------------------------------------------------------------------
# Deterministic + HPC settings
export MPLBACKEND=Agg
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC

# ----------------------------------------------------------------------
# Create output directory and logs
mkdir -p "${OUTPUT_DIR}"

# ----------------------------------------------------------------------
# Run analysis
python "${SCRIPT_PATH}" \
  --manifest "${MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size ${BATCH_SIZE} \
  --num_samples ${NUM_SAMPLES}

echo "Latent analysis completed successfully"
