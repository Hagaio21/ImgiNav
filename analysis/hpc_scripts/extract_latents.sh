#!/bin/bash
#BSUB -J extract_latents
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/analysis/hpc_scripts/logs/extract_latents.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/analysis/hpc_scripts/logs/extract_latents.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=32000]"
#BSUB -W 08:00
#BSUB -q gpuv100


# ----------------------------------------------------------------------
# Configuration
export TORCHVISION_DISABLE_NMS_WARNING=1

SCRIPT_PATH="/work3/s233249/ImgiNav/ImgiNav/analysis/extract_latents.py"
MANIFEST="/work3/s233249/ImgiNav/datasets/layouts.csv"
EXP_ROOT="/work3/s233249/ImgiNav/experiments"
OUTPUT_DIR="/work3/s233249/ImgiNav/ImgiNav/analysis/latent_analysis"

BATCH_SIZE=32
NUM_WORKERS=8

echo "Starting latent extraction job"
echo "Manifest: ${MANIFEST}"
echo "Experiment root: ${EXP_ROOT}"
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
# Create output directory
mkdir -p "${OUTPUT_DIR}"

# ----------------------------------------------------------------------
# Run extraction
python "${SCRIPT_PATH}" \
  --manifest "${MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --exp_root "${EXP_ROOT}" \
  --batch_size ${BATCH_SIZE} \
  --num_workers ${NUM_WORKERS}

echo "Latent extraction completed successfully"
