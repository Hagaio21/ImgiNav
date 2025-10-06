#!/bin/bash
#BSUB -J unet_train
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/unet_train.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/unet_train.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=32000]"
#BSUB -gpu "num=1"
#BSUB -W 24:00
#BSUB -q gpul40s

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav/"
PYTHON_SCRIPT="${BASE_DIR}/training/train_base_diffusion.py"

# =============================================================================
# CONFIG (fill this in)
# =============================================================================
EXP_CONFIG=/work3/s233249/ImgiNav/ImgiNav/config/unet_exp_config_deep_500.yml
# =============================================================================
# RESUME FLAG (uncomment to resume training)
# =============================================================================
# RESUME_FLAG="--resume"
RESUME_FLAG=""

# =============================================================================
# MODULE LOADS (for DTU HPC)
# =============================================================================
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# CONDA ACTIVATION
# =============================================================================
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || {
    echo "Failed to activate conda environment 'imginav'" >&2
    echo "Trying fallback environment 'scenefactor'..." >&2
    conda activate scenefactor || {
      echo "Failed to activate any conda environment" >&2
      exit 1
    }
  }
fi

# =============================================================================
# RUN TRAINING
# =============================================================================
echo "=========================================="
echo "Training Diffusion Model (U-Net)"
echo "Experiment Config: ${EXP_CONFIG}"
echo "Resume: ${RESUME_FLAG:-false}"
echo "Start: $(date)"
echo "=========================================="

export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"
cd "${BASE_DIR}"

if [ -n "$RESUME_FLAG" ]; then
  python "${PYTHON_SCRIPT}" \
    --exp_config "${EXP_CONFIG}" \
    ${RESUME_FLAG}
else
  python "${PYTHON_SCRIPT}" \
    --exp_config "${EXP_CONFIG}"
fi

EXIT_CODE=$?

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
  echo "Training COMPLETE"
else
  echo "Training FAILED with exit code $EXIT_CODE"
fi
echo "End: $(date)"
echo "=========================================="

# Extract experiment directory from config
EXP_DIR=$(grep -A 1 "^experiment:" "${EXP_CONFIG}" | grep "exp_dir:" | awk '{print $2}')

if [ -d "$EXP_DIR" ]; then