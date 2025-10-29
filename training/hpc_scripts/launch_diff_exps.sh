#!/bin/bash
#BSUB -J diffusion_sweep[1-6]
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/diffusion_sweep.%J.%I.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/diffusion_sweep.%J.%I.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 10:00
#BSUB -q gpuv100

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train_diffusion.py"
CONFIG_DIR="${BASE_DIR}/config/architecture/diffusion"

# Ordered list of YAMLs for unconditioned diffusion sweep
CONFIG_FILES=(
  "${CONFIG_DIR}/E1_Linear_64.yaml"
  "${CONFIG_DIR}/E2_Cosine_64.yaml"
  "${CONFIG_DIR}/E3_Quadratic_64.yaml"
  "${CONFIG_DIR}/E4_Linear_128.yaml"
  "${CONFIG_DIR}/E5_Cosine_128.yaml"
  "${CONFIG_DIR}/E6_Quadratic_128.yaml"
)

# Pick config for this array index
CONFIG_FILE="${CONFIG_FILES[$((LSB_JOBINDEX-1))]}"

# =============================================================================
# MODULES
# =============================================================================
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# CONDA ENV
# =============================================================================
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || {
    echo "Failed to activate conda environment 'imginav'" >&2
    conda activate scenefactor || {
      echo "Failed to activate any conda environment" >&2
      exit 1
    }
  }
fi

# =============================================================================
# RUN
# =============================================================================
echo "=========================================="
echo "Array job index: ${LSB_JOBINDEX}"
echo "Running Diffusion experiment"
echo "Config: ${CONFIG_FILE}"
echo "Start: $(date)"
echo "=========================================="

python "${PYTHON_SCRIPT}" --config "${CONFIG_FILE}"

echo "=========================================="
echo "Experiment COMPLETE"
echo "End: $(date)"
echo "=========================================="
