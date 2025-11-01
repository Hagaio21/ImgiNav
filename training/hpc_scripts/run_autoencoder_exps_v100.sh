#!/bin/bash
#BSUB -J autoencoder_exps_v100[1-4]
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/autoencoder_exps_v100.%J.%I.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/autoencoder_exps_v100.%J.%I.err
#BSUB -n 8
#BSUB -R "rusage[mem=8000]"
#BSUB -gpu "num=1"
#BSUB -W 20:00
#BSUB -q gpuv100

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train.py"
CONFIG_DIR="${BASE_DIR}/experiments/autoencoders"

# First 4 experiments (exp1-exp4)
CONFIG_FILES=(
  "${CONFIG_DIR}/exp1_rgb_only.yaml"
  "${CONFIG_DIR}/exp2_rgb_seg.yaml"
  "${CONFIG_DIR}/exp3_rgb_seg_cls.yaml"
  "${CONFIG_DIR}/exp4_rgb_only_large.yaml"
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
echo "Training Autoencoder experiment (V100)"
echo "Config: ${CONFIG_FILE}"
echo "Start: $(date)"
echo "=========================================="

cd "${BASE_DIR}"
python "${PYTHON_SCRIPT}" "${CONFIG_FILE}"

echo "=========================================="
echo "Training COMPLETE"
echo "End: $(date)"
echo "=========================================="

