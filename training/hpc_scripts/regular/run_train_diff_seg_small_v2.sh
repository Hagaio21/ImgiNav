#!/bin/bash
#BSUB -J diff_seg_small
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/train_diff_seg_small_v2.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/train_diff_seg_small_v2.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 48:00
#BSUB -q gpul40s

set -euo pipefail

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train_diffusion.py"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Config path (can be overridden via bsub -env)
CONFIG="${CONFIG:-${BASE_DIR}/experiments/diffusion/v2/seg/rooms/both/small_down_bottleneck.yaml}"

mkdir -p "${LOG_DIR}"

# Validate config file exists
if [ ! -f "${CONFIG}" ]; then
  echo "ERROR: Config file not found: ${CONFIG}" >&2
  exit 1
fi

module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

cd "${BASE_DIR}"
python "${PYTHON_SCRIPT}" "${CONFIG}" --resume

