#!/bin/bash
#BSUB -J diff_clip_regular_rooms_small_bottleneck
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/train_diff_clip_regular_rooms_bottleneck.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/train_diff_clip_regular_rooms_bottleneck.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 48:00
#BSUB -q gpul40s


set -euo pipefail

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train_diffusion.py"
CONFIG="${BASE_DIR}/experiments/diffusion/clip/regular_rooms/small_bottleneck.yaml"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

mkdir -p "${LOG_DIR}"

module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || conda activate scenefactor
fi

cd "${BASE_DIR}"
python "${PYTHON_SCRIPT}" "${CONFIG}" --resume

