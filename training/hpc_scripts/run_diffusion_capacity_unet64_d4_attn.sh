#!/bin/bash
#BSUB -J diffusion_capacity_unet64_d4_attn
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/diffusion_capacity_unet64_d4_attn.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/diffusion_capacity_unet64_d4_attn.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=10000]"
#BSUB -gpu "num=1"
#BSUB -W 24:00
#BSUB -q gpua100

set -euo pipefail

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
CONFIG_FILE="${BASE_DIR}/experiments/diffusion/ablation/capacity_unet64_d4_attn.yaml"
PYTHON_SCRIPT="${BASE_DIR}/training/train_diffusion.py"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

mkdir -p "${LOG_DIR}"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}" >&2
    exit 1
fi

module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || conda activate scenefactor || exit 1
fi

cd "${BASE_DIR}"
python -m training.train_diffusion "${CONFIG_FILE}"

