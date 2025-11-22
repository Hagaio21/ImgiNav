#!/bin/bash
#BSUB -J diff_clip_downs_bottleneck_small
#BSUB -q gpu
#BSUB -n 1
#BSUB -R "rusage[mem=32GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 48:00
#BSUB -o logs/run_conditional_crossattention_diffusion_clip_downs_bottleneck_small_%J.out
#BSUB -e logs/run_conditional_crossattention_diffusion_clip_downs_bottleneck_small_%J.err

export MKL_INTERFACE_LAYER=LP64
set -euo pipefail

# ----------------------------------------------------------------------
# Create Log Directory
# ----------------------------------------------------------------------
LOG_DIR="/work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs"
mkdir -p "${LOG_DIR}"

echo "[INFO] LSF Job $LSB_JOBID started on $(hostname)."

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train_diffusion.py"
CONFIG="${BASE_DIR}/experiments/diffusion/new_layouts/conditional_crossattention_diffusion_clip_downs_bottleneck_small.yaml"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Validate config exists
if [ ! -f "${CONFIG}" ]; then
  echo "ERROR: Config file not found: ${CONFIG}" >&2
  exit 1
fi

# ----------------------------------------------------------------------
# Conda environment
# ----------------------------------------------------------------------
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || {
        echo "[ERROR] Failed to activate conda env 'imginav'" >&2
        exit 1
    }
fi

cd "${BASE_DIR}"

# ----------------------------------------------------------------------
# Run Training
# ----------------------------------------------------------------------
echo "=============================================================="
echo " Training Conditional Cross-Attention Diffusion (CLIP, Small)"
echo "=============================================================="
echo " Config: ${CONFIG}"
echo " Script: ${PYTHON_SCRIPT}"
echo "=============================================================="

python "${PYTHON_SCRIPT}" "${CONFIG}"

echo ""
echo "=============================================================="
echo "✓ Training completed"
echo "=============================================================="

