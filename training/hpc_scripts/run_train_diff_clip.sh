#!/bin/bash
#BSUB -J train_diff_clip
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/train_diff_clip.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/train_diff_clip.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -gpu "num=1"
#BSUB -W 24:00
#BSUB -q gpuv100

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
# Get config path from first argument
CONFIG_PATH="${1:-}"

if [ -z "${CONFIG_PATH}" ]; then
  echo "ERROR: Config path required as first argument" >&2
  echo "Usage: $0 <config_path>" >&2
  exit 1
fi

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train_diffusion.py"
CONFIG="${BASE_DIR}/${CONFIG_PATH}"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Validate config file exists
if [ ! -f "${CONFIG}" ]; then
  echo "ERROR: Config file not found: ${CONFIG}" >&2
  exit 1
fi

# Extract experiment name from config
EXP_NAME=$(python3 -c "
import yaml
import sys
try:
    with open('${CONFIG}', 'r') as f:
        config = yaml.safe_load(f)
        exp_name = config.get('experiment', {}).get('name', 'unnamed')
        print(exp_name)
except Exception as e:
    print('unnamed', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null || echo "unnamed")

# =============================================================================
# MODULES
# =============================================================================
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
echo "Training CLIP Diffusion Model"
echo "=========================================="
echo "Experiment: ${EXP_NAME}"
echo "Config: ${CONFIG_PATH}"
echo "Working directory: ${BASE_DIR}"
echo "Start: $(date)"
echo "=========================================="

cd "${BASE_DIR}"

# Validate Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
  exit 1
fi

# Run training with resume support
python "${PYTHON_SCRIPT}" "${CONFIG}" --resume

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "Training COMPLETE - SUCCESS"
  echo "Experiment: ${EXP_NAME}"
  echo "Config: ${CONFIG_PATH}"
  echo "End: $(date)"
  echo "=========================================="
  exit 0
else
  echo ""
  echo "=========================================="
  echo "Training FAILED with exit code: ${EXIT_CODE}"
  echo "Experiment: ${EXP_NAME}"
  echo "Config: ${CONFIG_PATH}"
  echo "End: $(date)"
  echo "=========================================="
  exit $EXIT_CODE
fi

