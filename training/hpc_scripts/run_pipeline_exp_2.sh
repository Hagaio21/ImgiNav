#!/bin/bash
#BSUB -J pipeline_train
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/pipeline_train.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/pipeline_train.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=32000]"
#BSUB -gpu "num=1"
#BSUB -W 24:00
#BSUB -q gpul40s

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train_pipeline.py"
CONFIG_FILE="${BASE_DIR}/config/architecture/pipeline/exp02.yml"

# =============================================================================
# MODULES
# =============================================================================
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64

# PyTorch optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

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
else
  echo "Conda not found at $HOME/miniconda3" >&2
  exit 1
fi

# Verify environment
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# =============================================================================
# CREATE LOG DIRECTORIES
# =============================================================================
LOG_DIR="/work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs"
mkdir -p "${LOG_DIR}"

# =============================================================================
# RUN
# =============================================================================
echo "=========================================="
echo "Training Full Pipeline Model"
echo "Config: ${CONFIG_FILE}"
echo "Start: $(date)"
echo "=========================================="

python "${PYTHON_SCRIPT}" --config "${CONFIG_FILE}"

EXIT_CODE=$?

echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
  echo "Pipeline Training COMPLETE ✓"
else
  echo "Pipeline Training FAILED (exit code: ${EXIT_CODE})"
fi
echo "End: $(date)"
echo "=========================================="

exit ${EXIT_CODE}