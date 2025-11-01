#!/bin/bash
#BSUB -J phase1_3_l40s
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/phase1_3_l40s.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/phase1_3_l40s.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=32000]"
#BSUB -gpu "num=1"
#BSUB -W 06:00
#BSUB -q gpul40s

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train.py"
CONFIG_DIR="${BASE_DIR}/experiments/autoencoders/phase1"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Phase 1.3 experiment - F3 on L40s
CONFIG_FILE="${CONFIG_DIR}/phase1_3_AE_F3_rgb_seg_cls.yaml"

# Validate config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
  echo "ERROR: Config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

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
echo "Phase 1.3: Loss Tuning - F3 (L40s)"
echo "Config: ${CONFIG_FILE}"
echo "Working directory: ${BASE_DIR}"
echo "Start: $(date)"
echo "=========================================="

cd "${BASE_DIR}"

# Validate Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
  exit 1
fi

# Run training
python "${PYTHON_SCRIPT}" "${CONFIG_FILE}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "=========================================="
  echo "Training COMPLETE - SUCCESS"
  echo "Experiment: $(basename ${CONFIG_FILE})"
  echo "End: $(date)"
  echo "=========================================="
else
  echo "=========================================="
  echo "Training FAILED with exit code: ${EXIT_CODE}"
  echo "Experiment: $(basename ${CONFIG_FILE})"
  echo "End: $(date)"
  echo "=========================================="
  exit $EXIT_CODE
fi

