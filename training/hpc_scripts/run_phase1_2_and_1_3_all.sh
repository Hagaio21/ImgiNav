#!/bin/bash
#BSUB -J phase1_2_3_all[1-8]
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/phase1_2_3_all.%J.%I.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/phase1_2_3_all.%J.%I.err
#BSUB -n 8
#BSUB -R "rusage[mem=8000]"
#BSUB -gpu "num=1"
#BSUB -W 06:00
#BSUB -q gpuv100

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

# All 8 experiments: Phase 1.2 (2) + Phase 1.3 (6)
# Phase 1.2: V1 (deterministic), V2 (VAE)
# Phase 1.3: F1 (det), F1_vae, F2 (det), F2_vae, F3 (det), F3_vae
CONFIG_FILES=(
  # Phase 1.2: VAE Test
  "${CONFIG_DIR}/phase1_2_AE_V1_deterministic.yaml"
  "${CONFIG_DIR}/phase1_2_AE_V2_vae_light.yaml"
  # Phase 1.3: Loss Tuning (deterministic versions)
  "${CONFIG_DIR}/phase1_3_AE_F1_rgb_only.yaml"
  "${CONFIG_DIR}/phase1_3_AE_F2_rgb_seg.yaml"
  "${CONFIG_DIR}/phase1_3_AE_F3_rgb_seg_cls.yaml"
  # Phase 1.3: Loss Tuning (VAE versions)
  "${CONFIG_DIR}/phase1_3_AE_F1_rgb_only_vae.yaml"
  "${CONFIG_DIR}/phase1_3_AE_F2_rgb_seg_vae.yaml"
  "${CONFIG_DIR}/phase1_3_AE_F3_rgb_seg_cls_vae.yaml"
)

# Pick config for this array index
CONFIG_FILE="${CONFIG_FILES[$((LSB_JOBINDEX-1))]}"

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
echo "Phase 1.2 + 1.3: Combined Run (All 8 experiments)"
echo "Array job index: ${LSB_JOBINDEX}/8"
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

