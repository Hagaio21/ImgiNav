#!/bin/bash
#BSUB -J diffusion_ablation
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/diffusion_ablation_l40s.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/diffusion_ablation_l40s.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=32000]"
#BSUB -gpu "num=1"
#BSUB -W 48:00  # 48 hours (long training for diffusion)
#BSUB -q gpul40s

set -euo pipefail

# Get config file from command line
if [ $# -eq 0 ]; then
    echo "ERROR: No config file provided" >&2
    echo "Usage: $0 <config.yaml>" >&2
    exit 1
fi

CONFIG_FILE="$1"

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train_diffusion.py"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Validate config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}" >&2
    exit 1
fi

# Extract experiment name from config file
CONFIG_BASENAME=$(basename "${CONFIG_FILE}" .yaml)
echo "Config: ${CONFIG_FILE}"
echo "Experiment: ${CONFIG_BASENAME}"

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
echo "Diffusion Ablation Experiment (L40s)"
echo "Experiment: ${CONFIG_BASENAME}"
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
    echo "Experiment: ${CONFIG_BASENAME}"
    echo "End: $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "Training FAILED with exit code: ${EXIT_CODE}"
    echo "Experiment: ${CONFIG_BASENAME}"
    echo "End: $(date)"
    echo "=========================================="
    exit $EXIT_CODE
fi

