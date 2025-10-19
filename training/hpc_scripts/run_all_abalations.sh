#!/bin/bash
#BSUB -J "ablation[1-6]"
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/ablation_job_%J_task_%I.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/ablation_job_%J_task_%I.err
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -gpu "num=1"
#BSUB -W 8:00
#BSUB -q gpul40s

# --- Strict Mode ---
set -euo pipefail

# =============================================================================
# Configuration Variables
# =============================================================================
## --- Project Paths ---
readonly BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
readonly PYTHON_SCRIPT="${BASE_DIR}/training/train_cond_diffusion.py"

## --- Job Array Setup ---
# This bash array holds all 7 of your config files.
# Make sure the paths are correct.
readonly CONFIG_DIR="${BASE_DIR}/config/experiments/ablations"
readonly CONFIGS=(
    "${CONFIG_DIR}/scenes_graph_only.yml"
    "${CONFIG_DIR}/rooms_pov_graph.yml"
    "${CONFIG_DIR}/rooms_pov_only.yml"
    "${CONFIG_DIR}/rooms_graph_only.yml"
    "${CONFIG_DIR}/all_pov_graph.yml"
    "${CONFIG_DIR}/all_graph_only.yml"
)

# --- Select the config file for THIS job task ---
# LSF provides the $LSB_JOBINDEX variable, which will be 1, 2, 3... up to 7.
# Bash arrays are 0-indexed, so we subtract 1.
INDEX=$((LSB_JOBINDEX - 1))
readonly EXP_CONFIG="${CONFIGS[${INDEX}]}"
readonly JOB_NAME=$(basename "${EXP_CONFIG}" .yml)

## --- Model & Training Options ---
readonly RESUME_JOB="false"

# =============================================================================
# Environment Setup
# =============================================================================
echo "--- Loading Modules ---"
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64
echo "Modules loaded."

echo "--- Activating Conda Environment ---"
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    if conda activate imginav; then
        echo "Activated conda environment 'imginav'."
    elif conda activate scenefactor; then
        echo "Activated fallback conda environment 'scenefactor'."
    else
        echo "ERROR: Failed to activate 'imginav' or 'scenefactor' conda environment." >&2
        exit 1
    fi
else
    echo "WARNING: Conda initialization script not found." >&2
fi

# =============================================================================
# Run Training Script
# =============================================================================
echo "=========================================="
echo "Starting LSF Job Array Task ${LSB_JOBINDEX}"
echo "  Job Name:      ${JOB_NAME}"
echo "  Config:        ${EXP_CONFIG}"
echo "  Resume:        ${RESUME_JOB}"
echo "  Start Time:    $(date)"
echo "=========================================="

export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"
cd "${BASE_DIR}"

CMD_ARGS=(
    --exp_config "${EXP_CONFIG}"
)

if [[ "$RESUME_JOB" == "true" ]]; then
    CMD_ARGS+=(--resume)
fi

echo "--- Running Python Script ---"
python "${PYTHON_SCRIPT}" "${CMD_ARGS[@]}"
EXIT_CODE=$?

# =============================================================================
# Final Output & Cleanup
# =============================================================================
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Task ${LSB_JOBINDEX} (${JOB_NAME}) COMPLETE"
else
    echo "❌ Task ${LSB_JOBINDEX} (${JOB_NAME}) FAILED with exit code $EXIT_CODE"
fi
echo "  End Time:      $(date)"
echo "=========================================="

exit $EXIT_CODE