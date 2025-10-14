#!/bin/bash
#BSUB -J cond_diffusion
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/cond_diffusion.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/cond_diffusion.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=32000]"
#BSUB -gpu "num=1"
#BSUB -W 24:00
#BSUB -q gpul40s

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav/"
PYTHON_SCRIPT="${BASE_DIR}/training/train_conditioned_diffusion.py"

# =============================================================================
# CONFIG 
# =============================================================================
EXP_CONFIG=/work3/s233249/ImgiNav/ImgiNav/config/conditioned_exp_config.yml
ROOM_MANIFEST=/work3/s233249/ImgiNav/datasets/room_manifest_with_emb.csv
SCENE_MANIFEST=/work3/s233249/ImgiNav/datasets/scene_manifest_with_emb.csv

# =============================================================================
# MIXER TYPE (choices: concat, weighted, learned)
# =============================================================================
MIXER_TYPE="concat"

# =============================================================================
# POV MODE (choices: seg, tex, or leave empty for all)
# =============================================================================
POV_MODE="seg"  # Set to "seg", "tex", or "" for all POV types

# =============================================================================
# RESUME FLAG 
# =============================================================================
# RESUME_FLAG="--resume"
RESUME_FLAG=""

# =============================================================================
# MODULE LOADS
# =============================================================================
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# CONDA ACTIVATION
# =============================================================================
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || {
    echo "Failed to activate conda environment 'imginav'" >&2
    echo "Trying fallback environment 'scenefactor'..." >&2
    conda activate scenefactor || {
      echo "Failed to activate any conda environment" >&2
      exit 1
    }
  }
fi

# =============================================================================
# RUN TRAINING
# =============================================================================
echo "=========================================="
echo "Training Conditioned Diffusion Model (POV + Graph)"
echo "Experiment Config: ${EXP_CONFIG}"
echo "Room Manifest: ${ROOM_MANIFEST}"
echo "Scene Manifest: ${SCENE_MANIFEST}"
echo "Mixer Type: ${MIXER_TYPE}"
echo "POV Mode: ${POV_MODE:-all}"
echo "Resume: ${RESUME_FLAG:-false}"
echo "Start: $(date)"
echo "=========================================="

export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"
cd "${BASE_DIR}"

# Build command with optional POV mode
CMD_ARGS=(
  --exp_config "${EXP_CONFIG}"
  --room_manifest "${ROOM_MANIFEST}"
  --scene_manifest "${SCENE_MANIFEST}"
  --mixer_type "${MIXER_TYPE}"
)

if [ -n "$POV_MODE" ]; then
  CMD_ARGS+=(--pov_mode "${POV_MODE}")
fi

if [ -n "$RESUME_FLAG" ]; then
  CMD_ARGS+=("${RESUME_FLAG}")
fi

python "${PYTHON_SCRIPT}" "${CMD_ARGS[@]}"

EXIT_CODE=$?

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
  echo "Training COMPLETE"
else
  echo "Training FAILED with exit code $EXIT_CODE"
fi
echo "End: $(date)"
echo "=========================================="

# Extract experiment directory from config
EXP_DIR=$(grep -A 1 "^experiment:" "${EXP_CONFIG}" | grep "exp_dir:" | awk '{print $2}')

if [ -d "$EXP_DIR" ]; then
  echo ""
  echo "Experiment output saved to: ${EXP_DIR}"
  echo "  - Checkpoints: ${EXP_DIR}/checkpoints/"
  echo "  - Samples: ${EXP_DIR}/samples/"
  echo "  - Logs: ${EXP_DIR}/logs/"
  echo "  - Training stats: ${EXP_DIR}/logs/training_stats.json"
  echo "  - Training curves: ${EXP_DIR}/logs/training_curves.png"
fi

exit $EXIT_CODE