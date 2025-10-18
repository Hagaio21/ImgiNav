#!/bin/bash
#BSUB -J align_pretrain
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/align_pretrain.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/align_pretrain.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=32000]"
#BSUB -gpu "num=1"
#BSUB -W 12:00
#BSUB -q gpul40s

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train_alignment.py"

# =============================================================================
# CONFIGURATION
# =============================================================================
ROOM_MANIFEST="/work3/s233249/ImgiNav/datasets/room_dataset_with_emb.csv"
SCENE_MANIFEST="/work3/s233249/ImgiNav/datasets/scene_dataset_with_emb.csv"
EXP_DIR="/work3/s233249/ImgiNav/experiments/alignment_pretrain_01"

BATCH_SIZE=128
EPOCHS=10
LR=1e-4
SUBSAMPLE=20000

# =============================================================================
# MODULE LOADS
# =============================================================================
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# CONDA ENVIRONMENT
# =============================================================================
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || {
    echo "Failed to activate conda environment 'imginav'" >&2
    exit 1
  }
fi

# =============================================================================
# EXECUTION
# =============================================================================
echo "=========================================="
echo "Alignment Pretraining Job (POV + Graph)"
echo "Room Manifest: ${ROOM_MANIFEST}"
echo "Scene Manifest: ${SCENE_MANIFEST}"
echo "Experiment Dir: ${EXP_DIR}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Epochs: ${EPOCHS}"
echo "Learning Rate: ${LR}"
echo "Subsample: ${SUBSAMPLE}"
echo "Start: $(date)"
echo "=========================================="

export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"
cd "${BASE_DIR}"

python "${PYTHON_SCRIPT}" \
  --room_manifest "${ROOM_MANIFEST}" \
  --scene_manifest "${SCENE_MANIFEST}" \
  --exp_dir "${EXP_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --subsample "${SUBSAMPLE}"

EXIT_CODE=$?

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
  echo "Alignment pretraining COMPLETE"
else
  echo "Alignment pretraining FAILED with exit code $EXIT_CODE"
fi
echo "End: $(date)"
echo "=========================================="

if [ -d "$EXP_DIR" ]; then
  echo ""
  echo "Experiment outputs saved to: ${EXP_DIR}"
  echo "  - Best model: ${EXP_DIR}/best.pt"
  echo "  - Latest model: ${EXP_DIR}/latest.pt"
  echo "  - Stats: ${EXP_DIR}/stats.json"
  echo "  - Curves: ${EXP_DIR}/alignment_curves.png"
fi

exit $EXIT_CODE
