#!/bin/bash
#BSUB -J analyze_alignment
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/analysis/hpc_scripts/logs/analyze_alignment.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/analysis/hpc_scripts/logs/analyze_alignment.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=32000]"
#BSUB -gpu "num=1"
#BSUB -W 8:00
#BSUB -q gpul40s

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/analysis/analyze_alignment_effect.py"

ROOM_MANIFEST="/work3/s233249/ImgiNav/datasets/room_dataset_with_emb.csv"
SCENE_MANIFEST="/work3/s233249/ImgiNav/datasets/scene_dataset_with_emb.csv"
CHECKPOINT="/work3/s233249/ImgiNav/experiments/alignment_pretrain_01/best.pt"
OUT_DIR="/work3/s233249/ImgiNav/ImgiNav/analysis/alignment_effect_results"

SUBSAMPLE=5000

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
echo "Alignment Embedding Analysis Job"
echo "Room Manifest: ${ROOM_MANIFEST}"
echo "Scene Manifest: ${SCENE_MANIFEST}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output Dir: ${OUT_DIR}"
echo "Subsample: ${SUBSAMPLE}"
echo "Start: $(date)"
echo "=========================================="

export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"
cd "${BASE_DIR}"

python "${PYTHON_SCRIPT}" \
  --room_manifest "${ROOM_MANIFEST}" \
  --scene_manifest "${SCENE_MANIFEST}" \
  --checkpoint "${CHECKPOINT}" \
  --out_dir "${OUT_DIR}" \
  --subsample "${SUBSAMPLE}"

EXIT_CODE=$?

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
  echo "Alignment embedding analysis COMPLETE"
else
  echo "Alignment embedding analysis FAILED with exit code $EXIT_CODE"
fi
echo "End: $(date)"
echo "=========================================="

if [ -d "$OUT_DIR" ]; then
  echo ""
  echo "Analysis results saved to: ${OUT_DIR}"
  echo "  - Plots: ${OUT_DIR}/*.png"
  echo "  - Summary: ${OUT_DIR}/alignment_analysis_summary.json"
fi

exit $EXIT_CODE
