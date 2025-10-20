#!/bin/bash
#BSUB -J ae_train
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/ae_train.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/ae_train.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=32000]"
#BSUB -gpu "num=1"
#BSUB -W 20:00
#BSUB -q gpul40s

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav/"
PYTHON_SCRIPT="${BASE_DIR}/training/train_autoencoder.py"
LAYOUT_MANIFEST="/work3/s233249/ImgiNav/datasets/layouts.csv"
OUTPUT_DIR="/work3/s233249/ImgiNav/experiments/autoencoder_final_32x32x16_vanila"

# =============================================================================
# CONFIG (fill this in)
# =============================================================================
CONFIG_FILE="/work3/s233249/ImgiNav/ImgiNav/config/architecture/autoencoders/config_diff_16ch_32x32_vanilla.yml"  # <--- replace with your config path
JOB_NAME="ae_final_32"

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
BATCH_SIZE=64
EPOCHS=200
LEARNING_RATE=0.001
RESIZE=512
LAYOUT_MODE="all"
LOSS="mse"

# =============================================================================
# MODULE LOADS (for DTU HPC)
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
echo "Training: ${JOB_NAME}"
echo "Config: ${CONFIG_FILE}"
echo "Start: $(date)"
echo "=========================================="

python "${PYTHON_SCRIPT}" \
  --layout_manifest "${LAYOUT_MANIFEST}" \
  --config "${CONFIG_FILE}" \
  --name "${JOB_NAME}" \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --lr ${LEARNING_RATE} \
  --resize ${RESIZE} \
  --loss "${LOSS}" \
  --device cuda \
  --outdir "${OUTPUT_DIR}" \
  --layout_mode "${LAYOUT_MODE}" \
  --save_images \
  --save_every 10 \
  --keep_only_best \
  --train_split 0.8

echo "=========================================="
echo "Training COMPLETE"
echo "End: $(date)"
echo "=========================================="

RESULTS_DIR="${OUTPUT_DIR}"/*"${JOB_NAME}"*
if [ -d $RESULTS_DIR ]; then
  echo "Results saved to: $RESULTS_DIR"
  if [ -f "$RESULTS_DIR/metrics.csv" ]; then
    echo "Final metrics:"
    tail -n 5 "$RESULTS_DIR/metrics.csv"
  fi
fi
