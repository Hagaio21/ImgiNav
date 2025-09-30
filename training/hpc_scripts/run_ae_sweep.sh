#!/bin/bash
#BSUB -J ae_sweep[1-27]
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/ae_sweep.%I.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/ae_sweep.%I.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 12:00
#BSUB -q gpul40s

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav/"
PYTHON_SCRIPT="${BASE_DIR}/training/train_autoencoder.py"
LAYOUT_MANIFEST="/work3/s233249/ImgiNav/datasets/layouts.csv"
OUTPUT_DIR="/work3/s233249/ImgiNav/experiments/ae_diffusion_sweep"
CONFIG_DIR="/work3/s233249/ImgiNav/experiments/ae_configs"
TEMPLATE_DIR="/work3/s233249/ImgiNav/ImgiNav/modules/templates"

# =============================================================================
# SWEEP PARAMETERS - Optimized for diffusion models
# =============================================================================
# Latent spatial sizes: larger is better for diffusion (more detail preserved)
LATENT_BASE_OPTIONS=(16 32 64)

# Latent channels: 4-8 channels is standard for diffusion (e.g., Stable Diffusion uses 4)
LATENT_CHANNELS_OPTIONS=(4 8 16)

# Architectures
ARCHITECTURES=("vanilla" "skip" "deep")

# Total jobs: 3 bases × 3 channels × 3 architectures = 27
IDX=$((LSB_JOBINDEX - 1))
BASE_IDX=$((IDX / 9))
CHANNEL_IDX=$(((IDX / 3) % 3))
ARCH_IDX=$((IDX % 3))

LATENT_BASE=${LATENT_BASE_OPTIONS[$BASE_IDX]}
LATENT_CHANNELS=${LATENT_CHANNELS_OPTIONS[$CHANNEL_IDX]}
ARCH=${ARCHITECTURES[$ARCH_IDX]}

LATENT_DIM=$((LATENT_CHANNELS * LATENT_BASE * LATENT_BASE))

CONFIG_ID="diff_${LATENT_CHANNELS}ch_${LATENT_BASE}x${LATENT_BASE}_${ARCH}"

echo "=========================================="
echo "Task ${LSB_JOBINDEX}/27: Config ${CONFIG_ID}"
echo "  Latent Channels: ${LATENT_CHANNELS}"
echo "  Latent Spatial: ${LATENT_BASE}x${LATENT_BASE}"
echo "  Total Latent Dim: ${LATENT_DIM}"
echo "  Architecture: ${ARCH}"
echo "=========================================="

# =============================================================================
# CHANNEL PROGRESSION - Scale with latent complexity
# =============================================================================
# Use medium-large capacity for diffusion latents
case $LATENT_BASE in
  16)
    # Smaller latent space -> can use larger conv channels
    CH1=64; CH2=128; CH3=256 ;;
  32)
    # Medium latent space -> balanced
    CH1=48; CH2=96; CH3=192 ;;
  64)
    # Larger latent space -> smaller conv channels (memory constraint)
    CH1=32; CH2=64; CH3=128 ;;
esac
CH4=$((CH3 + CH3/2))
CH5=$((CH3 * 2))

echo "  Channel progression: ${CH1} -> ${CH2} -> ${CH3} -> ${CH4} -> ${CH5}"

# =============================================================================
# TEMPLATE SELECTION + CONFIG GENERATION
# =============================================================================
mkdir -p "${CONFIG_DIR}"
CONFIG_FILE="${CONFIG_DIR}/config_${CONFIG_ID}.yml"

TEMPLATE_FILE="${TEMPLATE_DIR}/ae_template_${ARCH}.yml"

sed \
  -e "s|\${LATENT_DIM}|${LATENT_DIM}|g" \
  -e "s|\${LATENT_CHANNELS}|${LATENT_CHANNELS}|g" \
  -e "s|\${LATENT_BASE}|${LATENT_BASE}|g" \
  -e "s|\${CH1}|${CH1}|g" \
  -e "s|\${CH2}|${CH2}|g" \
  -e "s|\${CH3}|${CH3}|g" \
  -e "s|\${CH4}|${CH4}|g" \
  -e "s|\${CH5}|${CH5}|g" \
  "${TEMPLATE_FILE}" > "${CONFIG_FILE}"

echo "Config file created: ${CONFIG_FILE}"
echo ""

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
BATCH_SIZE=32
EPOCHS=50  # More epochs for better convergence
LEARNING_RATE=0.001  # Lower LR for stability
RESIZE=512
LAYOUT_MODE="scene"
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
JOB_NAME="ae_${CONFIG_ID}"

echo "=========================================="
echo "Training: ${JOB_NAME}"
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
  --save_every 5 \
  --keep_only_best || {
  echo "ERROR: Training failed" >&2
  exit 1
}

echo "=========================================="
echo "Task ${LSB_JOBINDEX}/27 COMPLETED"
echo "End: $(date)"
echo "=========================================="

# Report final metrics
RESULTS_DIR="${OUTPUT_DIR}"/*"${JOB_NAME}"*
if [ -d $RESULTS_DIR ]; then
  echo "Results saved to: $RESULTS_DIR"
  if [ -f "$RESULTS_DIR/metrics.csv" ]; then
    echo "Final metrics:"
    tail -n 5 "$RESULTS_DIR/metrics.csv"
  fi
fi

echo "Configuration ${CONFIG_ID} complete"