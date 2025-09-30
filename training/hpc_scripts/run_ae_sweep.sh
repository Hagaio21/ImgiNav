#!/bin/bash
#BSUB -J ae_sweep[1-36]
#BSUB -o /zhome/62/5/203350/ws/ImgiNav/hpc_scripts/logs/ae_sweep.%I.%J.out
#BSUB -e /zhome/62/5/203350/ws/ImgiNav/hpc_scripts/logs/ae_sweep.%I.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -R "select[gpu40gb]"
#BSUB -gpu "num=1"
#BSUB -W 12:00
#BSUB -q gpua40

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/zhome/62/5/203350/ws/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train_autoencoder.py"
LAYOUT_MANIFEST="${BASE_DIR}/indexes/layouts.csv"
OUTPUT_DIR="${BASE_DIR}/experiments/ae_architecture_sweep"
CONFIG_DIR="${BASE_DIR}/experiments/ae_configs"

# =============================================================================
# SWEEP PARAMETERS (36 combinations total)
# =============================================================================
# 4 latent dimensions for finding optimal size
LATENT_DIMS=(16 32 64 128)

# 3 architecture families
ARCHITECTURES=("vanilla" "skip" "deep")

# 3 channel progressions for each architecture
CHANNEL_CONFIGS=("small" "medium" "large")

# Calculate configuration from job array index (1-36)
IDX=$((LSB_JOBINDEX - 1))
LATENT_IDX=$((IDX / 9))
ARCH_IDX=$(((IDX / 3) % 3))
CHANNEL_IDX=$((IDX % 3))

LATENT_DIM=${LATENT_DIMS[$LATENT_IDX]}
ARCH=${ARCHITECTURES[$ARCH_IDX]}
CHANNELS=${CHANNEL_CONFIGS[$CHANNEL_IDX]}

CONFIG_ID="ld${LATENT_DIM}_${ARCH}_${CHANNELS}"

echo "=========================================="
echo "Task ${LSB_JOBINDEX}/36: Config ${CONFIG_ID}"
echo "  Latent Dim: ${LATENT_DIM}"
echo "  Architecture: ${ARCH}"
echo "  Channel Config: ${CHANNELS}"
echo "=========================================="

# =============================================================================
# GENERATE CONFIG FILE BASED ON ARCHITECTURE
# =============================================================================
mkdir -p "${CONFIG_DIR}"
CONFIG_FILE="${CONFIG_DIR}/config_${CONFIG_ID}.yml"

# Define channel progressions based on size
case $CHANNELS in
  "small")
    CH1=16
    CH2=32
    CH3=64
    ;;
  "medium")
    CH1=32
    CH2=64
    CH3=128
    ;;
  "large")
    CH1=64
    CH2=128
    CH3=256
    ;;
esac

# Generate config based on architecture family
cat > "${CONFIG_FILE}" << EOF
encoder:
  in_channels: 3
  latent_dim: ${LATENT_DIM}
  image_size: 512
  global_norm: null
  global_act: null
  global_dropout: 0.0
  layers:
EOF

# Add layers based on architecture type
case $ARCH in
  "vanilla")
    # Simple 3-layer encoder
    cat >> "${CONFIG_FILE}" << EOF
    - out_channels: ${CH1}
      kernel_size: 3
      stride: 2
      padding: 1
      norm: batch
      act: relu
      dropout: 0.0
    - out_channels: ${CH2}
      kernel_size: 3
      stride: 2
      padding: 1
      norm: batch
      act: relu
      dropout: 0.0
    - out_channels: ${CH3}
      kernel_size: 3
      stride: 2
      padding: 1
      norm: batch
      act: relu
      dropout: 0.0
EOF
    ;;
  
  "skip")
    # 4-layer with skip connections simulation (using dropout for regularization)
    cat >> "${CONFIG_FILE}" << EOF
    - out_channels: ${CH1}
      kernel_size: 3
      stride: 2
      padding: 1
      norm: batch
      act: relu
      dropout: 0.1
    - out_channels: ${CH2}
      kernel_size: 3
      stride: 2
      padding: 1
      norm: batch
      act: relu
      dropout: 0.1
    - out_channels: ${CH3}
      kernel_size: 3
      stride: 2
      padding: 1
      norm: batch
      act: relu
      dropout: 0.1
    - out_channels: ${CH3}
      kernel_size: 3
      stride: 2
      padding: 1
      norm: batch
      act: relu
      dropout: 0.1
EOF
    ;;
  
  "deep")
    # 5-layer deep encoder
    CH4=$((CH3 + CH3/2))
    CH5=$((CH3 * 2))
    cat >> "${CONFIG_FILE}" << EOF
    - out_channels: ${CH1}
      kernel_size: 3
      stride: 2
      padding: 1
      norm: batch
      act: relu
      dropout: 0.0
    - out_channels: ${CH2}
      kernel_size: 3
      stride: 2
      padding: 1
      norm: batch
      act: relu
      dropout: 0.0
    - out_channels: ${CH3}
      kernel_size: 3
      stride: 2
      padding: 1
      norm: batch
      act: relu
      dropout: 0.0
    - out_channels: ${CH4}
      kernel_size: 3
      stride: 2
      padding: 1
      norm: batch
      act: relu
      dropout: 0.0
    - out_channels: ${CH5}
      kernel_size: 3
      stride: 2
      padding: 1
      norm: batch
      act: relu
      dropout: 0.0
EOF
    ;;
esac

# Add decoder config
cat >> "${CONFIG_FILE}" << EOF

decoder:
  out_channels: 3
  latent_dim: ${LATENT_DIM}
  image_size: 512
EOF

echo "Config file created: ${CONFIG_FILE}"

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
BATCH_SIZE=32
EPOCHS=50  # Reduced for sweep
LEARNING_RATE=0.001
RESIZE=512
LAYOUT_MODE="scene"
LOSS="mse"

# =============================================================================
# CONDA ACTIVATION
# =============================================================================
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate scenefactor || {
    echo "Failed to activate conda environment" >&2
    exit 1
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
  --save_every 10 \
  --keep_only_best || {
  echo "ERROR: Training failed" >&2
  exit 1
}

echo "=========================================="
echo "Task ${LSB_JOBINDEX}/36 COMPLETED"
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