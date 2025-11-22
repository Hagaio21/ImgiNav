#!/bin/bash
#BSUB -J embed_controlnet_dataset
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/embed_controlnet_dataset.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/embed_controlnet_dataset.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 24:00
#BSUB -q gpuv100

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/embed_controlnet_dataset.py"

# For STEP 1 (creating POV/graph embeddings for CLIP VAE training):
# Don't provide VAE checkpoint - script will create only POV and graph embeddings
# For STEP 2 (creating all embeddings including layouts):
# Uncomment and set VAE paths below

# STEP 1: Create POV and graph embeddings only (no VAE needed)
AE_CHECKPOINT=""
AE_CONFIG=""

# STEP 2: Uncomment to also create layout embeddings (requires VAE checkpoint)
# AE_CONFIG_CLIP="${BASE_DIR}/experiments/autoencoders/new_layouts/new_layouts_VAE_32x32_structural_256_clip.yaml"
# AE_CONFIG_REGULAR="${BASE_DIR}/experiments/autoencoders/new_layouts/new_layouts_VAE_32x32_structural_256.yaml"
# if [ -f "${AE_CONFIG_CLIP}" ]; then
#   AE_CONFIG="${AE_CONFIG_CLIP}"
#   echo "Using CLIP-trained VAE config: ${AE_CONFIG}"
# else
#   AE_CONFIG="${AE_CONFIG_REGULAR}"
#   echo "Using regular VAE config: ${AE_CONFIG}"
# fi
# # Find checkpoint
# AE_EXP_NAME=$(python3 -c "import yaml; f=open('${AE_CONFIG}'); c=yaml.safe_load(f); print(c.get('experiment',{}).get('name','unnamed'))" 2>/dev/null || echo "unnamed")
# AE_SAVE_PATH=$(python3 -c "import yaml; f=open('${AE_CONFIG}'); c=yaml.safe_load(f); print(c.get('experiment',{}).get('save_path','outputs'))" 2>/dev/null || echo "outputs")
# if [ -f "${AE_SAVE_PATH}/${AE_EXP_NAME}_checkpoint_best.pt" ]; then
#   AE_CHECKPOINT="${AE_SAVE_PATH}/${AE_EXP_NAME}_checkpoint_best.pt"
# fi

CONTROLNET_MANIFEST="/work3/s233249/ImgiNav/datasets/controlnet/manifest_seg.csv"

# Shared embedding location - all experiments will use this
SHARED_EMBEDDING_DIR="/work3/s233249/ImgiNav/experiments/shared_embeddings"
OUTPUT_MANIFEST="${SHARED_EMBEDDING_DIR}/manifest_with_embeddings.csv"

LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Validate manifest exists
if [ ! -f "${CONTROLNET_MANIFEST}" ]; then
  echo "ERROR: ControlNet manifest not found: ${CONTROLNET_MANIFEST}" >&2
  exit 1
fi

# If VAE config is provided, find checkpoint
if [ -n "${AE_CONFIG}" ] && [ -f "${AE_CONFIG}" ]; then
  # Find VAE checkpoint
  AE_EXP_NAME=$(python3 -c "
import yaml
with open('${AE_CONFIG}', 'r') as f:
    config = yaml.safe_load(f)
    print(config.get('experiment', {}).get('name', 'unnamed'))
" 2>/dev/null || echo "unnamed")

  AE_SAVE_PATH=$(python3 -c "
import yaml
with open('${AE_CONFIG}', 'r') as f:
    config = yaml.safe_load(f)
    save_path = config.get('experiment', {}).get('save_path', '')
    if save_path:
        print(save_path)
    else:
        print('outputs')
" 2>/dev/null || echo "outputs")

  # Try to find checkpoint
  if [ -f "${AE_SAVE_PATH}/${AE_EXP_NAME}_checkpoint_best.pt" ]; then
    AE_CHECKPOINT="${AE_SAVE_PATH}/${AE_EXP_NAME}_checkpoint_best.pt"
  elif [ -f "${AE_SAVE_PATH}/checkpoints/${AE_EXP_NAME}_checkpoint_best.pt" ]; then
    AE_CHECKPOINT="${AE_SAVE_PATH}/checkpoints/${AE_EXP_NAME}_checkpoint_best.pt"
  elif [ -f "${AE_SAVE_PATH}/${AE_EXP_NAME}_checkpoint_latest.pt" ]; then
    AE_CHECKPOINT="${AE_SAVE_PATH}/${AE_EXP_NAME}_checkpoint_latest.pt"
    echo "WARNING: Using latest checkpoint instead of best"
  elif [ -f "${AE_SAVE_PATH}/checkpoints/${AE_EXP_NAME}_checkpoint_latest.pt" ]; then
    AE_CHECKPOINT="${AE_SAVE_PATH}/checkpoints/${AE_EXP_NAME}_checkpoint_latest.pt"
    echo "WARNING: Using latest checkpoint instead of best"
  else
    echo "ERROR: VAE checkpoint not found in ${AE_SAVE_PATH}" >&2
    echo "  Expected: ${AE_EXP_NAME}_checkpoint_best.pt or ${AE_EXP_NAME}_checkpoint_latest.pt" >&2
    exit 1
  fi
fi

# =============================================================================
# MODULES
# =============================================================================
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
echo "Embedding ControlNet Dataset"
echo "=========================================="
if [ -n "${AE_CHECKPOINT}" ]; then
  echo "Mode: Creating layouts + POVs + graphs embeddings"
  echo "VAE checkpoint: ${AE_CHECKPOINT}"
  echo "VAE config: ${AE_CONFIG}"
else
  echo "Mode: Creating POVs + graphs embeddings only (STEP 1 - no VAE needed)"
fi
echo "Input manifest: ${CONTROLNET_MANIFEST}"
echo "Output manifest: ${OUTPUT_MANIFEST}"
echo "Working directory: ${BASE_DIR}"
echo "Start: $(date)"
echo "=========================================="

cd "${BASE_DIR}"

# Validate Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
  exit 1
fi

# Run embedding
echo ""
echo "Running embedding script..."
echo "=========================================="

if [ -n "${AE_CHECKPOINT}" ]; then
  python "${PYTHON_SCRIPT}" \
    --ae-checkpoint "${AE_CHECKPOINT}" \
    --ae-config "${AE_CONFIG}" \
    --input-manifest "${CONTROLNET_MANIFEST}" \
    --output-manifest "${OUTPUT_MANIFEST}" \
    --batch-size 32 \
    --num-workers 8
else
  python "${PYTHON_SCRIPT}" \
    --input-manifest "${CONTROLNET_MANIFEST}" \
    --output-manifest "${OUTPUT_MANIFEST}" \
    --batch-size 32 \
    --num-workers 8
fi

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "Embedding COMPLETE - SUCCESS"
  echo "=========================================="
  echo "Output manifest: ${OUTPUT_MANIFEST}"
  echo "End: $(date)"
  echo "=========================================="
  echo ""
  echo "This manifest can now be used by all experiments."
  echo "Update your diffusion configs to use: ${OUTPUT_MANIFEST}"
  exit 0
else
  echo ""
  echo "=========================================="
  echo "Embedding FAILED with exit code: ${EXIT_CODE}"
  echo "=========================================="
  exit $EXIT_CODE
fi

