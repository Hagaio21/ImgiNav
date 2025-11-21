#!/bin/bash
#BSUB -J conditional_crossattention_diffusion
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/conditional_crossattention_diffusion.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/conditional_crossattention_diffusion.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 24:00
#BSUB -q gpuv100

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train_diffusion.py"
CONFIG="${BASE_DIR}/experiments/diffusion/new_layouts/conditional_crossattention_diffusion.yaml"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Validate config file exists
if [ ! -f "${CONFIG}" ]; then
  echo "ERROR: Config file not found: ${CONFIG}" >&2
  exit 1
fi

# =============================================================================
# MODULES
# =============================================================================
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64
# PyTorch memory management to avoid fragmentation
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
echo "Conditional Cross-Attention Diffusion Training"
echo "UNet with Cross-Attention on Embeddings"
echo "=========================================="
echo "Config: ${CONFIG}"
echo "Working directory: ${BASE_DIR}"
echo "Start: $(date)"
echo "=========================================="
echo ""
echo "Experiment Overview:"
echo "  - Model: Conditional Diffusion with Cross-Attention"
echo "  - Conditioning: Text embeddings (graph) + POV embeddings"
echo "  - UNet: UnetWithAttention (base_channels=96, depth=3)"
echo "  - Cross-attention: Enabled at bottleneck, downs, and ups"
echo "  - Embedding projection: Converts 1D embeddings to spatial features"
echo "  - Dataset: ControlNet dataset (manifest_with_embeddings.csv)"
echo "    * Contains graph_embedding_path and pov_embedding_path"
echo "    * Embeddings: 384-dim (graph) + 512-dim (POV)"
echo "=========================================="

cd "${BASE_DIR}"

# Validate Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
  exit 1
fi

# Run training
# The script will automatically resume from latest checkpoint if available
python "${PYTHON_SCRIPT}" "${CONFIG}" --resume

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "Training COMPLETE - SUCCESS"
  echo "Config: $(basename ${CONFIG})"
  echo "End: $(date)"
  echo "=========================================="
else
  echo ""
  echo "=========================================="
  echo "Training FAILED with exit code: ${EXIT_CODE}"
  echo "Config: $(basename ${CONFIG})"
  echo "End: $(date)"
  echo "=========================================="
  exit $EXIT_CODE
fi

