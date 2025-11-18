#!/bin/bash
#BSUB -J phase2_3_diffusion_decreasing_cfg_unet48
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/phase2_3_diffusion_decreasing_cfg_unet48.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/phase2_3_diffusion_decreasing_cfg_unet48.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"  # 64x64x4 latents are much smaller than 512x512 images
#BSUB -gpu "num=1"
#BSUB -W 48:00  # Extended wall time for diffusion training (500 epochs)
#BSUB -q gpul40s

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train_diffusion.py"
DIFFUSION_CONFIG="${BASE_DIR}/experiments/diffusion/phase2/phase2_3_diffusion_64x64_bottleneck_attn_unet48_256_decreasing_cfg.yaml"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Validate config file exists
if [ ! -f "${DIFFUSION_CONFIG}" ]; then
  echo "ERROR: Diffusion config file not found: ${DIFFUSION_CONFIG}" >&2
  exit 1
fi

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
echo "Phase 2.3: Diffusion Training with Decreasing CFG Dropout (New Layouts)"
echo "=========================================="
echo "Diffusion config: ${DIFFUSION_CONFIG}"
echo "Working directory: ${BASE_DIR}"
echo "Start: $(date)"
echo "=========================================="
echo ""
echo "Experiment Features:"
echo "  - Decreasing CFG dropout schedule: 1.0 → 0.1 (linear)"
echo "  - Changes every 10 epochs, plateaus at 0.1 after epoch 200"
echo "  - Starts with fully unconditional training"
echo "  - Gradually transitions to conditional training"
echo "  - UNet48 architecture (base_channels=48)"
echo ""
echo "Training Details:"
echo "  - Uses VAE from new_layouts_VAE_64x64_structural_256"
echo "  - Loads from unconditional checkpoint (Stage 1)"
echo "  - 500 epochs"
echo "  - Batch size: 128"
echo "  - 1000 noise steps with LinearScheduler"
echo "  - CFG dropout tracked and plotted during training"
echo "=========================================="

cd "${BASE_DIR}"

# Validate Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
  exit 1
fi

# Check if embeddings exist (they should be created first if using pipeline)
EMBEDDING_MANIFEST="/work3/s233249/ImgiNav/experiments/new_layouts/new_layouts_diffusion_64x64_bottleneck_attn_unet48_256_unconditional/embeddings/manifest_with_latents.csv"
if [ ! -f "${EMBEDDING_MANIFEST}" ]; then
  echo ""
  echo "WARNING: Embeddings not found at: ${EMBEDDING_MANIFEST}"
  echo "You may need to create embeddings first using train_pipeline_phase2.py"
  echo "or ensure the manifest path in the config is correct."
  echo ""
  echo "Continuing with training (will fail if embeddings are missing)..."
  echo ""
fi

# Run diffusion training
python "${PYTHON_SCRIPT}" "${DIFFUSION_CONFIG}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "Diffusion Training COMPLETE - SUCCESS"
  echo "Config: $(basename ${DIFFUSION_CONFIG})"
  echo "End: $(date)"
  echo "=========================================="
else
  echo ""
  echo "=========================================="
  echo "Diffusion Training FAILED with exit code: ${EXIT_CODE}"
  echo "Config: $(basename ${DIFFUSION_CONFIG})"
  echo "End: $(date)"
  echo "=========================================="
  exit $EXIT_CODE
fi

