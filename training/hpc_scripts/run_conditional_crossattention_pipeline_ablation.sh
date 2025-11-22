#!/bin/bash
#BSUB -J conditional_crossattention_pipeline_ablation
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/conditional_crossattention_pipeline_ablation.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/conditional_crossattention_pipeline_ablation.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 24:00
#BSUB -q gpuv100

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
# Get experiment name from first argument
EXPERIMENT_NAME="${1:-conditional_crossattention_diffusion_downs}"
MODEL_SIZE="${2:-small}"  # small or large
CURRENT_ATTEMPT="${3:-1}"  # Current resubmission attempt (default: 1)

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train_pipeline_conditional_crossattention.py"
AE_CONFIG="${BASE_DIR}/experiments/autoencoders/new_layouts/new_layouts_VAE_32x32_structural_256.yaml"
DIFFUSION_CONFIG="${BASE_DIR}/experiments/diffusion/new_layouts/${EXPERIMENT_NAME}.yaml"
CONTROLNET_MANIFEST="/work3/s233249/ImgiNav/datasets/controlnet/manifest_seg.csv"
SHARED_EMBEDDING_MANIFEST="/work3/s233249/ImgiNav/experiments/shared_embeddings/manifest_with_embeddings.csv"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"
SCRIPT_DIR="${BASE_DIR}/training/hpc_scripts"
MAX_RESUBMITS=5  # Maximum number of auto-resubmissions
RESUBMIT_DELAY=300  # Wait 5 minutes before resubmitting

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Validate config files exist
if [ ! -f "${AE_CONFIG}" ]; then
  echo "ERROR: VAE config file not found: ${AE_CONFIG}" >&2
  exit 1
fi

if [ ! -f "${DIFFUSION_CONFIG}" ]; then
  echo "ERROR: Diffusion config file not found: ${DIFFUSION_CONFIG}" >&2
  exit 1
fi

# =============================================================================
# AUTO-RESUBMISSION FUNCTION
# =============================================================================
resubmit_job() {
  local attempt=$1
  local exit_code=$2
  
  if [ $attempt -gt $MAX_RESUBMITS ]; then
    echo "ERROR: Maximum resubmission attempts (${MAX_RESUBMITS}) reached. Giving up." >&2
    return 1
  fi
  
  echo ""
  echo "=========================================="
  echo "Job failed (attempt ${attempt}/${MAX_RESUBMITS})"
  echo "Exit code: ${exit_code}"
  echo "Waiting ${RESUBMIT_DELAY} seconds before resubmitting..."
  echo "=========================================="
  
  sleep $RESUBMIT_DELAY
  
  # Determine memory requirement based on model size
  if [ "${MODEL_SIZE}" = "large" ]; then
    MEM_REQ="rusage[mem=24000]"
  else
    MEM_REQ="rusage[mem=16000]"
  fi
  
  # Resubmit the same job using bsub
  cd "${SCRIPT_DIR}"
  ABLATION_SCRIPT="${SCRIPT_DIR}/run_conditional_crossattention_pipeline_ablation.sh"
  RESUBMIT_OUTPUT=$(bsub -J "ablation_${EXPERIMENT_NAME}_retry${attempt}" \
    -o "${LOG_DIR}/ablation_${EXPERIMENT_NAME}_retry${attempt}.%J.out" \
    -e "${LOG_DIR}/ablation_${EXPERIMENT_NAME}_retry${attempt}.%J.err" \
    -n 8 \
    -R "${MEM_REQ}" \
    -gpu "num=1" \
    -W 24:00 \
    -q gpuv100 \
    bash "${ABLATION_SCRIPT}" "${EXPERIMENT_NAME}" "${MODEL_SIZE}" "${attempt}" 2>&1)
  
  RESUBMIT_EXIT=$?
  if [ $RESUBMIT_EXIT -eq 0 ]; then
    echo "Job resubmitted successfully (attempt ${attempt})"
    echo "bsub output: ${RESUBMIT_OUTPUT}"
    exit 0
  else
    echo "ERROR: Failed to resubmit job (bsub exit code: ${RESUBMIT_EXIT})" >&2
    echo "bsub output: ${RESUBMIT_OUTPUT}" >&2
    return 1
  fi
}


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
echo "Conditional Cross-Attention Diffusion Pipeline (ABLATION)"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Model Size: ${MODEL_SIZE}"
echo "=========================================="
echo "VAE config: ${AE_CONFIG}"
echo "Diffusion config: ${DIFFUSION_CONFIG}"
echo "ControlNet manifest: ${CONTROLNET_MANIFEST}"
echo "Working directory: ${BASE_DIR}"
echo "Start: $(date)"
echo "=========================================="

cd "${BASE_DIR}"

# Validate Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
  exit 1
fi

# Check for shared embedding
if [ -f "${SHARED_EMBEDDING_MANIFEST}" ]; then
  echo ""
  echo "=========================================="
  echo "Found shared embedding manifest"
  echo "=========================================="
  echo "Using shared embedding: ${SHARED_EMBEDDING_MANIFEST}"
  echo "Pipeline will skip embedding step and use shared embeddings"
  echo "=========================================="
fi

# Run pipeline
echo ""
echo "Pipeline attempt ${CURRENT_ATTEMPT}/${MAX_RESUBMITS}"
echo "=========================================="

python "${PYTHON_SCRIPT}" \
  --ae-config "${AE_CONFIG}" \
  --diffusion-config "${DIFFUSION_CONFIG}" \
  --controlnet-manifest "${CONTROLNET_MANIFEST}" \
  --batch-size 32 \
  --num-workers 8

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "Pipeline COMPLETE - SUCCESS"
  echo "Experiment: ${EXPERIMENT_NAME}"
  echo "Model Size: ${MODEL_SIZE}"
  echo "Attempt: ${CURRENT_ATTEMPT}"
  echo "End: $(date)"
  echo "=========================================="
  exit 0
else
  echo ""
  echo "=========================================="
  echo "Pipeline FAILED with exit code: ${EXIT_CODE}"
  echo "Experiment: ${EXPERIMENT_NAME}"
  echo "Model Size: ${MODEL_SIZE}"
  echo "Attempt: ${CURRENT_ATTEMPT}/${MAX_RESUBMITS}"
  echo "=========================================="
  
  # Check if embedding stage completed (allows safe resubmission)
  # The pipeline creates a flags file in the experiment directory
  # If embedding is done, we can resubmit and the pipeline will resume from diffusion training
  DIFFUSION_SAVE_PATH=$(python3 -c "
import yaml
with open('${DIFFUSION_CONFIG}', 'r') as f:
    config = yaml.safe_load(f)
    print(config.get('experiment', {}).get('save_path', ''))
" 2>/dev/null || echo "")
  
  EMBEDDING_COMPLETED=0
  # Check if shared embedding exists (always safe to resubmit)
  if [ -f "${SHARED_EMBEDDING_MANIFEST}" ]; then
    EMBEDDING_COMPLETED=1
    echo "✓ Shared embedding exists - embedding stage always completed"
  elif [ -n "${DIFFUSION_SAVE_PATH}" ] && [ -d "${DIFFUSION_SAVE_PATH}" ]; then
    FLAGS_FILE="${DIFFUSION_SAVE_PATH}/flags.txt"
    if [ -f "${FLAGS_FILE}" ]; then
      # Check if EMBEDDED_MANIFEST flag exists (means embedding completed)
      if grep -q "EMBEDDED_MANIFEST" "${FLAGS_FILE}" 2>/dev/null; then
        EMBEDDING_COMPLETED=1
        echo "✓ Embedding stage completed (found EMBEDDED_MANIFEST flag)"
      fi
    fi
  fi
  
  # Resubmit if embedding completed (regardless of error type)
  # The pipeline will resume from diffusion training stage
  if [ $EMBEDDING_COMPLETED -eq 1 ] && [ $CURRENT_ATTEMPT -lt $MAX_RESUBMITS ]; then
    echo ""
    echo "=========================================="
    echo "RESUBMITTING JOB"
    echo "=========================================="
    echo "Embedding stage completed: YES"
    echo "Current attempt: ${CURRENT_ATTEMPT}/${MAX_RESUBMITS}"
    echo "Exit code: ${EXIT_CODE}"
    echo ""
    echo "Resubmitting job - pipeline will resume from diffusion training stage"
    echo "=========================================="
    NEXT_ATTEMPT=$((CURRENT_ATTEMPT + 1))
    resubmit_job $NEXT_ATTEMPT $EXIT_CODE
  elif [ $EMBEDDING_COMPLETED -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "NO RESUBMISSION"
    echo "=========================================="
    echo "Embedding stage not completed."
    echo "Cannot safely resubmit - embedding would need to restart."
    echo "Exiting without resubmission."
    echo "=========================================="
    exit $EXIT_CODE
  else
    echo ""
    echo "=========================================="
    echo "NO RESUBMISSION"
    echo "=========================================="
    echo "Maximum resubmission attempts (${MAX_RESUBMITS}) reached."
    echo "Exiting without resubmission."
    echo "=========================================="
    exit $EXIT_CODE
  fi
fi

