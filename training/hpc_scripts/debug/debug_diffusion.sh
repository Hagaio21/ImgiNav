#!/bin/bash
# Debug script for diffusion training pipeline
# Runs sanity checks (VAE round trip, noise schedule, overfit test) for all 3 model sizes

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
DEBUG_SCRIPT="${BASE_DIR}/debug_diffusion.py"

# Configs for all combinations: 3 model sizes × 2 dataset types (rooms/scenes)
CONFIGS=(
    "experiments/diffusion/clip/regular_rooms/small_down_bottleneck.yaml"
    "experiments/diffusion/clip/regular_rooms/medium_down_bottleneck.yaml"
    "experiments/diffusion/clip/regular_rooms/large_down_bottleneck.yaml"
    "experiments/diffusion/clip/regular_scenes/small_down_bottleneck.yaml"
    "experiments/diffusion/clip/regular_scenes/medium_down_bottleneck.yaml"
    "experiments/diffusion/clip/regular_scenes/large_down_bottleneck.yaml"
)

# Validate debug script exists
if [ ! -f "${DEBUG_SCRIPT}" ]; then
  echo "ERROR: Debug script not found: ${DEBUG_SCRIPT}" >&2
  exit 1
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
else
  echo "WARNING: conda.sh not found, trying to activate environment anyway..." >&2
  conda activate imginav || conda activate scenefactor || {
    echo "ERROR: Failed to activate conda environment" >&2
    exit 1
  }
fi

# =============================================================================
# RUN
# =============================================================================
echo "=============================================================================="
echo "Debugging Diffusion Training Pipeline"
echo "=============================================================================="
echo "Running debug tests for ${#CONFIGS[@]} configurations"
echo "  (3 model sizes × 2 dataset types: rooms + scenes)"
echo "Working directory: ${BASE_DIR}"
echo "Python: $(which python)"
echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
echo "Start: $(date)"
echo "=============================================================================="

cd "${BASE_DIR}"

for config in "${CONFIGS[@]}"; do
    config_path="${BASE_DIR}/${config}"
    
    if [ ! -f "${config_path}" ]; then
        echo "WARNING: Config not found: ${config} (skipping)"
        continue
    fi
    
    # Extract model size and dataset type from config path
    # e.g., regular_rooms/small_bottleneck.yaml -> rooms_small
    dataset_type=$(echo "${config}" | sed 's|.*/regular_||' | sed 's|/.*||')
    model_size=$(basename "${config}" | sed 's/_bottleneck.yaml//' | sed 's/.*_//')
    model_key="${dataset_type}_${model_size}"
    
    exp_name=$(python -c "
import yaml
import re
try:
    with open('${config_path}', 'r') as f:
        config_data = yaml.safe_load(f)
        exp_name = config_data.get('experiment', {}).get('name', 'unnamed')
        exp_name = re.sub(r'[^a-zA-Z0-9_]', '_', exp_name)
        exp_name = re.sub(r'_+', '_', exp_name).strip('_')
        if len(exp_name) > 50:
            exp_name = exp_name[:50]
        print(exp_name)
except:
    print('unnamed')
" 2>/dev/null || echo "unnamed")
    
    log_suffix=$(echo "${config}" | sed 's/[^a-zA-Z0-9]/_/g' | sed 's/_\+/_/g')
    
    echo ""
    echo "=============================================================================="
    echo "Running debug tests for: ${model_size} model on ${dataset_type} (${exp_name})"
    echo "Config: ${config}"
    echo "=============================================================================="
    
    # Create output directory for debug results (organized by dataset_type/model_size)
    output_dir="${BASE_DIR}/debug_outputs/${dataset_type}/${model_size}"
    mkdir -p "${output_dir}"
    
    # Run debug script
    echo "Running: python ${DEBUG_SCRIPT} ${config_path}"
    python "${DEBUG_SCRIPT}" "${config_path}" 2>&1 | tee "${output_dir}/debug_${model_key}.log"
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "WARNING: Debug test failed for ${model_key} (exit code: ${EXIT_CODE})"
    else
        echo "✓ Debug test completed successfully for ${model_key}"
    fi
    
    # Move generated debug images to output directory
    if [ -f "${BASE_DIR}/debug_vae_reconstruction.png" ]; then
        mv "${BASE_DIR}/debug_vae_reconstruction.png" "${output_dir}/"
    fi
    if [ -f "${BASE_DIR}/debug_forward_process.png" ]; then
        mv "${BASE_DIR}/debug_forward_process.png" "${output_dir}/"
    fi
    if [ -f "${BASE_DIR}/debug_overfit_loss.png" ]; then
        mv "${BASE_DIR}/debug_overfit_loss.png" "${output_dir}/"
    fi
    
    echo "Debug results saved to: ${output_dir}"
    echo ""
done

echo ""
echo "=============================================================================="
echo "All debug tests completed!"
echo "End: $(date)"
echo "=============================================================================="
echo ""
echo "Debug results saved to:"
for config in "${CONFIGS[@]}"; do
    dataset_type=$(echo "${config}" | sed 's|.*/regular_||' | sed 's|/.*||')
    model_size=$(basename "${config}" | sed 's/_bottleneck.yaml//' | sed 's/.*_//')
    echo "  - ${BASE_DIR}/debug_outputs/${dataset_type}/${model_size}/"
done
echo "=============================================================================="

