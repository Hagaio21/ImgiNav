#!/bin/bash
# Debug script for diffusion training pipeline
# Runs sanity checks (VAE round trip, noise schedule, overfit test) for all 3 model sizes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
DEBUG_SCRIPT="${BASE_DIR}/debug_diffusion.py"

# Configs for the 3 model sizes
CONFIGS=(
    "experiments/diffusion/clip/regular_scenes/small_bottleneck.yaml"
    "experiments/diffusion/clip/regular_scenes/medium_bottleneck.yaml"
    "experiments/diffusion/clip/regular_scenes/large_bottleneck.yaml"
)

echo "=============================================================================="
echo "Debugging Diffusion Training Pipeline"
echo "=============================================================================="
echo "Running debug tests for ${#CONFIGS[@]} model sizes"
echo "Date: $(date)"
echo "=============================================================================="

cd "${BASE_DIR}"

for config in "${CONFIGS[@]}"; do
    config_path="${BASE_DIR}/${config}"
    
    if [ ! -f "${config_path}" ]; then
        echo "WARNING: Config not found: ${config} (skipping)"
        continue
    fi
    
    # Extract model size from config path
    model_size=$(basename "${config}" | sed 's/_bottleneck.yaml//' | sed 's/.*_//')
    exp_name=$(python3 -c "
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
    echo "Running debug tests for: ${model_size} model (${exp_name})"
    echo "Config: ${config}"
    echo "=============================================================================="
    
    # Create output directory for debug results
    output_dir="${BASE_DIR}/debug_outputs/${model_size}"
    mkdir -p "${output_dir}"
    
    # Run debug script
    python "${DEBUG_SCRIPT}" "${config_path}" 2>&1 | tee "${output_dir}/debug_${model_size}.log"
    
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

echo "=============================================================================="
echo "All debug tests completed!"
echo "=============================================================================="

