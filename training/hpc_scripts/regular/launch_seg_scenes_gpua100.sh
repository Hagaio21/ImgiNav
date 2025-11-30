#!/bin/bash
# Launch script for Seg Scenes Diffusion experiments - gpua100 queue
# Submits 6 scenes experiments:
# - scenes/both/small_down_bottleneck.yaml
# - scenes/graph/small_down_bottleneck.yaml
# - scenes/povs/small_down_bottleneck.yaml
# - scenes/both/medium_down_bottleneck.yaml
# - scenes/graph/medium_down_bottleneck.yaml
# - scenes/povs/medium_down_bottleneck.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train_diffusion.py"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Seg scenes experiments (6 total)
CONFIGS=(
    # Small (3)
    "experiments/diffusion/v2/seg/scenes/both/small_down_bottleneck.yaml"
    "experiments/diffusion/v2/seg/scenes/graph/small_down_bottleneck.yaml"
    "experiments/diffusion/v2/seg/scenes/povs/small_down_bottleneck.yaml"
    # Medium (3)
    "experiments/diffusion/v2/seg/scenes/both/medium_down_bottleneck.yaml"
    "experiments/diffusion/v2/seg/scenes/graph/medium_down_bottleneck.yaml"
    "experiments/diffusion/v2/seg/scenes/povs/medium_down_bottleneck.yaml"
)

echo "=============================================================================="
echo "Launching Seg Scenes Diffusion Training (gpua100 queue)"
echo "=============================================================================="
echo "Submitting ${#CONFIGS[@]} jobs..."
echo ""

for config in "${CONFIGS[@]}"; do
    config_path="${BASE_DIR}/${config}"
    if [ ! -f "${config_path}" ]; then
        echo "WARNING: Config not found: ${config_path}"
        continue
    fi
    
    # Extract experiment name from config
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
except Exception as e:
    print('unnamed')
" 2>/dev/null || echo "unnamed")
    
    # Create log suffix from config path
    log_suffix=$(echo "${config}" | sed 's|experiments/diffusion/v2/||' | sed 's|/|_|g' | sed 's|\.yaml||')
    
    echo "Submitting: ${exp_name}"
    echo "  Config: ${config}"
    echo "  Log: train_diff_${log_suffix}.%J.out"
    
    bsub -J "${exp_name}" \
        -o "${LOG_DIR}/train_diff_${log_suffix}.%J.out" \
        -e "${LOG_DIR}/train_diff_${log_suffix}.%J.err" \
        -n 4 \
        -R "rusage[mem=16000]" \
        -gpu "num=1" \
        -W 24:00 \
        -q gpua100 \
        bash -c "cd ${BASE_DIR} && module load cuda/11.8 && module load cudnn/v8.6.0.163-prod-cuda-11.X && export MKL_INTERFACE_LAYER=LP64 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && if [ -f \"\$HOME/miniconda3/etc/profile.d/conda.sh\" ]; then source \"\$HOME/miniconda3/etc/profile.d/conda.sh\" && conda activate imginav || conda activate scenefactor; fi && python ${PYTHON_SCRIPT} ${config_path} --resume"
    
    sleep 2
    echo ""
done

echo "=============================================================================="
echo "Done! Submitted ${#CONFIGS[@]} jobs to gpua100 queue"
echo "=============================================================================="
echo "Check job status with: bjobs"
echo "Monitor logs in: ${LOG_DIR}"
echo "=============================================================================="

