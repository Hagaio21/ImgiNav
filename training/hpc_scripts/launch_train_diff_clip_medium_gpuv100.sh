#!/bin/bash
# Launch script for Medium CLIP Diffusion models on gpuv100 queue
# Submits: medium_all, medium_down, medium_up, medium_bottleneck

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
TRAIN_SCRIPT="${SCRIPT_DIR}/run_train_diff_clip.sh"

CONFIGS=(
    "experiments/diffusion/clip/regular/medium_all.yaml"
    "experiments/diffusion/clip/regular/medium_down.yaml"
    "experiments/diffusion/clip/regular/medium_up.yaml"
    "experiments/diffusion/clip/regular/medium_bottleneck.yaml"
)

echo "=============================================================================="
echo "Launching Medium CLIP Diffusion Training (gpuv100 queue)"
echo "=============================================================================="
echo "Submitting ${#CONFIGS[@]} jobs..."

for config in "${CONFIGS[@]}"; do
    config_path="${BASE_DIR}/${config}"
    if [ ! -f "${config_path}" ]; then
        echo "WARNING: Config not found: ${config}"
        continue
    fi
    
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
    
    echo "Submitting: ${config} (${exp_name})"
    
    bsub -J "${exp_name}" \
        -o "${BASE_DIR}/training/hpc_scripts/logs/train_diff_clip_${log_suffix}.%J.out" \
        -e "${BASE_DIR}/training/hpc_scripts/logs/train_diff_clip_${log_suffix}.%J.err" \
        -n 8 \
        -R "rusage[mem=16000]" \
        -gpu "num=1" \
        -W 24:00 \
        -q gpuv100 \
        bash "${TRAIN_SCRIPT}" "${config}"
    
    sleep 1
done

echo "Done! Submitted ${#CONFIGS[@]} jobs to gpuv100 queue"

