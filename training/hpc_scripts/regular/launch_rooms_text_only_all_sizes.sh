#!/bin/bash
# Launch script for rooms-only experiments with text/graph conditioning only (no POV)
# Runs small, medium, and large sizes with down+bottleneck cross attention

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
TRAIN_SCRIPT="${SCRIPT_DIR}/../run_train_diff_clip.sh"

CONFIGS=(
    "experiments/diffusion/clip/regular_rooms/small_down_bottleneck_text_only.yaml"
    "experiments/diffusion/clip/regular_rooms/medium_down_bottleneck_text_only.yaml"
    "experiments/diffusion/clip/regular_rooms/large_down_bottleneck_text_only.yaml"
)

echo "=============================================================================="
echo "Launching Rooms-Only Experiments (Text/Graph Only, Down+Bottleneck Attention)"
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
        -n 4 \
        -R "rusage[mem=8000]" \
        -gpu "num=1" \
        -W 24:00 \
        -q gpuv100 \
        bash "${TRAIN_SCRIPT}" "${config}"
    
    sleep 1
done

echo "Done! Submitted ${#CONFIGS[@]} jobs to gpuv100 queue"

