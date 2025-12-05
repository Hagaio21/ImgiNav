#!/bin/bash
# Launch script for 9 Seg Diffusion v2 experiments
# Queue distribution:
# - 2 for gpul40s: large_povs, large_graphs
# - 2 for gpua100: large_both, medium_both
# - 5 for gpuv100: all small (3) + medium_povs, medium_graphs (2)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train_diffusion.py"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

echo "=============================================================================="
echo "Launching 9 Seg Diffusion v2 Experiments"
echo "=============================================================================="
echo ""

# ============================================================================
# gpul40s queue: 2 experiments (large_povs, large_graphs)
# ============================================================================
echo "Submitting 2 jobs to gpul40s queue (large_povs, large_graphs)..."
echo ""

GPUL40S_CONFIGS=(
    "experiments/diffusion/v2/seg/rooms/povs/large_down_bottleneck_v2.yaml"
    "experiments/diffusion/v2/seg/rooms/graphs/large_down_bottleneck_v2.yaml"
)

for config in "${GPUL40S_CONFIGS[@]}"; do
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
    
    echo "Submitting to gpul40s: ${exp_name}"
    echo "  Config: ${config}"
    echo "  Log: train_diff_${log_suffix}.%J.out"
    
    bsub -J "${exp_name}" \
        -o "${LOG_DIR}/train_diff_${log_suffix}.%J.out" \
        -e "${LOG_DIR}/train_diff_${log_suffix}.%J.err" \
        -n 4 \
        -R "rusage[mem=20000]" \
        -gpu "num=1" \
        -W 48:00 \
        -q gpul40s \
        bash -c "cd ${BASE_DIR} && module load cuda/11.8 && module load cudnn/v8.6.0.163-prod-cuda-11.X && export MKL_INTERFACE_LAYER=LP64 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && if [ -f \"\$HOME/miniconda3/etc/profile.d/conda.sh\" ]; then source \"\$HOME/miniconda3/etc/profile.d/conda.sh\" && conda activate imginav || conda activate scenefactor; fi && python ${PYTHON_SCRIPT} ${config_path} --resume"
    
    sleep 2
    echo ""
done

# ============================================================================
# gpua100 queue: 2 experiments (large_both, medium_both)
# ============================================================================
echo "Submitting 2 jobs to gpua100 queue (large_both, medium_both)..."
echo ""

GPUA100_CONFIGS=(
    "experiments/diffusion/v2/seg/rooms/both/large_down_bottleneck_v2.yaml"
    "experiments/diffusion/v2/seg/rooms/both/medium_down_bottleneck_v2.yaml"
)

for config in "${GPUA100_CONFIGS[@]}"; do
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
    
    echo "Submitting to gpua100: ${exp_name}"
    echo "  Config: ${config}"
    echo "  Log: train_diff_${log_suffix}.%J.out"
    
    bsub -J "${exp_name}" \
        -o "${LOG_DIR}/train_diff_${log_suffix}.%J.out" \
        -e "${LOG_DIR}/train_diff_${log_suffix}.%J.err" \
        -n 4 \
        -R "rusage[mem=20000]" \
        -gpu "num=1" \
        -W 48:00 \
        -q gpua100 \
        bash -c "cd ${BASE_DIR} && module load cuda/11.8 && module load cudnn/v8.6.0.163-prod-cuda-11.X && export MKL_INTERFACE_LAYER=LP64 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && if [ -f \"\$HOME/miniconda3/etc/profile.d/conda.sh\" ]; then source \"\$HOME/miniconda3/etc/profile.d/conda.sh\" && conda activate imginav || conda activate scenefactor; fi && python ${PYTHON_SCRIPT} ${config_path} --resume"
    
    sleep 2
    echo ""
done

# ============================================================================
# gpuv100 queue: 5 experiments (all small: 3 + medium_povs, medium_graphs: 2)
# ============================================================================
echo "Submitting 5 jobs to gpuv100 queue (all small: 3 + medium_povs, medium_graphs: 2)..."
echo ""

GPUV100_CONFIGS=(
    "experiments/diffusion/v2/seg/rooms/both/small_down_bottleneck_v2.yaml"
    "experiments/diffusion/v2/seg/rooms/graphs/small_down_bottleneck_v2.yaml"
    "experiments/diffusion/v2/seg/rooms/povs/small_down_bottleneck_v2.yaml"
    "experiments/diffusion/v2/seg/rooms/povs/medium_down_bottleneck_v2.yaml"
    "experiments/diffusion/v2/seg/rooms/graphs/medium_down_bottleneck_v2.yaml"
)

for config in "${GPUV100_CONFIGS[@]}"; do
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
    
    echo "Submitting to gpuv100: ${exp_name}"
    echo "  Config: ${config}"
    echo "  Log: train_diff_${log_suffix}.%J.out"
    
    bsub -J "${exp_name}" \
        -o "${LOG_DIR}/train_diff_${log_suffix}.%J.out" \
        -e "${LOG_DIR}/train_diff_${log_suffix}.%J.err" \
        -n 4 \
        -R "rusage[mem=16000]" \
        -gpu "num=1" \
        -W 48:00 \
        -q gpuv100 \
        bash -c "cd ${BASE_DIR} && module load cuda/11.8 && module load cudnn/v8.6.0.163-prod-cuda-11.X && export MKL_INTERFACE_LAYER=LP64 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && if [ -f \"\$HOME/miniconda3/etc/profile.d/conda.sh\" ]; then source \"\$HOME/miniconda3/etc/profile.d/conda.sh\" && conda activate imginav || conda activate scenefactor; fi && python ${PYTHON_SCRIPT} ${config_path} --resume"
    
    sleep 2
    echo ""
done

echo "=============================================================================="
echo "Done! Submitted 9 jobs:"
echo "  - 2 to gpul40s queue (large_povs, large_graphs)"
echo "  - 2 to gpua100 queue (large_both, medium_both)"
echo "  - 5 to gpuv100 queue (all small: 3 + medium_povs, medium_graphs: 2)"
echo "=============================================================================="
echo "Check job status with: bjobs"
echo "Monitor logs in: ${LOG_DIR}"
echo "=============================================================================="

