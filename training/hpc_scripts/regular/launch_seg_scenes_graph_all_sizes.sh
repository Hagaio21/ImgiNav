#!/bin/bash
# Launch script for Seg Scenes Graph Diffusion experiments (all sizes)
# Submits: small, medium, and large models
# Queue distribution:
# - small: gpuv100 (12 hours)
# - medium: gpuv100 (12 hours)
# - large: gpua100 (12 hours)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train_diffusion.py"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

echo "=============================================================================="
echo "Launching Seg Scenes Graph Diffusion Experiments (all sizes, 12 hours each)"
echo "=============================================================================="
echo ""

# ============================================================================
# gpuv100 queue: 2 experiments (small and medium)
# ============================================================================
echo "Submitting 2 jobs to gpuv100 queue (small and medium)..."
echo ""

GPUV100_CONFIGS=(
    "experiments/diffusion/v2/seg/scenes/graph/small_down_bottleneck.yaml"
    "experiments/diffusion/v2/seg/scenes/graph/medium_down_bottleneck.yaml"
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
        -W 12:00 \
        -q gpuv100 \
        bash -c "cd ${BASE_DIR} && module load cuda/11.8 && module load cudnn/v8.6.0.163-prod-cuda-11.X && export MKL_INTERFACE_LAYER=LP64 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && if [ -f \"\$HOME/miniconda3/etc/profile.d/conda.sh\" ]; then source \"\$HOME/miniconda3/etc/profile.d/conda.sh\" && conda activate imginav || conda activate scenefactor; fi && python ${PYTHON_SCRIPT} ${config_path} --resume"
    
    sleep 2
    echo ""
done

# ============================================================================
# gpua100 queue: 1 experiment (large)
# ============================================================================
echo "Submitting 1 job to gpua100 queue (large)..."
echo ""

GPUA100_CONFIGS=(
    "experiments/diffusion/v2/seg/scenes/graph/large_down_bottleneck.yaml"
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
        -R "rusage[mem=16000]" \
        -gpu "num=1" \
        -W 12:00 \
        -q gpua100 \
        bash -c "cd ${BASE_DIR} && module load cuda/11.8 && module load cudnn/v8.6.0.163-prod-cuda-11.X && export MKL_INTERFACE_LAYER=LP64 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && if [ -f \"\$HOME/miniconda3/etc/profile.d/conda.sh\" ]; then source \"\$HOME/miniconda3/etc/profile.d/conda.sh\" && conda activate imginav || conda activate scenefactor; fi && python ${PYTHON_SCRIPT} ${config_path} --resume"
    
    sleep 2
    echo ""
done

echo "=============================================================================="
echo "Done! Submitted 3 jobs:"
echo "  - 2 to gpuv100 queue (scenes/graph: small, medium)"
echo "  - 1 to gpua100 queue (scenes/graph: large)"
echo "=============================================================================="
echo "Check job status with: bjobs"
echo "Monitor logs in: ${LOG_DIR}"
echo "=============================================================================="

