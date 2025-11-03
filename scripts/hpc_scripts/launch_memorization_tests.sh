#!/bin/bash
# Launch memorization check jobs for all currently training diffusion models

set -euo pipefail

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
ABLATION_DIR="${BASE_DIR}/experiments/diffusion/ablation"
MANIFEST_PATH="/work3/s233249/ImgiNav/datasets/layouts_latents.csv"
OUTPUT_BASE="/work3/s233249/ImgiNav/outputs/memorization_checks"
JOB_SCRIPT="${BASE_DIR}/scripts/hpc_scripts/run_memorization_check.sh"

# List of config files to test
declare -a CONFIG_FILES=(
    "${ABLATION_DIR}/capacity_unet64_d4.yaml"
    "${ABLATION_DIR}/capacity_unet64_d5.yaml"
    "${ABLATION_DIR}/capacity_unet128_d3.yaml"
    "${ABLATION_DIR}/capacity_unet128_d4.yaml"
    "${ABLATION_DIR}/capacity_unet256_d4.yaml"
    "${ABLATION_DIR}/scheduler_linear.yaml"
)

# Function to find the best checkpoint for a config
find_checkpoint() {
    local config_file="$1"
    
    # Extract experiment name (can be on line after experiment: or anywhere in experiment section)
    local exp_name=$(grep "^experiment:" -A 10 "$config_file" | grep "name:" | head -1 | awk '{print $2}' | tr -d '"' | tr -d "'")
    
    # Extract save_path (can be on line after experiment: or anywhere in experiment section)
    local save_path=$(grep "^experiment:" -A 10 "$config_file" | grep "save_path:" | head -1 | awk '{print $2}' | tr -d '"' | tr -d "'")
    
    # Validate we got both values
    if [ -z "${exp_name}" ] || [ -z "${save_path}" ]; then
        echo ""
        return 1
    fi
    
    # Check if save_path exists
    if [ ! -d "${save_path}" ]; then
        echo ""
        return 1
    fi
    
    # Try best checkpoint first
    local best_ckpt="${save_path}/${exp_name}_checkpoint_best.pt"
    if [ -f "${best_ckpt}" ]; then
        echo "${best_ckpt}"
        return 0
    fi
    
    # Try latest checkpoint
    local latest_ckpt="${save_path}/${exp_name}_checkpoint_latest.pt"
    if [ -f "${latest_ckpt}" ]; then
        echo "${latest_ckpt}"
        return 0
    fi
    
    # Try any checkpoint
    local any_ckpt=$(find "${save_path}" -name "${exp_name}_checkpoint_*.pt" 2>/dev/null | head -1)
    if [ -n "${any_ckpt}" ] && [ -f "${any_ckpt}" ]; then
        echo "${any_ckpt}"
        return 0
    fi
    
    # No checkpoint found
    echo ""
    return 1
}

echo "========================================="
echo "Launching Memorization Tests"
echo "========================================="
echo ""

submitted=0
skipped=0

for config_file in "${CONFIG_FILES[@]}"; do
    if [ ! -f "${config_file}" ]; then
        echo "⚠ Skipping: Config not found - ${config_file}"
        ((skipped++))
        continue
    fi
    
    # Extract experiment name (more robust parsing)
    exp_name=$(grep "^experiment:" -A 10 "${config_file}" | grep "name:" | head -1 | awk '{print $2}' | tr -d '"' | tr -d "'")
    if [ -z "${exp_name}" ]; then
        exp_name=$(basename "${config_file}" .yaml)
    fi
    
    echo "Processing: ${exp_name}"
    
    # Find checkpoint
    checkpoint=$(find_checkpoint "${config_file}")
    if [ -z "${checkpoint}" ]; then
        echo "  ⚠ Skipping: No checkpoint found"
        ((skipped++))
        continue
    fi
    
    echo "  ✓ Checkpoint found: ${checkpoint}"
    
    # Create output directory
    output_dir="${OUTPUT_BASE}/${exp_name}"
    mkdir -p "${output_dir}"
    
    # Submit job
    echo "  → Submitting job..."
    job_id=$(bsub \
        -J "mem_${exp_name}" \
        -o "${output_dir}/job.%J.out" \
        -e "${output_dir}/job.%J.err" \
        -n 4 \
        -R "rusage[mem=16000]" \
        -gpu "num=1" \
        -W 4:00 \
        -q gpuv100 \
        "${JOB_SCRIPT}" \
        "${config_file}" \
        "${checkpoint}" \
        "${MANIFEST_PATH}" \
        "${output_dir}" \
        100 \
        5000 | grep -oP '<\K[0-9]+')
    
    if [ -n "${job_id}" ]; then
        echo "  ✓ Job submitted: ${job_id}"
        echo "    Output: ${output_dir}"
        ((submitted++))
    else
        echo "  ✗ Failed to submit job"
        ((skipped++))
    fi
    
    echo ""
done

echo "========================================="
echo "Summary"
echo "========================================="
echo "  Submitted: ${submitted}"
echo "  Skipped: ${skipped}"
echo "  Total: $((submitted + skipped))"
echo ""
echo "Monitor jobs with: bjobs"
echo "Check results in: ${OUTPUT_BASE}/"

