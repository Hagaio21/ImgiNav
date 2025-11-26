#!/bin/bash
# Automatically launch all unfinished experiments

# Default values
BASE_DIR="${1:-/work3/s233249/ImgiNav/experiments/clip}"
DEFAULT_TARGET_EPOCHS="${2:-1000}"
REPO_DIR="${3:-/work3/s233249/ImgiNav/ImgiNav}"
DRY_RUN="${4:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${REPO_DIR}/training/hpc_scripts/run_train_diff_clip.sh"
FIND_CONFIG_SCRIPT="${SCRIPT_DIR}/find_experiment_config.sh"

echo "================================================================================"
echo "Launch Unfinished Experiments"
echo "================================================================================"
echo "Base directory: ${BASE_DIR}"
echo "Default target epochs: ${DEFAULT_TARGET_EPOCHS}"
echo "Repository directory: ${REPO_DIR}"
if [ "${DRY_RUN}" = "true" ]; then
    echo "DRY RUN MODE - No jobs will be submitted"
fi
echo ""

# Get list of unfinished experiments by parsing the summary section
echo "Scanning for unfinished experiments..."
temp_output=$(mktemp)
bash "${SCRIPT_DIR}/list_unfinished_experiments.sh" "${BASE_DIR}" "${DEFAULT_TARGET_EPOCHS}" > "${temp_output}" 2>&1
cat "${temp_output}"

# Extract unfinished experiments from summary section
# Also include "not started" experiments
unfinished_list=$(grep -A 1000 "Unfinished experiments:" "${temp_output}" | grep -E "^\s+diff_clip_" | sed 's/^[[:space:]]*//' | sed 's/ - .*$//' || true)

# Also get experiments that haven't started (from the scanning output)
not_started=$(grep "No metrics CSV (not started)" "${temp_output}" | sed 's/^[[:space:]]*//' | sed 's/:.*$//' || true)
if [ -n "${not_started}" ]; then
    unfinished_list="${unfinished_list}"$'\n'"${not_started}"
fi
rm -f "${temp_output}"

if [ -z "${unfinished_list}" ]; then
    echo ""
    echo "No unfinished experiments found!"
    exit 0
fi

# Convert to array
readarray -t unfinished_experiments <<< "${unfinished_list}"

echo "Found ${#unfinished_experiments[@]} unfinished experiments:"
for exp in "${unfinished_experiments[@]}"; do
    echo "  - ${exp}"
done
echo ""

# Launch each experiment
launched=0
failed=0

for exp_name in "${unfinished_experiments[@]}"; do
    echo "Processing: ${exp_name}"
    
    # Find config file
    config_path=$(bash "${FIND_CONFIG_SCRIPT}" "${exp_name}" "${REPO_DIR}" 2>/dev/null)
    
    if [ -z "${config_path}" ]; then
        echo "  ERROR: Could not find config file for ${exp_name}"
        ((failed++))
        continue
    fi
    
    echo "  Config: ${config_path}"
    
    if [ "${DRY_RUN}" = "true" ]; then
        echo "  [DRY RUN] Would launch: ${exp_name}"
        echo "  [DRY RUN] Command: bsub ... bash ${TRAIN_SCRIPT} ${config_path}"
    else
        # Extract experiment name for job name (sanitize)
        job_name=$(echo "${exp_name}" | sed 's/[^a-zA-Z0-9_]/_/g' | sed 's/_\+/_/g')
        if [ ${#job_name} -gt 50 ]; then
            job_name="${job_name:0:50}"
        fi
        
        # Generate log suffix
        log_suffix=$(echo "${config_path}" | sed 's/[^a-zA-Z0-9]/_/g' | sed 's/_\+/_/g')
        
        echo "  Submitting job: ${job_name}"
        
        # Submit job
        bsub -J "${job_name}" \
            -o "${REPO_DIR}/training/hpc_scripts/logs/train_diff_clip_${log_suffix}.%J.out" \
            -e "${REPO_DIR}/training/hpc_scripts/logs/train_diff_clip_${log_suffix}.%J.err" \
            -n 4 \
            -R "rusage[mem=8000]" \
            -gpu "num=1" \
            -W 24:00 \
            -q gpuv100 \
            bash "${TRAIN_SCRIPT}" "${config_path}" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Submitted successfully"
            ((launched++))
        else
            echo "  ✗ Failed to submit"
            ((failed++))
        fi
        
        sleep 1
    fi
    echo ""
done

echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo "Total unfinished: ${#unfinished_experiments[@]}"
if [ "${DRY_RUN}" != "true" ]; then
    echo "Successfully launched: ${launched}"
    echo "Failed to launch: ${failed}"
else
    echo "Would launch: ${#unfinished_experiments[@]}"
fi
echo "================================================================================"

