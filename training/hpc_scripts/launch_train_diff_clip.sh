#!/bin/bash
# Launch script for CLIP Diffusion Training
# This script can launch a single experiment or batch of experiments

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# =============================================================================
# CONFIGURATION
# =============================================================================
TRAIN_SCRIPT="${SCRIPT_DIR}/run_train_diff_clip.sh"

# =============================================================================
# VALIDATION
# =============================================================================
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "ERROR: Training script not found: ${TRAIN_SCRIPT}" >&2
    exit 1
fi

# Make script executable
chmod +x "${TRAIN_SCRIPT}"

# Verify the script is actually executable
if [ ! -x "${TRAIN_SCRIPT}" ]; then
    echo "ERROR: Training script is not executable: ${TRAIN_SCRIPT}" >&2
    exit 1
fi

# =============================================================================
# USAGE
# =============================================================================
if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_path> [config_path2] [config_path3] ..."
    echo ""
    echo "Examples:"
    echo "  # Launch single experiment:"
    echo "  $0 experiments/diffusion/clip/regular/small_down.yaml"
    echo ""
    echo "  # Launch multiple experiments:"
    echo "  $0 experiments/diffusion/clip/regular/small_*.yaml"
    echo ""
    echo "  # Launch all regular CLIP experiments:"
    echo "  $0 experiments/diffusion/clip/regular/*.yaml"
    echo ""
    echo "  # Launch all spatial CLIP experiments:"
    echo "  $0 experiments/diffusion/clip/spatial/*.yaml"
    echo ""
    exit 1
fi

# =============================================================================
# MAIN
# =============================================================================
echo "=============================================================================="
echo "Launching CLIP Diffusion Training"
echo "=============================================================================="
echo ""
echo "Script directory: ${SCRIPT_DIR}"
echo "Base directory: ${BASE_DIR}"
echo ""
echo "Configs to launch:"
for config in "$@"; do
    config_path="${BASE_DIR}/${config}"
    if [ -f "${config_path}" ]; then
        exp_name=$(python3 -c "
import yaml
try:
    with open('${config_path}', 'r') as f:
        config_data = yaml.safe_load(f)
        print(config_data.get('experiment', {}).get('name', 'unnamed'))
except:
    print('unnamed')
" 2>/dev/null || echo "unnamed")
        echo "  - ${config} (${exp_name})"
    else
        echo "  - ${config} (NOT FOUND - will be skipped)"
    fi
done
echo ""
echo "=============================================================================="
echo ""

# Prompt for confirmation
read -p "Submit ${#} job(s)? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# =============================================================================
# SUBMIT JOBS
# =============================================================================
echo ""
echo "Submitting jobs..."
echo "=============================================================================="

SUBMITTED=0
FAILED=0
JOB_IDS=()

for config in "$@"; do
    config_path="${BASE_DIR}/${config}"
    
    # Skip if config doesn't exist
    if [ ! -f "${config_path}" ]; then
        echo "SKIPPING: ${config} (file not found)"
        ((FAILED++))
        continue
    fi
    
    # Extract experiment name for job name
    exp_name=$(python3 -c "
import yaml
import re
try:
    with open('${config_path}', 'r') as f:
        config_data = yaml.safe_load(f)
        exp_name = config_data.get('experiment', {}).get('name', 'unnamed')
        # Sanitize for job name (bsub job names have limits)
        exp_name = re.sub(r'[^a-zA-Z0-9_]', '_', exp_name)
        exp_name = re.sub(r'_+', '_', exp_name).strip('_')
        # Truncate if too long (bsub limit is ~64 chars)
        if len(exp_name) > 50:
            exp_name = exp_name[:50]
        print(exp_name)
except:
    print('unnamed')
" 2>/dev/null || echo "unnamed")
    
    # Sanitize config path for log filename
    log_suffix=$(echo "${config}" | sed 's/[^a-zA-Z0-9]/_/g' | sed 's/_\+/_/g')
    
    echo ""
    echo "Submitting: ${config}"
    echo "  Experiment: ${exp_name}"
    
    JOB_OUTPUT=$(bsub -J "${exp_name}" \
        -o "${BASE_DIR}/training/hpc_scripts/logs/train_diff_clip_${log_suffix}.%J.out" \
        -e "${BASE_DIR}/training/hpc_scripts/logs/train_diff_clip_${log_suffix}.%J.err" \
        -n 4 \
        -R "rusage[mem=8000]" \
        -gpu "num=1" \
        -W 24:00 \
        -q gpuv100 \
        bash "${TRAIN_SCRIPT}" "${config}" 2>&1)
    
    BSUB_EXIT_CODE=$?
    if [ $BSUB_EXIT_CODE -eq 0 ]; then
        # Extract job ID from bsub output
        JOB_ID=$(echo "${JOB_OUTPUT}" | grep -oP 'Job <\K[0-9]+(?=>)' || echo "")
        if [ -n "${JOB_ID}" ]; then
            echo "  SUCCESS - Job ID: ${JOB_ID}"
            JOB_IDS+=("${JOB_ID}")
            ((SUBMITTED++))
        else
            if echo "${JOB_OUTPUT}" | grep -qi "submitted"; then
                echo "  SUBMITTED (could not extract job ID)"
                ((SUBMITTED++))
            else
                echo "  FAILED - bsub output: ${JOB_OUTPUT}"
                ((FAILED++))
            fi
        fi
    else
        echo "  FAILED (bsub exit code: ${BSUB_EXIT_CODE})"
        echo "  bsub output: ${JOB_OUTPUT}"
        ((FAILED++))
    fi
done

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=============================================================================="
echo "Submission Summary"
echo "=============================================================================="
echo "Submitted: ${SUBMITTED}"
echo "Failed: ${FAILED}"
if [ ${#JOB_IDS[@]} -gt 0 ]; then
    echo ""
    echo "Job IDs:"
    for job_id in "${JOB_IDS[@]}"; do
        echo "  - ${job_id}"
    done
    echo ""
    echo "Monitor jobs with: bjobs ${JOB_IDS[0]}"
    echo "Check logs in: ${BASE_DIR}/training/hpc_scripts/logs/"
fi
echo "=============================================================================="

if [ $FAILED -gt 0 ]; then
    exit 1
fi

