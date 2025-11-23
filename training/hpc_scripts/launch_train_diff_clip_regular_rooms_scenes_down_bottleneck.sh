#!/bin/bash
# Launch script for CLIP Diffusion Training - Regular CLIP Rooms & Scenes (Bottleneck)
# Launches 2 experiments: regular_rooms (bottleneck) and regular_scenes (bottleneck)

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

# Configs to launch
CONFIGS=(
    # Regular CLIP - Rooms
    "experiments/diffusion/clip/regular_rooms/small_bottleneck.yaml"
    
    # Regular CLIP - Scenes
    "experiments/diffusion/clip/regular_scenes/small_bottleneck.yaml"
)

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

# Validate configs exist
MISSING_CONFIGS=()
for config in "${CONFIGS[@]}"; do
    if [ ! -f "${BASE_DIR}/${config}" ]; then
        MISSING_CONFIGS+=("${config}")
    fi
done

if [ ${#MISSING_CONFIGS[@]} -gt 0 ]; then
    echo "ERROR: Some configs not found:" >&2
    for config in "${MISSING_CONFIGS[@]}"; do
        echo "  - ${config}" >&2
    done
    exit 1
fi

# =============================================================================
# MAIN
# =============================================================================
echo "=============================================================================="
echo "Launching CLIP Diffusion Training - Regular CLIP (Rooms & Scenes, Bottleneck)"
echo "=============================================================================="
echo ""
echo "Script directory: ${SCRIPT_DIR}"
echo "Base directory: ${BASE_DIR}"
echo ""
echo "Experiments to launch (2 total):"
echo ""
echo "Regular CLIP - Rooms:"
echo "  1. small_bottleneck"
echo ""
echo "Regular CLIP - Scenes:"
echo "  2. small_bottleneck"
echo ""
echo "=============================================================================="
echo ""

# Prompt for confirmation
read -p "Submit 2 jobs? (y/N): " -n 1 -r
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

CONFIG_INDEX=0
for config in "${CONFIGS[@]}"; do
    CONFIG_INDEX=$((CONFIG_INDEX + 1))
    config_path="${BASE_DIR}/${config}"
    
    # Verify config exists (should have been validated, but double-check)
    if [ ! -f "${config_path}" ]; then
        echo ""
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
except Exception as e:
    print('unnamed')
" 2>/dev/null || echo "unnamed")
    
    # Add index to job name to ensure uniqueness
    # This prevents duplicate job names if experiments have similar names
    unique_job_name="${exp_name}_${CONFIG_INDEX}"
    
    # Sanitize config path for log filename
    log_suffix=$(echo "${config}" | sed 's/[^a-zA-Z0-9]/_/g' | sed 's/_\+/_/g')
    
    echo ""
    echo "Submitting job ${CONFIG_INDEX}/${#CONFIGS[@]}: ${config}"
    echo "  Experiment: ${exp_name}"
    echo "  Job name: ${unique_job_name}"
    
    # Use set +e to continue on error (we handle errors manually)
    set +e
    JOB_OUTPUT=$(bsub -J "${unique_job_name}" \
        -o "${BASE_DIR}/training/hpc_scripts/logs/train_diff_clip_${log_suffix}.%J.out" \
        -e "${BASE_DIR}/training/hpc_scripts/logs/train_diff_clip_${log_suffix}.%J.err" \
        -n 8 \
        -R "rusage[mem=16000]" \
        -gpu "num=1" \
        -W 24:00 \
        -q gpuv100 \
        bash "${TRAIN_SCRIPT}" "${config}" 2>&1)
    BSUB_EXIT_CODE=$?
    set -e
    
    if [ $BSUB_EXIT_CODE -eq 0 ]; then
        # Extract job ID from bsub output
        JOB_ID=$(echo "${JOB_OUTPUT}" | grep -oP 'Job <\K[0-9]+(?=>)' || echo "")
        if [ -n "${JOB_ID}" ]; then
            echo "  ✓ SUCCESS - Job ID: ${JOB_ID}"
            JOB_IDS+=("${JOB_ID}")
            ((SUBMITTED++))
        else
            if echo "${JOB_OUTPUT}" | grep -qi "submitted"; then
                echo "  ✓ SUBMITTED (could not extract job ID)"
                echo "  Output: ${JOB_OUTPUT}"
                ((SUBMITTED++))
            else
                echo "  ✗ FAILED - bsub output: ${JOB_OUTPUT}"
                ((FAILED++))
            fi
        fi
    else
        echo "  ✗ FAILED (bsub exit code: ${BSUB_EXIT_CODE})"
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

