#!/bin/bash
# Launch script for CLIP Diffusion Training - Small Down (Rooms and Scenes)
# Launches both regular_rooms and regular_scenes small_down experiments

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
CONFIG_ROOMS="experiments/diffusion/clip/regular_rooms/small_down.yaml"
CONFIG_SCENES="experiments/diffusion/clip/regular_scenes/small_down.yaml"

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
if [ ! -f "${BASE_DIR}/${CONFIG_ROOMS}" ]; then
    echo "ERROR: Config not found: ${CONFIG_ROOMS}" >&2
    exit 1
fi

if [ ! -f "${BASE_DIR}/${CONFIG_SCENES}" ]; then
    echo "ERROR: Config not found: ${CONFIG_SCENES}" >&2
    exit 1
fi

# =============================================================================
# MAIN
# =============================================================================
echo "=============================================================================="
echo "Launching CLIP Diffusion Training - Small Down (Rooms & Scenes)"
echo "=============================================================================="
echo ""
echo "Script directory: ${SCRIPT_DIR}"
echo "Base directory: ${BASE_DIR}"
echo ""
echo "Experiments to launch:"
echo "  1. Regular CLIP - Rooms (small, down attention)"
echo "     Config: ${CONFIG_ROOMS}"
echo "  2. Regular CLIP - Scenes (small, down attention)"
echo "     Config: ${CONFIG_SCENES}"
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

# Submit rooms experiment
echo ""
echo "Submitting: ${CONFIG_ROOMS}"
exp_name_rooms=$(python3 -c "
import yaml
import re
try:
    with open('${BASE_DIR}/${CONFIG_ROOMS}', 'r') as f:
        config_data = yaml.safe_load(f)
        exp_name = config_data.get('experiment', {}).get('name', 'unnamed')
        exp_name = re.sub(r'[^a-zA-Z0-9_]', '_', exp_name)
        exp_name = re.sub(r'_+', '_', exp_name).strip('_')
        if len(exp_name) > 50:
            exp_name = exp_name[:50]
        print(exp_name)
except:
    print('diff_clip_regular_rooms_small_down')
" 2>/dev/null || echo "diff_clip_regular_rooms_small_down")

log_suffix_rooms=$(echo "${CONFIG_ROOMS}" | sed 's/[^a-zA-Z0-9]/_/g' | sed 's/_\+/_/g')

JOB_OUTPUT_ROOMS=$(bsub -J "${exp_name_rooms}" \
    -o "${BASE_DIR}/training/hpc_scripts/logs/train_diff_clip_${log_suffix_rooms}.%J.out" \
    -e "${BASE_DIR}/training/hpc_scripts/logs/train_diff_clip_${log_suffix_rooms}.%J.err" \
    -n 8 \
    -R "rusage[mem=16000]" \
    -gpu "num=1" \
    -W 24:00 \
    -q gpuv100 \
    bash "${TRAIN_SCRIPT}" "${CONFIG_ROOMS}" 2>&1)

BSUB_EXIT_CODE=$?
if [ $BSUB_EXIT_CODE -eq 0 ]; then
    JOB_ID=$(echo "${JOB_OUTPUT_ROOMS}" | grep -oP 'Job <\K[0-9]+(?=>)' || echo "")
    if [ -n "${JOB_ID}" ]; then
        echo "  SUCCESS - Job ID: ${JOB_ID}"
        JOB_IDS+=("${JOB_ID}")
        ((SUBMITTED++))
    else
        if echo "${JOB_OUTPUT_ROOMS}" | grep -qi "submitted"; then
            echo "  SUBMITTED (could not extract job ID)"
            ((SUBMITTED++))
        else
            echo "  FAILED - bsub output: ${JOB_OUTPUT_ROOMS}"
            ((FAILED++))
        fi
    fi
else
    echo "  FAILED (bsub exit code: ${BSUB_EXIT_CODE})"
    ((FAILED++))
fi

# Submit scenes experiment
echo ""
echo "Submitting: ${CONFIG_SCENES}"
exp_name_scenes=$(python3 -c "
import yaml
import re
try:
    with open('${BASE_DIR}/${CONFIG_SCENES}', 'r') as f:
        config_data = yaml.safe_load(f)
        exp_name = config_data.get('experiment', {}).get('name', 'unnamed')
        exp_name = re.sub(r'[^a-zA-Z0-9_]', '_', exp_name)
        exp_name = re.sub(r'_+', '_', exp_name).strip('_')
        if len(exp_name) > 50:
            exp_name = exp_name[:50]
        print(exp_name)
except:
    print('diff_clip_regular_scenes_small_down')
" 2>/dev/null || echo "diff_clip_regular_scenes_small_down")

log_suffix_scenes=$(echo "${CONFIG_SCENES}" | sed 's/[^a-zA-Z0-9]/_/g' | sed 's/_\+/_/g')

JOB_OUTPUT_SCENES=$(bsub -J "${exp_name_scenes}" \
    -o "${BASE_DIR}/training/hpc_scripts/logs/train_diff_clip_${log_suffix_scenes}.%J.out" \
    -e "${BASE_DIR}/training/hpc_scripts/logs/train_diff_clip_${log_suffix_scenes}.%J.err" \
    -n 8 \
    -R "rusage[mem=16000]" \
    -gpu "num=1" \
    -W 24:00 \
    -q gpuv100 \
    bash "${TRAIN_SCRIPT}" "${CONFIG_SCENES}" 2>&1)

BSUB_EXIT_CODE=$?
if [ $BSUB_EXIT_CODE -eq 0 ]; then
    JOB_ID=$(echo "${JOB_OUTPUT_SCENES}" | grep -oP 'Job <\K[0-9]+(?=>)' || echo "")
    if [ -n "${JOB_ID}" ]; then
        echo "  SUCCESS - Job ID: ${JOB_ID}"
        JOB_IDS+=("${JOB_ID}")
        ((SUBMITTED++))
    else
        if echo "${JOB_OUTPUT_SCENES}" | grep -qi "submitted"; then
            echo "  SUBMITTED (could not extract job ID)"
            ((SUBMITTED++))
        else
            echo "  FAILED - bsub output: ${JOB_OUTPUT_SCENES}"
            ((FAILED++))
        fi
    fi
else
    echo "  FAILED (bsub exit code: ${BSUB_EXIT_CODE})"
    ((FAILED++))
fi

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

