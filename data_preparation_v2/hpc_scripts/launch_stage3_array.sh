#!/bin/bash
# Launcher script for Stage 3 Layout Rendering array job
# Submits 10 parallel jobs to process valid_scenes.txt in shards

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/run_stage3_array.sh"
LOG_DIR="${SCRIPT_DIR}/logs"

# =============================================================================
# VALIDATION
# =============================================================================
if [ ! -f "${JOB_SCRIPT}" ]; then
    echo "ERROR: Job script not found: ${JOB_SCRIPT}" >&2
    exit 1
fi

# Make script executable
chmod +x "${JOB_SCRIPT}"

# Verify the script is actually executable
if [ ! -x "${JOB_SCRIPT}" ]; then
    echo "ERROR: Job script is not executable: ${JOB_SCRIPT}" >&2
    exit 1
fi

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# =============================================================================
# USAGE
# =============================================================================
if [ $# -gt 0 ]; then
    echo "Usage: $0"
    echo ""
    echo "This script submits an array job with 10 parallel workers"
    echo "to process valid_scenes.txt through Stage 3 (Layout Rendering)."
    echo ""
    echo "Before running, make sure to update the paths in:"
    echo "  ${JOB_SCRIPT}"
    echo ""
    exit 1
fi

# =============================================================================
# CONFIRMATION
# =============================================================================
echo "=============================================================================="
echo "Launching Stage 3 Layout Rendering Array Job"
echo "=============================================================================="
echo ""
echo "Script directory: ${SCRIPT_DIR}"
echo "Base directory: ${BASE_DIR}"
echo "Job script: ${JOB_SCRIPT}"
echo "Log directory: ${LOG_DIR}"
echo ""
echo "This will submit an array job with 10 parallel workers."
echo "Each worker will process 1/10 of the scenes from valid_scenes.txt"
echo ""

# Prompt for confirmation
read -p "Submit array job? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# =============================================================================
# SUBMIT JOB
# =============================================================================
echo ""
echo "Submitting array job..."
echo "=============================================================================="

JOB_OUTPUT=$(bsub < "${JOB_SCRIPT}" 2>&1)
BSUB_EXIT_CODE=$?

if [ $BSUB_EXIT_CODE -eq 0 ]; then
    # Extract job ID from bsub output
    JOB_ID=$(echo "${JOB_OUTPUT}" | grep -oP 'Job <\K[0-9]+(?=>)' || echo "")
    if [ -n "${JOB_ID}" ]; then
        echo "SUCCESS - Array Job ID: ${JOB_ID}"
        echo ""
        echo "Monitor jobs with: bjobs ${JOB_ID}"
        echo "Check logs in: ${LOG_DIR}/"
        echo ""
        echo "To check status of all array tasks:"
        echo "  bjobs -a ${JOB_ID}"
        echo ""
        echo "To cancel all tasks:"
        echo "  bkill ${JOB_ID}"
    else
        if echo "${JOB_OUTPUT}" | grep -qi "submitted"; then
            echo "SUBMITTED (could not extract job ID)"
            echo "Output: ${JOB_OUTPUT}"
        else
            echo "FAILED - bsub output: ${JOB_OUTPUT}"
            exit 1
        fi
    fi
else
    echo "FAILED (bsub exit code: ${BSUB_EXIT_CODE})"
    echo "bsub output: ${JOB_OUTPUT}"
    exit 1
fi

echo "=============================================================================="

