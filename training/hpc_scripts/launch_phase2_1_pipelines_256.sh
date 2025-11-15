#!/bin/bash
# Launch script for Phase 2.1 pipelines with 256x256 input (unet48, unet64, unet128)
# This script submits all three pipeline jobs to the HPC cluster

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# =============================================================================
# CONFIGURATION
# =============================================================================
PIPELINE_SCRIPTS=(
    "run_phase2_1_pipeline_unet48_256.sh"
    "run_phase2_1_pipeline_unet64_256.sh"
    "run_phase2_1_pipeline_unet94_256.sh"
    "run_phase2_1_pipeline_unet128_256.sh"
)

# =============================================================================
# MAIN
# =============================================================================
echo "=============================================================================="
echo "Launching Phase 2.1 Pipeline Jobs (256x256 input)"
echo "=============================================================================="
echo ""
echo "Script directory: ${SCRIPT_DIR}"
echo "Base directory: ${BASE_DIR}"
echo ""
echo "Jobs to launch:"
for script in "${PIPELINE_SCRIPTS[@]}"; do
    echo "  - ${script}"
done
echo ""
echo "=============================================================================="
echo ""

# Launch each pipeline job
JOB_IDS=()

for script in "${PIPELINE_SCRIPTS[@]}"; do
    script_path="${SCRIPT_DIR}/${script}"
    
    if [ ! -f "${script_path}" ]; then
        echo "ERROR: Script not found: ${script_path}" >&2
        exit 1
    fi
    
    echo "Submitting job: ${script}"
    OUTPUT=$(bsub < "${script_path}" 2>&1)
    
    if [ $? -eq 0 ]; then
        # Extract job ID from bsub output (format: "Job <12345> is submitted to queue <gpul40s>")
        JOB_ID=$(echo "${OUTPUT}" | grep -oP 'Job <\K[0-9]+')
        if [ -n "${JOB_ID}" ]; then
            JOB_IDS+=("${JOB_ID}")
            echo "  ✓ Job submitted successfully (Job ID: ${JOB_ID})"
        else
            echo "  ✓ Job submitted successfully (could not extract job ID)"
        fi
    else
        echo "  ✗ Failed to submit job: ${OUTPUT}" >&2
        exit 1
    fi
    echo ""
done

# Summary
echo "=============================================================================="
echo "Summary"
echo "=============================================================================="
echo "Total jobs submitted: ${#JOB_IDS[@]}"
if [ ${#JOB_IDS[@]} -gt 0 ]; then
    echo "Job IDs: ${JOB_IDS[*]}"
    echo ""
    echo "To check job status, use:"
    echo "  bjobs ${JOB_IDS[0]}"
    echo ""
    echo "To check all your jobs:"
    echo "  bjobs"
    echo ""
    echo "To cancel a job:"
    echo "  bkill <job_id>"
fi
echo "=============================================================================="

