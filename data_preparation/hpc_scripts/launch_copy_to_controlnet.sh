#!/bin/bash
# Master script to launch all three copy jobs for ControlNet dataset creation.
#
# This script submits three independent jobs that can run in parallel:
# 1. Copy graphs (JSON + text conversion)
# 2. Copy layouts
# 3. Copy POVs
#
# Usage (run directly on login node):
#     bash data_preparation/hpc_scripts/launch_copy_to_controlnet.sh
# OR:
#     ./data_preparation/hpc_scripts/launch_copy_to_controlnet.sh

# ----------------------------------------------------------------------
# Create Log Directory
# ----------------------------------------------------------------------
LOG_DIR="/work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs"
mkdir -p "${LOG_DIR}"

# --- Log that the script has started ---
echo "[INFO] Launch script started on $(hostname)."
echo "[INFO] Log directory ${LOG_DIR} ensured."

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
PROJECT_ROOT="/work3/s233249/ImgiNav"
JOB_DIR="${PROJECT_ROOT}/ImgiNav/data_preparation/hpc_scripts"

# Job scripts
JOB_GRAPHS="${JOB_DIR}/run_copy_graphs_to_controlnet.sh"
JOB_LAYOUTS="${JOB_DIR}/run_copy_layouts_to_controlnet.sh"
JOB_POVS="${JOB_DIR}/run_copy_povs_to_controlnet.sh"

# ----------------------------------------------------------------------
# Validate job scripts exist
# ----------------------------------------------------------------------
echo "[INFO] Validating job scripts..."
for job_script in "${JOB_GRAPHS}" "${JOB_LAYOUTS}" "${JOB_POVS}"; do
    if [ ! -f "${job_script}" ]; then
        echo "[ERROR] Job script not found: ${job_script}"
        exit 1
    fi
done
echo "[INFO] All job scripts found."

# ----------------------------------------------------------------------
# Submit jobs
# ----------------------------------------------------------------------
echo "=============================================================="
echo " Launching ControlNet Dataset Copy Jobs"
echo "=============================================================="
echo " Jobs to submit:"
echo "  1. Copy graphs (JSON + text conversion)"
echo "  2. Copy layouts"
echo "  3. Copy POVs"
echo ""
echo " All jobs will run in parallel (independent)"
echo "=============================================================="
echo ""

# Submit graphs job
echo "[INFO] Submitting job: Copy graphs..."
bsub < "${JOB_GRAPHS}" || {
    echo "  ✗ Failed to submit graphs job"
    exit 1
}
echo "  ✓ Graphs job submitted"

# Submit layouts job
echo "[INFO] Submitting job: Copy layouts..."
bsub < "${JOB_LAYOUTS}" || {
    echo "  ✗ Failed to submit layouts job"
    exit 1
}
echo "  ✓ Layouts job submitted"

# Submit POVs job
echo "[INFO] Submitting job: Copy POVs..."
bsub < "${JOB_POVS}" || {
    echo "  ✗ Failed to submit POVs job"
    exit 1
}
echo "  ✓ POVs job submitted"

echo ""
echo "=============================================================="
echo "[DONE] All jobs submitted successfully!"
echo "=============================================================="
echo ""
echo " Monitor jobs with:"
echo "   bjobs"
echo "   bjobs -J copy_graphs_controlnet"
echo "   bjobs -J copy_layouts_controlnet"
echo "   bjobs -J copy_povs_controlnet"
echo ""
echo " Check logs in: ${LOG_DIR}"
echo "=============================================================="

