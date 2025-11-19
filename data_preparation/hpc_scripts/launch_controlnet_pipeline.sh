#!/bin/bash
# Master script to launch the complete ControlNet pipeline.
#
# This script submits the first job in the chain, and each job will
# automatically submit the next one upon successful completion.
#
# Pipeline:
# 1. Create joint manifest (layouts + POVs + graphs)
# 2. Embed POVs and graphs
# 3. Create ControlNet training manifest
# 4. Train ControlNet model
#
# Usage (run directly on login node):
#     bash data_preparation/hpc_scripts/launch_controlnet_pipeline.sh
# OR:
#     ./data_preparation/hpc_scripts/launch_controlnet_pipeline.sh

# ----------------------------------------------------------------------
# Create Log Directory
# ----------------------------------------------------------------------
LOG_DIR="/work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs"
mkdir -p "${LOG_DIR}"

# --- Log that the script has started ---
echo "[INFO] Launch script started on $(hostname)."
echo "[INFO] Log directory ${LOG_DIR} ensured."

# ----------------------------------------------------------------------
# Submit the first job in the pipeline
# ----------------------------------------------------------------------
PROJECT_ROOT="/work3/s233249/ImgiNav"
FIRST_JOB="${PROJECT_ROOT}/ImgiNav/data_preparation/hpc_scripts/run_create_joint_manifest.sh"

if [ ! -f "${FIRST_JOB}" ]; then
    echo "[ERROR] First job script not found: ${FIRST_JOB}"
    exit 1
fi

echo "=============================================================="
echo " Launching ControlNet Pipeline"
echo "=============================================================="
echo " Pipeline steps:"
echo "  1. Create joint manifest"
echo "  2. Embed POVs and graphs"
echo "  3. Create ControlNet training manifest"
echo "  4. Train ControlNet model"
echo "=============================================================="
echo ""
echo "[INFO] Submitting first job: Create joint manifest..."
bsub < "${FIRST_JOB}"

echo "[INFO] Pipeline launched! Jobs will chain automatically."
echo "       Monitor progress with: bjobs"

