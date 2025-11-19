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
#
# Usage:
#     bsub < data_preparation/hpc_scripts/launch_controlnet_pipeline.sh

# Submit the first job in the pipeline
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
echo "=============================================================="
echo ""
echo "[INFO] Submitting first job: Create joint manifest..."
bsub < "${FIRST_JOB}"

echo "[INFO] Pipeline launched! Jobs will chain automatically."
echo "       Monitor progress with: bjobs"

