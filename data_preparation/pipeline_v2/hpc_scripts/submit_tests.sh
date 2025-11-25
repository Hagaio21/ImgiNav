#!/bin/bash
# Submission script for geometry export and layout rendering tests

PROJECT_ROOT="/work3/s233249/ImgiNav"
SCRIPTS_DIR="${PROJECT_ROOT}/ImgiNav/data_preparation/pipeline_v2/hpc_scripts"

echo "=========================================="
echo "Submitting Geometry Export Test"
echo "=========================================="
cd "${SCRIPTS_DIR}"
bsub < run_export_geometry_test.sh

echo ""
echo "Waiting 5 seconds before submitting render test..."
sleep 5

echo ""
echo "=========================================="
echo "Submitting Layout Rendering Test"
echo "=========================================="
bsub < run_render_test.sh

echo ""
echo "=========================================="
echo "Both tests submitted!"
echo "=========================================="
echo "Check job status with: bjobs"
echo "Check logs in: ${SCRIPTS_DIR}/logs/"

