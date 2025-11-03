#!/bin/bash
# Launch memorization check jobs for all currently training diffusion models
# Uses job array to run all models in parallel

set -euo pipefail

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
ABLATION_DIR="${BASE_DIR}/experiments/diffusion/ablation"
JOB_SCRIPT="${BASE_DIR}/scripts/hpc_scripts/run_memorization_check_array.sh"

# List of config files to test (must match array in run_memorization_check_array.sh)
declare -a CONFIG_FILES=(
    "${ABLATION_DIR}/capacity_unet64_d4.yaml"
    "${ABLATION_DIR}/capacity_unet64_d5.yaml"
    "${ABLATION_DIR}/capacity_unet128_d3.yaml"
    "${ABLATION_DIR}/capacity_unet128_d4.yaml"
    "${ABLATION_DIR}/capacity_unet256_d4.yaml"
    "${ABLATION_DIR}/scheduler_linear.yaml"
)

NUM_MODELS=${#CONFIG_FILES[@]}

echo "========================================="
echo "Launching Memorization Tests (Job Array)"
echo "========================================="
echo "Number of models: ${NUM_MODELS}"
echo ""

# Verify all configs exist
missing=0
for config_file in "${CONFIG_FILES[@]}"; do
    if [ ! -f "${config_file}" ]; then
        echo "⚠ Warning: Config not found - ${config_file}"
        ((missing++))
    fi
done

if [ ${missing} -gt 0 ]; then
    echo ""
    echo "⚠ Warning: ${missing} config file(s) not found"
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "${response}" != "y" ]; then
        exit 1
    fi
fi

echo ""
echo "Submitting job array for ${NUM_MODELS} models..."
echo ""

# Submit job array
job_id=$(bsub < "${JOB_SCRIPT}" | grep -oP '<\K[0-9]+')

if [ -n "${job_id}" ]; then
    echo "✓ Job array submitted: ${job_id}[1-${NUM_MODELS}]"
    echo ""
    echo "Monitor jobs with:"
    echo "  bjobs ${job_id}"
    echo ""
    echo "Check results in:"
    echo "  /work3/s233249/ImgiNav/outputs/memorization_checks/"
    echo ""
    echo "Individual job logs:"
    echo "  /work3/s233249/ImgiNav/ImgiNav/scripts/hpc_scripts/logs/memorization_check.${job_id}.*.out"
else
    echo "✗ Failed to submit job array"
    exit 1
fi
