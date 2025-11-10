#!/bin/bash
# Wrapper script to submit add_layout_columns_to_manifest.sh to HPC queue

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/add_layout_columns_to_manifest.sh"

if [ ! -f "${JOB_SCRIPT}" ]; then
    echo "ERROR: Job script not found: ${JOB_SCRIPT}" >&2
    exit 1
fi

echo "Submitting job: add_layout_columns_to_manifest.sh"
bsub < "${JOB_SCRIPT}"

echo "Job submitted. Check status with: bjobs"
echo "Monitor output with: tail -f data_preparation/hpc_scripts/logs/add_layout_columns.*.out"

