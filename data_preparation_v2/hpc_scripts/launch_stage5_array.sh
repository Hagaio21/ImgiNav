#!/bin/bash
# Launcher script for Stage 5 array job
# This submits run_stage5_array.sh as an LSF array job

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_stage5_array.sh"

if [ ! -f "${RUN_SCRIPT}" ]; then
  echo "ERROR: Run script not found: ${RUN_SCRIPT}" >&2
  exit 1
fi

# Ensure logs directory exists
mkdir -p "${SCRIPT_DIR}/logs"

echo "Submitting Stage 5 array job..."
bsub < "${RUN_SCRIPT}"

echo "Stage 5 array job submitted. Check status with: bjobs"

