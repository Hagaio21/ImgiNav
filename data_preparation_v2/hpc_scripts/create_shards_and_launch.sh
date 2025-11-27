#!/bin/bash
# Launcher script that creates shard files upfront and submits Stage 1
# Stage 1 will automatically chain to subsequent stages

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALID_SCENES_FILE="/work3/s233249/ImgiNav/ImgiNav/valid_scenes.txt"
SHARDS_DIR="${SCRIPT_DIR}/shards"
N_SHARDS=10
STAGE1_SCRIPT="${SCRIPT_DIR}/run_stage1_array.sh"

# =============================================================================
# VALIDATION
# =============================================================================
if [ ! -f "${VALID_SCENES_FILE}" ]; then
  echo "ERROR: valid_scenes.txt not found at: ${VALID_SCENES_FILE}" >&2
  exit 1
fi

if [ ! -f "${STAGE1_SCRIPT}" ]; then
  echo "ERROR: Stage 1 script not found: ${STAGE1_SCRIPT}" >&2
  exit 1
fi

# =============================================================================
# CREATE SHARD FILES
# =============================================================================
echo "=============================================================================="
echo "Creating Shard Files"
echo "=============================================================================="
echo "Valid scenes file: ${VALID_SCENES_FILE}"
echo "Number of shards: ${N_SHARDS}"
echo "Shards directory: ${SHARDS_DIR}"
echo ""

# Create shards directory
mkdir -p "${SHARDS_DIR}"

# Calculate total lines
TOTAL_LINES=$(wc -l < "${VALID_SCENES_FILE}")
LINES_PER_SHARD=$(( (TOTAL_LINES + N_SHARDS - 1) / N_SHARDS ))

echo "Total scenes: ${TOTAL_LINES}"
echo "Scenes per shard: ${LINES_PER_SHARD}"
echo ""

# Create each shard file
echo "Creating shard files..."
for i in $(seq 1 ${N_SHARDS}); do
  START_LINE=$(( (i - 1) * LINES_PER_SHARD + 1 ))
  END_LINE=$(( i * LINES_PER_SHARD ))
  SHARD_FILE="${SHARDS_DIR}/shard_${i}.txt"
  
  sed -n "${START_LINE},${END_LINE}p" "${VALID_SCENES_FILE}" > "${SHARD_FILE}"
  
  SHARD_COUNT=$(wc -l < "${SHARD_FILE}")
  echo "  Shard ${i}: ${SHARD_COUNT} scenes (lines ${START_LINE}-${END_LINE}) -> ${SHARD_FILE}"
  
  if [ ! -s "${SHARD_FILE}" ]; then
    echo "WARNING: Shard ${i} is empty!" >&2
  fi
done

echo ""
echo "All shard files created in ${SHARDS_DIR}"
echo ""

# =============================================================================
# SUBMIT STAGE 1 ARRAY JOB
# =============================================================================
echo "=============================================================================="
echo "Submitting Stage 1 Array Job"
echo "=============================================================================="
echo "This will submit ${N_SHARDS} parallel jobs for Stage 1"
echo "Each job will automatically chain to subsequent stages upon completion"
echo ""

# Ensure logs directory exists
mkdir -p "${SCRIPT_DIR}/logs"

# Submit the array job
echo "Submitting Stage 1 array job..."
JOB_OUTPUT=$(bsub < "${STAGE1_SCRIPT}" 2>&1)
BSUB_EXIT_CODE=$?

if [ $BSUB_EXIT_CODE -eq 0 ]; then
  # Extract job ID from bsub output
  JOB_ID=$(echo "${JOB_OUTPUT}" | grep -oP 'Job <\K[0-9]+(?=>)' || echo "")
  if [ -n "${JOB_ID}" ]; then
    echo "SUCCESS - Stage 1 Array Job ID: ${JOB_ID}"
    echo ""
    echo "Job chain will be:"
    echo "  Stage 1 [${JOB_ID}] → Stage 2 → Stage 3 → Stage 4 → Stage 5"
    echo ""
    echo "Monitor jobs with:"
    echo "  bjobs ${JOB_ID}"
    echo "  bjobs -a ${JOB_ID}  # All array tasks"
    echo ""
    echo "Check logs in: ${SCRIPT_DIR}/logs/"
    echo ""
    echo "Shard files are in: ${SHARDS_DIR}/"
    echo "  (These can be reused if needed)"
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

