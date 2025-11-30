#!/bin/bash
#
# Launch Parallel Dataset Cleaning using Job Array
#
# This script:
# 1. Discovers all scene IDs in the dataset
# 2. Splits them into N shards
# 3. Submits a single job array to process all shards
# 4. Submits a merge job that waits for the array to complete
#
# Usage:
#   ./launch_clean_dataset.sh                     # Default: 10 shards
#   ./launch_clean_dataset.sh --num-shards 20    # 20 parallel jobs
#   ./launch_clean_dataset.sh --dry-run          # Don't submit, just create shards
#   ./launch_clean_dataset.sh --no-merge         # Don't submit merge job

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav"
DATASET_ROOT="${BASE_DIR}/dataset_v2"
SCRIPTS_DIR="${BASE_DIR}/ImgiNav/data_preparation_v2"
HPC_SCRIPTS_DIR="${SCRIPTS_DIR}/hpc_scripts"
LOG_DIR="${HPC_SCRIPTS_DIR}/logs"

# Output directories
SHARDS_DIR="${DATASET_ROOT}/shards"
REJECTIONS_DIR="${DATASET_ROOT}/rejections"

# Defaults
NUM_SHARDS=50
DRY_RUN=0
SUBMIT_MERGE=0

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-shards)
            NUM_SHARDS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --no-merge)
            SUBMIT_MERGE=0
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --num-shards N    Number of parallel shards (default: 10)"
            echo "  --dry-run         Don't submit jobs, just create shards"
            echo "  --no-merge        Don't submit the merge job"
            echo "  --help            Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# SETUP
# =============================================================================
echo "=========================================="
echo "Launching Parallel Dataset Cleaning"
echo "=========================================="
echo "Dataset Root: ${DATASET_ROOT}"
echo "Num Shards: ${NUM_SHARDS}"
echo "Dry Run: ${DRY_RUN}"
echo "Submit Merge: ${SUBMIT_MERGE}"
echo ""

# Create directories
mkdir -p "${SHARDS_DIR}"
mkdir -p "${REJECTIONS_DIR}"
mkdir -p "${LOG_DIR}"

# =============================================================================
# DISCOVER SCENE IDS
# =============================================================================
echo "Discovering scene IDs..."

METADATA_SCENES_DIR="${DATASET_ROOT}/metadata/scenes"

if [ ! -d "${METADATA_SCENES_DIR}" ]; then
    echo "ERROR: Metadata scenes directory not found: ${METADATA_SCENES_DIR}"
    exit 1
fi

SCENE_IDS_FILE="${SHARDS_DIR}/all_scene_ids.txt"
find "${METADATA_SCENES_DIR}" -name "*.json" -printf "%f\n" | sed 's/\.json$//' | sort -u > "${SCENE_IDS_FILE}"

TOTAL_SCENES=$(wc -l < "${SCENE_IDS_FILE}")
echo "  Found ${TOTAL_SCENES} scenes"

if [ "${TOTAL_SCENES}" -eq 0 ]; then
    echo "ERROR: No scenes found"
    exit 1
fi

# =============================================================================
# CREATE SHARDS
# =============================================================================
echo ""
echo "Creating ${NUM_SHARDS} shards..."

SCENES_PER_SHARD=$(( (TOTAL_SCENES + NUM_SHARDS - 1) / NUM_SHARDS ))
echo "  ~${SCENES_PER_SHARD} scenes per shard"

# Remove old shard files
rm -f "${SHARDS_DIR}"/shard_*.txt

# Split into shards (0-indexed for job array)
split -n "l/${NUM_SHARDS}" -d -a 3 "${SCENE_IDS_FILE}" "${SHARDS_DIR}/shard_"

# Rename to .txt and count actual shards created
ACTUAL_SHARDS=0
for f in "${SHARDS_DIR}"/shard_[0-9][0-9][0-9]; do
    if [ -f "$f" ]; then
        mv "$f" "${f}.txt"
        ACTUAL_SHARDS=$((ACTUAL_SHARDS + 1))
    fi
done

# Report shard sizes
echo "  Created ${ACTUAL_SHARDS} shards:"
for f in "${SHARDS_DIR}"/shard_*.txt; do
    if [ -f "$f" ]; then
        SHARD_NAME=$(basename "$f" .txt)
        SHARD_SIZE=$(wc -l < "$f")
        echo "    ${SHARD_NAME}: ${SHARD_SIZE} scenes"
    fi
done

# Calculate array indices (1-indexed for LSF)
ARRAY_END=${ACTUAL_SHARDS}

# =============================================================================
# WRITE CONFIG FILE (for job array to read)
# =============================================================================
CONFIG_FILE="${SHARDS_DIR}/clean_config.sh"
cat > "${CONFIG_FILE}" <<EOF
# Auto-generated config for clean_dataset job array
DATASET_ROOT="${DATASET_ROOT}"
SCRIPTS_DIR="${SCRIPTS_DIR}"
SHARDS_DIR="${SHARDS_DIR}"
REJECTIONS_DIR="${REJECTIONS_DIR}"

# Quality thresholds
MIN_PIXELS="100"
MAX_BLACK_FRACTION="0.95"
MIN_CONTENT_FRACTION="0.05"
MAX_FLOOR_FRACTION="0.85"
MAX_WALL_FRACTION="0.90"
EOF

echo ""
echo "Config written to: ${CONFIG_FILE}"

# =============================================================================
# SUBMIT JOB ARRAY
# =============================================================================
if [ "${DRY_RUN}" = "1" ]; then
    echo ""
    echo "[DRY RUN] Would submit job array: clean_dataset[1-${ARRAY_END}]"
    echo "  Each job processes one shard file"
else
    echo ""
    echo "Submitting job array [1-${ARRAY_END}]..."
    
    ARRAY_OUTPUT=$(bsub -J "clean_dataset[1-${ARRAY_END}]" \
        -o "${LOG_DIR}/clean_dataset.%J.%I.out" \
        -e "${LOG_DIR}/clean_dataset.%J.%I.err" \
        -n 2 \
        -R "rusage[mem=4000]" \
        -W 01:00 \
        -q hpc \
        < "${HPC_SCRIPTS_DIR}/run_clean_array.sh")
    
    ARRAY_JOB_ID=$(echo "${ARRAY_OUTPUT}" | grep -oP '(?<=Job <)\d+(?=>)' || echo "")
    
    if [ -n "${ARRAY_JOB_ID}" ]; then
        echo "  Submitted: clean_dataset[1-${ARRAY_END}] (Job ${ARRAY_JOB_ID})"
    else
        echo "  ERROR: Failed to submit job array"
        echo "  Output: ${ARRAY_OUTPUT}"
        exit 1
    fi
fi

# =============================================================================
# SUBMIT MERGE JOB
# =============================================================================
if [ "${SUBMIT_MERGE}" = "1" ]; then
    if [ "${DRY_RUN}" = "1" ]; then
        echo ""
        echo "[DRY RUN] Would submit merge job after array completes"
    else
        echo ""
        echo "Submitting merge job (depends on array completion)..."
        
        MERGE_OUTPUT=$(bsub -J "merge_rejections" \
            -o "${LOG_DIR}/merge_rejections.%J.out" \
            -e "${LOG_DIR}/merge_rejections.%J.err" \
            -n 1 \
            -R "rusage[mem=2000]" \
            -W 00:30 \
            -q hpc \
            -w "done(${ARRAY_JOB_ID})" \
            < "${HPC_SCRIPTS_DIR}/run_merge_rejections.sh")
        
        MERGE_JOB_ID=$(echo "${MERGE_OUTPUT}" | grep -oP '(?<=Job <)\d+(?=>)' || echo "")
        
        if [ -n "${MERGE_JOB_ID}" ]; then
            echo "  Submitted: merge_rejections (Job ${MERGE_JOB_ID})"
            echo "  Depends on: ${ARRAY_JOB_ID}"
        else
            echo "  WARNING: Failed to submit merge job"
            echo "  Output: ${MERGE_OUTPUT}"
        fi
    fi
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Shards created: ${ACTUAL_SHARDS}"
echo "Shard files: ${SHARDS_DIR}/shard_*.txt"
echo "Output dir: ${REJECTIONS_DIR}/"

if [ "${DRY_RUN}" = "1" ]; then
    echo ""
    echo "This was a DRY RUN. To actually submit:"
    echo "  $0 --num-shards ${NUM_SHARDS}"
else
    echo ""
    echo "Monitor with: bjobs"
    echo ""
    echo "After completion, update manifest with:"
    echo "  python ${SCRIPTS_DIR}/update_manifest_rejections.py \\"
    echo "      --manifest ${DATASET_ROOT}/manifests/manifest_tex.csv \\"
    echo "      --rejections ${REJECTIONS_DIR}/rejections_merged.csv \\"
    echo "      --output ${DATASET_ROOT}/manifests/manifest_tex_filtered.csv"
fi

echo "=========================================="