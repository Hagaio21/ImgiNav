#!/bin/bash
#
# Launch Parallel Dataset Cleaning
#
# This script:
# 1. Discovers all scene IDs in the dataset
# 2. Splits them into N shards
# 3. Submits N parallel jobs to process each shard
# 4. Optionally submits a merge job that waits for all cleaning jobs
#
# Usage:
#   ./launch_clean_dataset.sh                     # Default: 10 shards
#   ./launch_clean_dataset.sh --num-shards 20    # 20 parallel jobs
#   ./launch_clean_dataset.sh --dry-run          # Don't submit, just show what would happen
#   ./launch_clean_dataset.sh --no-merge         # Don't submit merge job
#
# After completion, rejections will be in:
#   dataset_v2/rejections/rejections_merged.csv
#
# Then update your manifest:
#   python update_manifest_rejections.py \
#       --manifest dataset_v2/manifests/manifest_tex.csv \
#       --rejections dataset_v2/rejections/rejections_merged.csv \
#       --output dataset_v2/manifests/manifest_tex_filtered.csv

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
NUM_SHARDS=10
DRY_RUN=0
SUBMIT_MERGE=1
SKIP_POVS=0
SKIP_LAYOUTS=0

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
        --skip-povs)
            SKIP_POVS=1
            shift
            ;;
        --skip-layouts)
            SKIP_LAYOUTS=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --num-shards N    Number of parallel shards (default: 10)"
            echo "  --dry-run         Don't submit jobs, just show what would happen"
            echo "  --no-merge        Don't submit the merge job"
            echo "  --skip-povs       Skip POV quality checks"
            echo "  --skip-layouts    Skip layout quality checks"
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
echo "Skip POVs: ${SKIP_POVS}"
echo "Skip Layouts: ${SKIP_LAYOUTS}"
echo ""

# Create directories
mkdir -p "${SHARDS_DIR}"
mkdir -p "${REJECTIONS_DIR}"
mkdir -p "${LOG_DIR}"

# =============================================================================
# DISCOVER SCENE IDS
# =============================================================================
echo "Discovering scene IDs..."

# Get scene IDs from metadata directory (most reliable source)
METADATA_SCENES_DIR="${DATASET_ROOT}/metadata/scenes"

if [ ! -d "${METADATA_SCENES_DIR}" ]; then
    echo "ERROR: Metadata scenes directory not found: ${METADATA_SCENES_DIR}"
    exit 1
fi

# Extract scene IDs from JSON filenames
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

# Calculate scenes per shard
SCENES_PER_SHARD=$(( (TOTAL_SCENES + NUM_SHARDS - 1) / NUM_SHARDS ))
echo "  ~${SCENES_PER_SHARD} scenes per shard"

# Split into shards
split -n "l/${NUM_SHARDS}" -d -a 3 "${SCENE_IDS_FILE}" "${SHARDS_DIR}/shard_"

# Rename to .txt and report sizes
for i in $(seq -w 0 $((NUM_SHARDS - 1))); do
    SHARD_FILE="${SHARDS_DIR}/shard_${i}"
    if [ -f "${SHARD_FILE}" ]; then
        mv "${SHARD_FILE}" "${SHARD_FILE}.txt"
        SHARD_SIZE=$(wc -l < "${SHARD_FILE}.txt")
        echo "  Shard ${i}: ${SHARD_SIZE} scenes"
    fi
done

# =============================================================================
# SUBMIT CLEANING JOBS
# =============================================================================
echo ""
echo "Submitting cleaning jobs..."

JOB_IDS=()

for i in $(seq -w 0 $((NUM_SHARDS - 1))); do
    SHARD_FILE="${SHARDS_DIR}/shard_${i}.txt"
    OUTPUT_FILE="${REJECTIONS_DIR}/rejections_shard_${i}.csv"
    
    if [ ! -f "${SHARD_FILE}" ]; then
        continue
    fi
    
    # Build extra args
    EXTRA_ARGS=""
    if [ "${SKIP_POVS}" = "1" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --skip-povs"
    fi
    if [ "${SKIP_LAYOUTS}" = "1" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --skip-layouts"
    fi
    
    if [ "${DRY_RUN}" = "1" ]; then
        echo "  [DRY RUN] Would submit: clean_shard_${i}"
        echo "    Shard: ${SHARD_FILE}"
        echo "    Output: ${OUTPUT_FILE}"
    else
        # Submit job
        JOB_OUTPUT=$(bsub -J "clean_shard_${i}" \
            -o "${LOG_DIR}/clean_shard_${i}.%J.out" \
            -e "${LOG_DIR}/clean_shard_${i}.%J.err" \
            -n 2 \
            -R "rusage[mem=4000]" \
            -W 01:00 \
            -q hpc \
            -env "SHARD_FILE=${SHARD_FILE},OUTPUT_FILE=${OUTPUT_FILE},EXTRA_ARGS=${EXTRA_ARGS}" \
            < "${HPC_SCRIPTS_DIR}/run_clean_shard.sh")
        
        # Extract job ID
        JOB_ID=$(echo "${JOB_OUTPUT}" | grep -oP '(?<=Job <)\d+(?=>)')
        JOB_IDS+=("${JOB_ID}")
        echo "  Submitted: clean_shard_${i} (Job ${JOB_ID})"
    fi
done

# =============================================================================
# SUBMIT MERGE JOB
# =============================================================================
if [ "${SUBMIT_MERGE}" = "1" ] && [ "${#JOB_IDS[@]}" -gt 0 ]; then
    echo ""
    echo "Submitting merge job..."
    
    # Build dependency string
    DEPS=$(IFS=':'; echo "done(${JOB_IDS[*]//:/\&\&})")
    # Actually for bsub it's: -w "done(id1) && done(id2)"
    DEP_STRING=""
    for jid in "${JOB_IDS[@]}"; do
        if [ -z "${DEP_STRING}" ]; then
            DEP_STRING="done(${jid})"
        else
            DEP_STRING="${DEP_STRING} && done(${jid})"
        fi
    done
    
    if [ "${DRY_RUN}" = "1" ]; then
        echo "  [DRY RUN] Would submit merge job after: ${JOB_IDS[*]}"
    else
        MERGE_OUTPUT=$(bsub -J "merge_rejections" \
            -o "${LOG_DIR}/merge_rejections.%J.out" \
            -e "${LOG_DIR}/merge_rejections.%J.err" \
            -n 1 \
            -R "rusage[mem=2000]" \
            -W 00:30 \
            -q hpc \
            -w "${DEP_STRING}" \
            -env "REJECTIONS_DIR=${REJECTIONS_DIR}" \
            < "${HPC_SCRIPTS_DIR}/run_merge_rejections.sh")
        
        MERGE_JOB_ID=$(echo "${MERGE_OUTPUT}" | grep -oP '(?<=Job <)\d+(?=>)')
        echo "  Submitted: merge_rejections (Job ${MERGE_JOB_ID})"
        echo "  Depends on: ${JOB_IDS[*]}"
    fi
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Shards created: ${NUM_SHARDS}"
echo "Shard files: ${SHARDS_DIR}/shard_*.txt"
echo "Output dir: ${REJECTIONS_DIR}/"

if [ "${DRY_RUN}" = "1" ]; then
    echo ""
    echo "This was a DRY RUN. To actually submit:"
    echo "  $0 --num-shards ${NUM_SHARDS}"
else
    echo ""
    echo "Jobs submitted: ${#JOB_IDS[@]}"
    echo "Monitor with: bjobs"
    echo ""
    echo "After completion, update manifest with:"
    echo "  python ${SCRIPTS_DIR}/update_manifest_rejections.py \\"
    echo "      --manifest ${DATASET_ROOT}/manifests/manifest_tex.csv \\"
    echo "      --rejections ${REJECTIONS_DIR}/rejections_merged.csv \\"
    echo "      --output ${DATASET_ROOT}/manifests/manifest_tex_filtered.csv"
fi

echo "=========================================="