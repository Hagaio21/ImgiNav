#!/bin/bash
#
# Launch Parallel Dataset Cleaning
#
# Creates shards of scene IDs and submits an LSF job array to check
# layout quality in parallel. Each shard outputs a rejections CSV,
# which are merged after all jobs complete.
#
# Usage:
#   ./launch_clean_dataset.sh                    # Default: 10 shards
#   ./launch_clean_dataset.sh --num-shards 100  # More parallelism
#   ./launch_clean_dataset.sh --dry-run         # Just create shards, don't submit

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav"
DATASET_ROOT="${BASE_DIR}/dataset_v2"
SCRIPTS_DIR="${BASE_DIR}/ImgiNav/data_preparation_v2"
HPC_SCRIPTS_DIR="${SCRIPTS_DIR}/hpc_scripts"
LOG_DIR="${HPC_SCRIPTS_DIR}/logs"

SHARDS_DIR="${DATASET_ROOT}/shards"
OUTPUT_DIR="${DATASET_ROOT}/rejections"

# Defaults
NUM_SHARDS=50
DRY_RUN=0
MANIFEST_PATH=""
CHECK_POV_PALETTE=0
POV_COLOR_TOLERANCE=20
POV_MIN_MATCH_RATIO=0.3
PALETTE_COLOR_TOLERANCE=10

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-shards)
            NUM_SHARDS="$2"
            shift 2
            ;;
        --manifest)
            MANIFEST_PATH="$2"
            CHECK_POV_PALETTE=1
            shift 2
            ;;
        --check-pov-palette)
            CHECK_POV_PALETTE=1
            shift
            ;;
        --pov-color-tolerance)
            POV_COLOR_TOLERANCE="$2"
            shift 2
            ;;
        --pov-min-match-ratio)
            POV_MIN_MATCH_RATIO="$2"
            shift 2
            ;;
        --palette-color-tolerance)
            PALETTE_COLOR_TOLERANCE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --num-shards N              Number of parallel shards (default: 50)"
            echo "  --manifest PATH             Enable POV palette checking with manifest (optional)"
            echo "  --check-pov-palette         Enable POV palette checking (requires --manifest)"
            echo "  --pov-color-tolerance N     Color distance tolerance (default: 20)"
            echo "  --pov-min-match-ratio F     Min match ratio (default: 0.3)"
            echo "  --palette-color-tolerance N Palette quantization tolerance (default: 10)"
            echo "  --dry-run                   Don't submit jobs, just create shards"
            echo "  --help                      Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Launching Parallel Dataset Cleaning"
echo "=========================================="
echo "Dataset Root: ${DATASET_ROOT}"
echo "Num Shards: ${NUM_SHARDS}"
if [ "${CHECK_POV_PALETTE}" = "1" ]; then
    echo "POV Palette Checking: ENABLED"
    if [ -n "${MANIFEST_PATH}" ]; then
        echo "  Manifest: ${MANIFEST_PATH}"
    fi
    echo "  POV Color Tolerance: ${POV_COLOR_TOLERANCE}"
    echo "  POV Min Match Ratio: ${POV_MIN_MATCH_RATIO}"
    echo "  Palette Color Tolerance: ${PALETTE_COLOR_TOLERANCE}"
else
    echo "POV Palette Checking: DISABLED"
fi
echo "Dry Run: ${DRY_RUN}"
echo ""

# =============================================================================
# CREATE DIRECTORIES
# =============================================================================
mkdir -p "${SHARDS_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# =============================================================================
# DISCOVER SCENE IDS FROM METADATA
# =============================================================================
echo "Discovering scene IDs from metadata..."

METADATA_DIR="${DATASET_ROOT}/metadata/scenes"
SCENE_IDS_FILE="${SHARDS_DIR}/all_scene_ids.txt"

if [ ! -d "${METADATA_DIR}" ]; then
    echo "ERROR: Metadata directory not found: ${METADATA_DIR}" >&2
    exit 1
fi

# Extract scene IDs from JSON filenames
find "${METADATA_DIR}" -name "*.json" -printf "%f\n" | sed 's/\.json$//' | sort -u > "${SCENE_IDS_FILE}"

TOTAL_SCENES=$(wc -l < "${SCENE_IDS_FILE}")
echo "  Found ${TOTAL_SCENES} scenes"

if [ "${TOTAL_SCENES}" -eq 0 ]; then
    echo "ERROR: No scene IDs found" >&2
    exit 1
fi

# =============================================================================
# CREATE SHARDS
# =============================================================================
echo ""
echo "Creating ${NUM_SHARDS} shards..."

# Remove old shard files
rm -f "${SHARDS_DIR}"/shard_*.txt

# Split scene IDs into shards
# -n l/N splits into N files by line count (balanced)
# -d uses numeric suffixes
# -a 3 uses 3-digit suffixes (000, 001, ...)
split -n "l/${NUM_SHARDS}" -d -a 3 "${SCENE_IDS_FILE}" "${SHARDS_DIR}/shard_"

# Rename to add .txt extension
ACTUAL_SHARDS=0
for f in "${SHARDS_DIR}"/shard_[0-9][0-9][0-9]; do
    if [ -f "$f" ]; then
        mv "$f" "${f}.txt"
        ACTUAL_SHARDS=$((ACTUAL_SHARDS + 1))
    fi
done

echo "  Created ${ACTUAL_SHARDS} shards"
SCENES_PER_SHARD=$((TOTAL_SCENES / ACTUAL_SHARDS))
echo "  ~${SCENES_PER_SHARD} scenes per shard"

# =============================================================================
# RESOLVE MANIFEST PATH
# =============================================================================
RESOLVED_MANIFEST_PATH=""
if [ -n "${MANIFEST_PATH}" ]; then
    # Try relative to dataset root first, then absolute
    if [ -f "${DATASET_ROOT}/${MANIFEST_PATH}" ]; then
        RESOLVED_MANIFEST_PATH="${DATASET_ROOT}/${MANIFEST_PATH}"
    elif [ -f "${MANIFEST_PATH}" ]; then
        RESOLVED_MANIFEST_PATH="${MANIFEST_PATH}"
    else
        echo "WARNING: Manifest file not found: ${MANIFEST_PATH}" >&2
        echo "  POV palette checking will be disabled" >&2
        CHECK_POV_PALETTE=0
        RESOLVED_MANIFEST_PATH=""
    fi
fi

# =============================================================================
# WRITE CONFIG FILE (for job array to read)
# =============================================================================
CONFIG_FILE="${SHARDS_DIR}/clean_config.sh"
cat > "${CONFIG_FILE}" <<EOF
# Auto-generated config for clean_dataset job array
DATASET_ROOT="${DATASET_ROOT}"
SCRIPTS_DIR="${SCRIPTS_DIR}"
SHARDS_DIR="${SHARDS_DIR}"
OUTPUT_DIR="${OUTPUT_DIR}"

# Quality thresholds (for layouts)
MIN_PIXELS="100"
MAX_BLACK_FRACTION="0.95"
MIN_CONTENT_FRACTION="0.05"

# POV palette checking (optional - set MANIFEST_PATH to enable)
MANIFEST_PATH="${RESOLVED_MANIFEST_PATH}"
CHECK_POV_PALETTE="${CHECK_POV_PALETTE}"
POV_COLOR_TOLERANCE="${POV_COLOR_TOLERANCE}"
POV_MIN_MATCH_RATIO="${POV_MIN_MATCH_RATIO}"
PALETTE_COLOR_TOLERANCE="${PALETTE_COLOR_TOLERANCE}"
EOF

echo ""
echo "Config written to: ${CONFIG_FILE}"

# =============================================================================
# SUBMIT JOBS
# =============================================================================
if [ "${DRY_RUN}" = "1" ]; then
    echo ""
    echo "[DRY RUN] Would submit:"
    echo "  - Job array: clean_dataset[1-${ACTUAL_SHARDS}]"
    echo "  - Merge job: merge_rejections (after array completes)"
    echo ""
    echo "Shard files created in: ${SHARDS_DIR}"
    exit 0
fi

echo ""
echo "Submitting job array [1-${ACTUAL_SHARDS}]..."

# Submit the array job
ARRAY_OUTPUT=$(bsub -J "clean_dataset[1-${ACTUAL_SHARDS}]" \
    -o "${LOG_DIR}/clean_dataset.%J.%I.out" \
    -e "${LOG_DIR}/clean_dataset.%J.%I.err" \
    -n 2 \
    -R "rusage[mem=4000]" \
    -W 01:00 \
    -q hpc \
    < "${HPC_SCRIPTS_DIR}/run_clean_dataset.sh")

# Extract job ID
ARRAY_JOB_ID=$(echo "${ARRAY_OUTPUT}" | grep -oP '(?<=Job <)\d+(?=>)' || echo "")

if [ -z "${ARRAY_JOB_ID}" ]; then
    echo "ERROR: Failed to extract job ID from bsub output" >&2
    echo "Output was: ${ARRAY_OUTPUT}" >&2
    exit 1
fi

echo "  Submitted: Job ${ARRAY_JOB_ID}"
echo "  Logs: ${LOG_DIR}/clean_dataset.${ARRAY_JOB_ID}.*.out"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=========================================="
echo "Job Submitted"
echo "=========================================="
echo "Array job: ${ARRAY_JOB_ID} (${ACTUAL_SHARDS} tasks)"
echo ""
echo "Monitor with:"
echo "  bjobs -A ${ARRAY_JOB_ID}"
echo ""
echo "Output will be in:"
echo "  ${OUTPUT_DIR}/rejections_shard_*.csv"
echo ""
echo "After completion, merge with:"
echo "  head -1 ${OUTPUT_DIR}/rejections_shard_000.csv > ${OUTPUT_DIR}/rejections.csv"
echo "  tail -n +2 -q ${OUTPUT_DIR}/rejections_shard_*.csv >> ${OUTPUT_DIR}/rejections.csv"
echo "=========================================="