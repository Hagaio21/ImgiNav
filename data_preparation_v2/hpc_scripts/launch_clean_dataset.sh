#!/bin/bash
#
# Launch Parallel Dataset Cleaning
#
# Creates shards of scene IDs and submits an LSF job array to check
# layout quality in parallel. Each shard outputs a rejections CSV,
# which are merged after all jobs complete.
#
# Usage:
#   bash launch_clean_dataset.sh --manifest /work3/s233249/ImgiNav/dataset_v2/manifests/manifest_seg.csv
#   bash launch_clean_dataset.sh --manifest manifests/manifest_seg.csv --num-shards 100
#   bash launch_clean_dataset.sh --manifest manifests/manifest_seg.csv --no-pov-check  # Disable POV
#   bash launch_clean_dataset.sh  # Without manifest (layout checking only, uses metadata)

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
NUM_SHARDS=100
DRY_RUN=0
MANIFEST_PATH="/work3/s233249/ImgiNav/dataset_v2/manifests/manifest_seg.csv"  # Default manifest
CHECK_POV_PALETTE=1  # POV checking is default when manifest provided
POV_COLOR_TOLERANCE=20
POV_MIN_MATCH_RATIO=0.3
PALETTE_COLOR_TOLERANCE=10
SKIP_ADD_SAMPLE_ID=0
MANIFESTS_DIR="${DATASET_ROOT}/manifests"

# Layout quality thresholds
MIN_PIXELS=100
MAX_BLACK_FRACTION=0.95
MIN_CONTENT_FRACTION=0.05

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
            CHECK_POV_PALETTE=1  # Manifest enables POV checking by default
            shift 2
            ;;
        --no-manifest)
            MANIFEST_PATH=""
            CHECK_POV_PALETTE=0
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
        --min-pixels)
            MIN_PIXELS="$2"
            shift 2
            ;;
        --max-black-fraction)
            MAX_BLACK_FRACTION="$2"
            shift 2
            ;;
        --min-content-fraction)
            MIN_CONTENT_FRACTION="$2"
            shift 2
            ;;
        --skip-add-sample-id)
            SKIP_ADD_SAMPLE_ID=1
            shift
            ;;
        --no-pov-check)
            CHECK_POV_PALETTE=0
            shift
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
            echo ""
            echo "Layout Quality Thresholds:"
            echo "  --min-pixels N              Minimum pixels for required classes (default: 100)"
            echo "  --max-black-fraction F     Max fraction of black pixels (default: 0.95)"
            echo "  --min-content-fraction F   Min fraction of non-background content (default: 0.05)"
            echo ""
            echo "POV Palette Checking:"
            echo "  --manifest PATH             Manifest CSV file with sample_id (required, enables POV checking)"
            echo "  --no-pov-check              Disable POV palette checking"
            echo "  --skip-add-sample-id        Skip sample_id check (assumes already present)"
            echo "  --pov-color-tolerance N     Color distance tolerance (default: 20)"
            echo "  --pov-min-match-ratio F     Min match ratio (default: 0.3)"
            echo "  --palette-color-tolerance N Palette quantization tolerance (default: 10)"
            echo ""
            echo "Other:"
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
echo ""
echo "Layout Quality Thresholds:"
echo "  Min Pixels: ${MIN_PIXELS}"
echo "  Max Black Fraction: ${MAX_BLACK_FRACTION}"
echo "  Min Content Fraction: ${MIN_CONTENT_FRACTION}"
echo ""
if [ "${CHECK_POV_PALETTE}" = "1" ]; then
    echo "POV Palette Checking: ENABLED (default)"
    if [ -n "${MANIFEST_PATH}" ]; then
        echo "  Manifest: ${MANIFEST_PATH}"
    fi
    echo "  POV Color Tolerance: ${POV_COLOR_TOLERANCE}"
    echo "  POV Min Match Ratio: ${POV_MIN_MATCH_RATIO}"
    echo "  Palette Color Tolerance: ${PALETTE_COLOR_TOLERANCE}"
else
    echo "POV Palette Checking: DISABLED"
fi
echo ""
echo "Dry Run: ${DRY_RUN}"
echo ""

# =============================================================================
# CREATE DIRECTORIES
# =============================================================================
mkdir -p "${SHARDS_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# =============================================================================
# RESOLVE MANIFEST PATH AND ADD SAMPLE_ID
# =============================================================================
RESOLVED_MANIFEST_PATH=""
MANIFEST_WITH_IDS=""

if [ -n "${MANIFEST_PATH}" ]; then
    # Resolve manifest path
    if [ -f "${DATASET_ROOT}/${MANIFEST_PATH}" ]; then
        RESOLVED_MANIFEST_PATH="${DATASET_ROOT}/${MANIFEST_PATH}"
    elif [ -f "${MANIFEST_PATH}" ]; then
        RESOLVED_MANIFEST_PATH="${MANIFEST_PATH}"
    else
        echo "ERROR: Manifest file not found: ${MANIFEST_PATH}" >&2
        exit 1
    fi
    
    # Check if sample_id exists, if not warn (user needs to add it via job)
    if [ "${SKIP_ADD_SAMPLE_ID}" = "0" ]; then
        if ! head -1 "${RESOLVED_MANIFEST_PATH}" | grep -q "sample_id"; then
            echo "WARNING: sample_id column not found in manifest" >&2
            echo "  You need to add sample_id first by submitting add_sample_id.py as a job" >&2
            echo "  Or use --skip-add-sample-id if sample_id already exists" >&2
            exit 1
        fi
    fi
    
    MANIFEST_WITH_IDS="${RESOLVED_MANIFEST_PATH}"
fi

# =============================================================================
# DISCOVER SCENE IDS (from manifest if provided, otherwise from metadata)
# =============================================================================
SCENE_IDS_FILE="${SHARDS_DIR}/all_scene_ids.txt"

if [ -n "${MANIFEST_WITH_IDS}" ]; then
    echo "Extracting scene IDs from manifest..."
    
    # Find scene_id column index
    SCENE_ID_COL=$(head -1 "${MANIFEST_WITH_IDS}" | awk -F',' '{for(i=1;i<=NF;i++) if($i=="scene_id") print i}')
    
    if [ -z "${SCENE_ID_COL}" ]; then
        echo "ERROR: scene_id column not found in manifest" >&2
        exit 1
    fi
    
    # Extract scene IDs using awk (skip header, get unique, sort)
    tail -n +2 "${MANIFEST_WITH_IDS}" | \
        awk -F',' -v col="${SCENE_ID_COL}" '{print $col}' | \
        sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | \
        grep -v '^$' | \
        sort -u > "${SCENE_IDS_FILE}"
    
    if [ $? -ne 0 ] || [ ! -s "${SCENE_IDS_FILE}" ]; then
        echo "ERROR: Failed to extract scene IDs from manifest" >&2
        exit 1
    fi
else
    echo "Discovering scene IDs from metadata..."
    
    METADATA_DIR="${DATASET_ROOT}/metadata/scenes"
    
    if [ ! -d "${METADATA_DIR}" ]; then
        echo "ERROR: Metadata directory not found: ${METADATA_DIR}" >&2
        exit 1
    fi
    
    # Extract scene IDs from JSON filenames
    find "${METADATA_DIR}" -name "*.json" -printf "%f\n" | sed 's/\.json$//' | sort -u > "${SCENE_IDS_FILE}"
fi

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
MIN_PIXELS="${MIN_PIXELS}"
MAX_BLACK_FRACTION="${MAX_BLACK_FRACTION}"
MIN_CONTENT_FRACTION="${MIN_CONTENT_FRACTION}"

# POV palette checking (manifest with sample_id)
MANIFEST_PATH="${MANIFEST_WITH_IDS}"
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