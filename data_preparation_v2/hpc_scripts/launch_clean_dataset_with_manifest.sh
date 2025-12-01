#!/bin/bash
#
# Launch Parallel Dataset Cleaning with Manifest
#
# This script:
# 1. Adds sample_id column to manifest (if not already present)
# 2. Shards the manifest by scene_id
# 3. Submits LSF job array to check samples in parallel
#
# Usage:
#   ./launch_clean_dataset_with_manifest.sh --manifest manifests/manifest_tex.csv
#   ./launch_clean_dataset_with_manifest.sh --manifest manifests/manifest_tex.csv --num-shards 100
#   ./launch_clean_dataset_with_manifest.sh --manifest manifests/manifest_tex.csv --no-pov-check  # Disable POV checking

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
MANIFESTS_DIR="${DATASET_ROOT}/manifests"

# Defaults
NUM_SHARDS=100
DRY_RUN=0
MANIFEST_PATH=""
CHECK_POV_PALETTE=1  # POV checking is default/basic usage
POV_COLOR_TOLERANCE=20
POV_MIN_MATCH_RATIO=0.3
PALETTE_COLOR_TOLERANCE=10
SKIP_ADD_SAMPLE_ID=0

# Layout quality thresholds
MIN_PIXELS=100
MAX_BLACK_FRACTION=0.95
MIN_CONTENT_FRACTION=0.05

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --manifest)
            MANIFEST_PATH="$2"
            shift 2
            ;;
        --num-shards)
            NUM_SHARDS="$2"
            shift 2
            ;;
        --check-pov-palette)
            CHECK_POV_PALETTE=1
            shift
            ;;
        --no-pov-check)
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
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --manifest PATH             Manifest CSV file (required)"
            echo "  --num-shards N              Number of parallel shards (default: 100)"
            echo ""
            echo "Layout Quality Thresholds:"
            echo "  --min-pixels N              Minimum pixels for required classes (default: 100)"
            echo "  --max-black-fraction F     Max fraction of black pixels (default: 0.95)"
            echo "  --min-content-fraction F   Min fraction of non-background content (default: 0.05)"
            echo ""
            echo "POV Palette Checking:"
            echo "  --no-pov-check              Disable POV palette checking (enabled by default)"
            echo "  --pov-color-tolerance N     Color distance tolerance (default: 20)"
            echo "  --pov-min-match-ratio F     Min match ratio (default: 0.3)"
            echo "  --palette-color-tolerance N Palette quantization tolerance (default: 10)"
            echo ""
            echo "Other:"
            echo "  --skip-add-sample-id        Skip adding sample_id (assumes already present)"
            echo "  --dry-run                   Don't submit jobs, just prepare"
            echo "  --help                      Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# VALIDATE MANIFEST
# =============================================================================
if [ -z "${MANIFEST_PATH}" ]; then
    echo "ERROR: --manifest is required" >&2
    exit 1
fi

# Resolve manifest path
RESOLVED_MANIFEST_PATH=""
if [ -f "${DATASET_ROOT}/${MANIFEST_PATH}" ]; then
    RESOLVED_MANIFEST_PATH="${DATASET_ROOT}/${MANIFEST_PATH}"
elif [ -f "${MANIFEST_PATH}" ]; then
    RESOLVED_MANIFEST_PATH="${MANIFEST_PATH}"
else
    echo "ERROR: Manifest file not found: ${MANIFEST_PATH}" >&2
    exit 1
fi

echo "=========================================="
echo "Launching Parallel Dataset Cleaning"
echo "=========================================="
echo ""
echo "Workflow:"
echo "  1. Add sample_id to manifest"
echo "  2. Create shards from manifest (by scene_id)"
echo "  3. Submit job array to clean samples in parallel"
echo ""
echo "Configuration:"
echo "  Manifest: ${RESOLVED_MANIFEST_PATH}"
echo "  Num Shards: ${NUM_SHARDS}"
echo ""
echo "Layout Quality Thresholds:"
echo "  Min Pixels: ${MIN_PIXELS}"
echo "  Max Black Fraction: ${MAX_BLACK_FRACTION}"
echo "  Min Content Fraction: ${MIN_CONTENT_FRACTION}"
echo ""
if [ "${CHECK_POV_PALETTE}" = "1" ]; then
    echo "POV Palette Checking: ENABLED"
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
mkdir -p "${MANIFESTS_DIR}"

# =============================================================================
# STEP 1: Add sample_id to manifest (if needed)
# =============================================================================
MANIFEST_WITH_IDS="${RESOLVED_MANIFEST_PATH}"

if [ "${SKIP_ADD_SAMPLE_ID}" = "0" ]; then
    echo "=========================================="
    echo "Step 1: Adding sample_id to manifest"
    echo "=========================================="
    
    # Check if sample_id already exists
    if head -1 "${RESOLVED_MANIFEST_PATH}" | grep -q "sample_id"; then
        echo "  sample_id column already exists, skipping"
    else
        # Create output path
        MANIFEST_BASENAME=$(basename "${RESOLVED_MANIFEST_PATH}" .csv)
        MANIFEST_WITH_IDS="${MANIFESTS_DIR}/${MANIFEST_BASENAME}_with_ids.csv"
        
        echo "  Adding sample_id column..."
        echo "  Input: ${RESOLVED_MANIFEST_PATH}"
        echo "  Output: ${MANIFEST_WITH_IDS}"
        
        # Activate conda if available
        if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
            conda activate imginav || conda activate scenefactor || true
        fi
        
        python "${SCRIPTS_DIR}/add_sample_id.py" \
            --manifest "${RESOLVED_MANIFEST_PATH}" \
            --output "${MANIFEST_WITH_IDS}"
        
        if [ ! -f "${MANIFEST_WITH_IDS}" ]; then
            echo "ERROR: Failed to add sample_id to manifest" >&2
            exit 1
        fi
        
        echo "  Done!"
    fi
else
    echo "Skipping sample_id addition (--skip-add-sample-id)"
fi

echo ""

# =============================================================================
# STEP 2: Extract scene IDs from manifest and create shards
# =============================================================================
echo "=========================================="
echo "Step 2: Creating shards from manifest"
echo "=========================================="

# Extract unique scene_ids from manifest (skip header, get scene_id column)
# Assuming scene_id is the first or a named column
SCENE_IDS_FILE="${SHARDS_DIR}/all_scene_ids.txt"

echo "  Extracting scene IDs from manifest..."
python3 <<PYTHON_SCRIPT
import csv
import sys
from pathlib import Path

manifest_path = Path("${MANIFEST_WITH_IDS}")
output_path = Path("${SCENE_IDS_FILE}")

scene_ids = set()

try:
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene_id = row.get("scene_id", "").strip()
            if scene_id:
                scene_ids.add(scene_id)
    
    # Write sorted scene IDs
    with open(output_path, "w", encoding="utf-8") as f:
        for scene_id in sorted(scene_ids):
            f.write(f"{scene_id}\n")
    
    print(f"  Extracted {len(scene_ids)} unique scene IDs")
    sys.exit(0)
except Exception as e:
    print(f"ERROR: Failed to extract scene IDs: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_SCRIPT

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to extract scene IDs from manifest" >&2
    exit 1
fi

TOTAL_SCENES=$(wc -l < "${SCENE_IDS_FILE}")
echo "  Found ${TOTAL_SCENES} unique scenes"

if [ "${TOTAL_SCENES}" -eq 0 ]; then
    echo "ERROR: No scene IDs found in manifest" >&2
    exit 1
fi

# Create shards
echo ""
echo "  Creating ${NUM_SHARDS} shards..."

# Remove old shard files
rm -f "${SHARDS_DIR}"/shard_*.txt

# Split scene IDs into shards
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
echo ""

# =============================================================================
# STEP 3: Write config file
# =============================================================================
echo "=========================================="
echo "Step 3: Writing config file"
echo "=========================================="

CONFIG_FILE="${SHARDS_DIR}/clean_config.sh"
cat > "${CONFIG_FILE}" <<EOF
# Auto-generated config for clean_dataset job array
DATASET_ROOT="${DATASET_ROOT}"
SCRIPTS_DIR="${SCRIPTS_DIR}"
SHARDS_DIR="${SHARDS_DIR}"
OUTPUT_DIR="${OUTPUT_DIR}"

# Manifest with sample_ids
MANIFEST_PATH="${MANIFEST_WITH_IDS}"

# Quality thresholds (for layouts)
MIN_PIXELS="${MIN_PIXELS}"
MAX_BLACK_FRACTION="${MAX_BLACK_FRACTION}"
MIN_CONTENT_FRACTION="${MIN_CONTENT_FRACTION}"

# POV palette checking
CHECK_POV_PALETTE="${CHECK_POV_PALETTE}"
POV_COLOR_TOLERANCE="${POV_COLOR_TOLERANCE}"
POV_MIN_MATCH_RATIO="${POV_MIN_MATCH_RATIO}"
PALETTE_COLOR_TOLERANCE="${PALETTE_COLOR_TOLERANCE}"
EOF

echo "  Config written to: ${CONFIG_FILE}"
echo ""

# =============================================================================
# STEP 4: Submit job array
# =============================================================================
if [ "${DRY_RUN}" = "1" ]; then
    echo "=========================================="
    echo "[DRY RUN] Would submit:"
    echo "  - Job array: clean_dataset[1-${ACTUAL_SHARDS}]"
    echo ""
    echo "Shard files created in: ${SHARDS_DIR}"
    echo "Manifest with IDs: ${MANIFEST_WITH_IDS}"
    exit 0
fi

echo "=========================================="
echo "Step 4: Submitting job array"
echo "=========================================="
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
echo ""

# =============================================================================
# SUMMARY
# =============================================================================
echo "=========================================="
echo "Job Submitted"
echo "=========================================="
echo "Array job: ${ARRAY_JOB_ID} (${ACTUAL_SHARDS} tasks)"
echo "Manifest: ${MANIFEST_WITH_IDS}"
echo ""
echo "Monitor with:"
echo "  bjobs -A ${ARRAY_JOB_ID}"
echo ""
echo "Output will be in:"
echo "  ${OUTPUT_DIR}/rejections_shard_*.csv"
echo ""
echo "After completion:"
echo "  1. Merge rejections:"
echo "     python ${SCRIPTS_DIR}/merge_rejections.py \\"
echo "         --input-dir ${OUTPUT_DIR} \\"
echo "         --output ${OUTPUT_DIR}/rejections_merged.csv"
echo ""
echo "  2. Add rejections to manifest:"
echo "     python ${SCRIPTS_DIR}/add_rejections_to_manifest.py \\"
echo "         --rejections ${OUTPUT_DIR}/rejections_merged.csv \\"
echo "         --manifest-tex ${MANIFEST_WITH_IDS} \\"
echo "         --output-dir ${MANIFESTS_DIR}/"
echo "=========================================="

