#!/bin/bash
#
# Launch Clean Dataset v2 - Shard manifest and submit array job
#
# Splits manifest into shards and submits parallel array job to check
# layout quality sample by sample. Each job processes one shard.
#
# Usage:
#   bash launch_clean_dataset_v2.sh --manifest manifest_seg.csv --num-shards 100
#   bash launch_clean_dataset_v2.sh --manifest manifest_seg.csv --num-shards 100 --dataset-root dataset_v2
#

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
SHARDS_DIR="${DATASET_ROOT}/shards_clean_dataset"

# Defaults
NUM_SHARDS=100
DRY_RUN=0
MANIFEST_PATH=""
OUTPUT_PATH=""

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
        --output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --dataset-root)
            DATASET_ROOT="$2"
            shift 2
            ;;
        --num-shards)
            NUM_SHARDS="$2"
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
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --manifest PATH             Input manifest CSV file (required)"
            echo "  --output PATH               Output manifest CSV file (required)"
            echo "  --dataset-root PATH         Dataset root directory (default: ${DATASET_ROOT})"
            echo "  --num-shards N              Number of shards/jobs (default: 100)"
            echo "  --min-pixels N              Minimum pixels for required classes (default: 100)"
            echo "  --max-black-fraction F      Max fraction of black pixels (default: 0.95)"
            echo "  --min-content-fraction F    Min fraction of non-background content (default: 0.05)"
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

# Validate required arguments
if [ -z "${MANIFEST_PATH}" ]; then
    echo "ERROR: --manifest is required" >&2
    exit 1
fi

if [ -z "${OUTPUT_PATH}" ]; then
    echo "ERROR: --output is required" >&2
    exit 1
fi

# Resolve manifest path
if [ -f "${DATASET_ROOT}/${MANIFEST_PATH}" ]; then
    RESOLVED_MANIFEST_PATH="${DATASET_ROOT}/${MANIFEST_PATH}"
elif [ -f "${MANIFEST_PATH}" ]; then
    RESOLVED_MANIFEST_PATH="${MANIFEST_PATH}"
else
    echo "ERROR: Manifest file not found: ${MANIFEST_PATH}" >&2
    exit 1
fi

# Resolve output path
if [[ "${OUTPUT_PATH}" != /* ]]; then
    RESOLVED_OUTPUT_PATH="${DATASET_ROOT}/${OUTPUT_PATH}"
else
    RESOLVED_OUTPUT_PATH="${OUTPUT_PATH}"
fi

echo "=========================================="
echo "Launch Clean Dataset v2"
echo "=========================================="
echo "Manifest: ${RESOLVED_MANIFEST_PATH}"
echo "Output: ${RESOLVED_OUTPUT_PATH}"
echo "Dataset Root: ${DATASET_ROOT}"
echo "Num Shards: ${NUM_SHARDS}"
echo ""
echo "Layout Quality Thresholds:"
echo "  Min Pixels: ${MIN_PIXELS}"
echo "  Max Black Fraction: ${MAX_BLACK_FRACTION}"
echo "  Min Content Fraction: ${MIN_CONTENT_FRACTION}"
echo ""
echo "Dry Run: ${DRY_RUN}"
echo "=========================================="

# =============================================================================
# CREATE DIRECTORIES
# =============================================================================
mkdir -p "${SHARDS_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${RESOLVED_OUTPUT_PATH}")"

# =============================================================================
# CREATE SHARDS
# =============================================================================
echo ""
echo "Creating ${NUM_SHARDS} shards from manifest..."

# Activate conda environment
export MKL_INTERFACE_LAYER=LP64
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    set +u
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || conda activate scenefactor || {
        echo "ERROR: Failed to activate conda environment" >&2
        set -u
        exit 1
    }
    set -u
fi

# Create shards using Python
python <<PYTHON_SCRIPT
import pandas as pd
from pathlib import Path

manifest_path = "${RESOLVED_MANIFEST_PATH}"
shards_dir = Path("${SHARDS_DIR}")
num_shards = ${NUM_SHARDS}

# Load manifest
print(f"Loading manifest: {manifest_path}")
df = pd.read_csv(manifest_path, low_memory=False)
df.columns = df.columns.str.strip()

print(f"  Loaded {len(df)} rows")

# Check for layout_path column
if "layout_path" not in df.columns:
    print("ERROR: Manifest must have 'layout_path' column")
    exit(1)

# Calculate samples per shard
samples_per_shard = (len(df) + num_shards - 1) // num_shards

# Create shards
shards_dir.mkdir(parents=True, exist_ok=True)

# Clean old shards
for old_shard in shards_dir.glob("manifest_shard_*.csv"):
    old_shard.unlink()

# Split into shards
for shard_id in range(num_shards):
    start_idx = shard_id * samples_per_shard
    end_idx = min((shard_id + 1) * samples_per_shard, len(df))
    
    if start_idx >= len(df):
        # Empty shard - create with header only
        shard_df = df.iloc[0:0].copy()
    else:
        shard_df = df.iloc[start_idx:end_idx].copy()
    
    shard_file = shards_dir / f"manifest_shard_{shard_id:04d}.csv"
    shard_df.to_csv(shard_file, index=False)
    print(f"  Created {shard_file} with {len(shard_df)} samples (rows {start_idx} to {end_idx-1})")

# Write metadata
(shards_dir / "num_shards.txt").write_text(str(num_shards))
(shards_dir / "manifest_path.txt").write_text("${RESOLVED_MANIFEST_PATH}")
(shards_dir / "output_path.txt").write_text("${RESOLVED_OUTPUT_PATH}")

print(f"\nCreated {num_shards} shards")
PYTHON_SCRIPT

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create shards" >&2
    exit 1
fi

# Count actual shards created
ACTUAL_SHARDS=$(ls -1 "${SHARDS_DIR}"/manifest_shard_*.csv 2>/dev/null | wc -l)
if [ "${ACTUAL_SHARDS}" -eq 0 ]; then
    echo "ERROR: No shards were created" >&2
    exit 1
fi

echo "  Created ${ACTUAL_SHARDS} shards"

# =============================================================================
# WRITE CONFIG FILE
# =============================================================================
CONFIG_FILE="${SHARDS_DIR}/config.sh"
cat > "${CONFIG_FILE}" <<EOF
# Auto-generated config for clean_dataset job array
DATASET_ROOT="${DATASET_ROOT}"
SCRIPTS_DIR="${SCRIPTS_DIR}"
SHARDS_DIR="${SHARDS_DIR}"
OUTPUT_PATH="${RESOLVED_OUTPUT_PATH}"

# Quality thresholds
MIN_PIXELS="${MIN_PIXELS}"
MAX_BLACK_FRACTION="${MAX_BLACK_FRACTION}"
MIN_CONTENT_FRACTION="${MIN_CONTENT_FRACTION}"
EOF

echo ""
echo "Config written to: ${CONFIG_FILE}"

# =============================================================================
# SUBMIT JOBS
# =============================================================================
if [ "${DRY_RUN}" = "1" ]; then
    echo ""
    echo "[DRY RUN] Would submit:"
    echo "  - Job array: clean_dataset_v2[1-${ACTUAL_SHARDS}]"
    echo "  - Merge job: merge_clean_dataset_v2 (after array completes)"
    echo ""
    echo "Shard files created in: ${SHARDS_DIR}"
    exit 0
fi

echo ""
echo "Submitting job array [1-${ACTUAL_SHARDS}]..."

# Submit the array job
ARRAY_OUTPUT=$(bsub -J "clean_dataset_v2[1-${ACTUAL_SHARDS}]" \
    -o "${LOG_DIR}/clean_dataset_v2.%J.%I.out" \
    -e "${LOG_DIR}/clean_dataset_v2.%J.%I.err" \
    -n 2 \
    -R "rusage[mem=4000]" \
    -W 02:00 \
    -q hpc \
    < "${HPC_SCRIPTS_DIR}/run_clean_dataset_v2.sh")

# Extract job ID
ARRAY_JOB_ID=$(echo "${ARRAY_OUTPUT}" | grep -oP '(?<=Job <)\d+(?=>)' || echo "")

if [ -z "${ARRAY_JOB_ID}" ]; then
    echo "ERROR: Failed to extract job ID from bsub output" >&2
    echo "Output was: ${ARRAY_OUTPUT}" >&2
    exit 1
fi

echo "  Submitted array job: ${ARRAY_JOB_ID}"

# Submit merge job (depends on array job)
echo ""
echo "Submitting merge job (will run after array completes)..."

MERGE_OUTPUT=$(bsub -J "merge_clean_dataset_v2" \
    -w "done(${ARRAY_JOB_ID})" \
    -o "${LOG_DIR}/merge_clean_dataset_v2.%J.out" \
    -e "${LOG_DIR}/merge_clean_dataset_v2.%J.err" \
    -n 1 \
    -R "rusage[mem=8000]" \
    -W 00:30 \
    -q hpc \
    < "${HPC_SCRIPTS_DIR}/run_merge_clean_dataset_v2.sh")

MERGE_JOB_ID=$(echo "${MERGE_OUTPUT}" | grep -oP '(?<=Job <)\d+(?=>)' || echo "")

if [ -z "${MERGE_JOB_ID}" ]; then
    echo "WARNING: Failed to extract merge job ID" >&2
else
    echo "  Submitted merge job: ${MERGE_JOB_ID}"
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=========================================="
echo "Jobs Submitted"
echo "=========================================="
echo "Array job: ${ARRAY_JOB_ID} (${ACTUAL_SHARDS} tasks)"
if [ -n "${MERGE_JOB_ID}" ]; then
    echo "Merge job: ${MERGE_JOB_ID} (depends on array)"
fi
echo ""
echo "Monitor with:"
echo "  bjobs -A ${ARRAY_JOB_ID}"
if [ -n "${MERGE_JOB_ID}" ]; then
    echo "  bjobs ${MERGE_JOB_ID}"
fi
echo ""
echo "Shard outputs will be in:"
echo "  ${SHARDS_DIR}/manifest_shard_*_cleaned.csv"
echo ""
echo "Final merged output will be:"
echo "  ${RESOLVED_OUTPUT_PATH}"
echo "=========================================="

