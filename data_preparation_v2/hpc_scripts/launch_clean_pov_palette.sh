#!/bin/bash
#
# Launch POV Palette Checking
#
# Checks POV images for palette matching with layouts using a manifest file.
# This is different from layout-only cleaning which uses scene shards.
#
# Usage:
#   ./launch_clean_pov_palette.sh --manifest manifest_seg.csv
#   ./launch_clean_pov_palette.sh --manifest manifest_seg.csv --pov-color-tolerance 30
#   ./launch_clean_pov_palette.sh --manifest manifest_seg.csv --dry-run

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
OUTPUT_DIR="${DATASET_ROOT}/rejections"

# Defaults
MANIFEST_PATH=""
POV_COLOR_TOLERANCE=20
POV_MIN_MATCH_RATIO=0.3
PALETTE_COLOR_TOLERANCE=10
DRY_RUN=0

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --manifest)
            MANIFEST_PATH="$2"
            shift 2
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
            echo "  --manifest PATH              Manifest CSV file (required)"
            echo "  --pov-color-tolerance N      Color distance tolerance (default: 20)"
            echo "  --pov-min-match-ratio F      Min match ratio (default: 0.3)"
            echo "  --palette-color-tolerance N  Palette quantization tolerance (default: 10)"
            echo "  --dry-run                    Don't submit job, just show command"
            echo "  --help                       Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# VALIDATE ARGUMENTS
# =============================================================================
if [ -z "${MANIFEST_PATH}" ]; then
    echo "ERROR: --manifest is required" >&2
    exit 1
fi

# Resolve manifest path (try relative to dataset root, then absolute)
if [ ! -f "${MANIFEST_PATH}" ]; then
    TRY_PATH="${DATASET_ROOT}/${MANIFEST_PATH}"
    if [ -f "${TRY_PATH}" ]; then
        MANIFEST_PATH="${TRY_PATH}"
    else
        echo "ERROR: Manifest file not found: ${MANIFEST_PATH}" >&2
        exit 1
    fi
fi

# Determine output filename from manifest
MANIFEST_BASENAME=$(basename "${MANIFEST_PATH}" .csv)
OUTPUT_FILE="${OUTPUT_DIR}/pov_palette_rejections_${MANIFEST_BASENAME}.csv"

echo "=========================================="
echo "Launching POV Palette Checking"
echo "=========================================="
echo "Dataset Root: ${DATASET_ROOT}"
echo "Manifest: ${MANIFEST_PATH}"
echo "Output: ${OUTPUT_FILE}"
echo "POV Color Tolerance: ${POV_COLOR_TOLERANCE}"
echo "POV Min Match Ratio: ${POV_MIN_MATCH_RATIO}"
echo "Palette Color Tolerance: ${PALETTE_COLOR_TOLERANCE}"
echo "Dry Run: ${DRY_RUN}"
echo ""

# =============================================================================
# CREATE DIRECTORIES
# =============================================================================
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# =============================================================================
# WRITE CONFIG FILE
# =============================================================================
CONFIG_FILE="${OUTPUT_DIR}/pov_palette_config.sh"
cat > "${CONFIG_FILE}" <<EOF
# Auto-generated config for POV palette checking
DATASET_ROOT="${DATASET_ROOT}"
SCRIPTS_DIR="${SCRIPTS_DIR}"
MANIFEST_PATH="${MANIFEST_PATH}"
OUTPUT_FILE="${OUTPUT_FILE}"
POV_COLOR_TOLERANCE="${POV_COLOR_TOLERANCE}"
POV_MIN_MATCH_RATIO="${POV_MIN_MATCH_RATIO}"
PALETTE_COLOR_TOLERANCE="${PALETTE_COLOR_TOLERANCE}"
EOF

echo "Config written to: ${CONFIG_FILE}"
echo ""

# =============================================================================
# SUBMIT JOB
# =============================================================================
if [ "${DRY_RUN}" = "1" ]; then
    echo "[DRY RUN] Would submit:"
    echo "  Job: clean_pov_palette"
    echo "  Command: python clean_dataset.py --dataset-root ... --manifest ... --check-pov-palette ..."
    echo ""
    echo "Config file: ${CONFIG_FILE}"
    exit 0
fi

echo "Submitting job..."
echo ""

# Submit the job
JOB_OUTPUT=$(bsub -J "clean_pov_palette" \
    -o "${LOG_DIR}/clean_pov_palette.%J.out" \
    -e "${LOG_DIR}/clean_pov_palette.%J.err" \
    -n 4 \
    -R "rusage[mem=8000]" \
    -W 02:00 \
    -q hpc \
    < "${HPC_SCRIPTS_DIR}/run_clean_pov_palette.sh")

# Extract job ID
JOB_ID=$(echo "${JOB_OUTPUT}" | grep -oP '(?<=Job <)\d+(?=>)' || echo "")

if [ -z "${JOB_ID}" ]; then
    echo "ERROR: Failed to extract job ID from bsub output" >&2
    echo "Output was: ${JOB_OUTPUT}" >&2
    exit 1
fi

echo "  Submitted: Job ${JOB_ID}"
echo "  Logs: ${LOG_DIR}/clean_pov_palette.${JOB_ID}.out"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=========================================="
echo "Job Submitted"
echo "=========================================="
echo "Job: ${JOB_ID}"
echo ""
echo "Monitor with:"
echo "  bjobs ${JOB_ID}"
echo ""
echo "Output will be in:"
echo "  ${OUTPUT_FILE}"
echo "=========================================="

