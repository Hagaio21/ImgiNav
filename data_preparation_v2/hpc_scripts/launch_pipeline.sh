#!/bin/bash
# ==============================================================================
# Pipeline Launcher
# 
# Launches the data preparation pipeline using paths.yaml configuration.
#
# Usage:
#   ./launch_pipeline.sh --config paths.yaml [options]
#
# Options:
#   --config FILE   Path to paths.yaml (required)
#   --stage N       Start from stage N (default: 2)
#   --num-shards N  Number of shards (default: auto-detect)
#   --dry-run       Show what would be submitted
#
# ==============================================================================

set -euo pipefail

# ==============================================================================
# Parse arguments
# ==============================================================================
CONFIG_FILE="/work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/paths.yaml"
START_STAGE=2
NUM_SHARDS="10"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --stage)
            START_STAGE="$2"
            shift 2
            ;;
        --num-shards)
            NUM_SHARDS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --config paths.yaml [--stage N] [--num-shards N] [--dry-run]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "${CONFIG_FILE}" ]; then
    echo "ERROR: --config is required"
    echo "Usage: $0 --config paths.yaml [--stage N] [--num-shards N] [--dry-run]"
    exit 1
fi

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Make config path absolute
CONFIG_FILE="$(cd "$(dirname "${CONFIG_FILE}")" && pwd)/$(basename "${CONFIG_FILE}")"

# ==============================================================================
# Load configuration
# ==============================================================================
get_config() {
    python3 -c "
import yaml
with open('${CONFIG_FILE}') as f:
    config = yaml.safe_load(f)
print(config.get('$1', '') or '')
"
}

BASE_DIR="$(get_config base_dir)"
OUTPUT_DATASET_ROOT="$(get_config output_dataset_root)"
SHARDS_DIR="$(get_config shards_dir)"
LOG_DIR="$(get_config log_dir)"

# Make paths absolute if relative
if [[ "${SHARDS_DIR}" != /* ]]; then
    SHARDS_DIR="${BASE_DIR}/${SHARDS_DIR}"
fi
if [[ "${LOG_DIR}" != /* ]]; then
    LOG_DIR="${BASE_DIR}/${LOG_DIR}"
fi

SCRIPTS_DIR="$(dirname "${CONFIG_FILE}")"

echo "=========================================="
echo "Pipeline Launcher"
echo "=========================================="
echo "Config file: ${CONFIG_FILE}"
echo "Output Dataset Root: ${OUTPUT_DATASET_ROOT}"
echo "Shards Dir: ${SHARDS_DIR}"
echo "Log Dir: ${LOG_DIR}"
echo "Start Stage: ${START_STAGE}"
echo "=========================================="

# ==============================================================================
# Validate and count shards
# ==============================================================================
if [ ! -d "${SHARDS_DIR}" ]; then
    echo "ERROR: Shards directory not found: ${SHARDS_DIR}"
    echo "Create shards first with:"
    echo "  python create_shards.py --scenes-dir <3D-FRONT-DIR> --output-dir ${SHARDS_DIR} --num-shards 10"
    exit 1
fi

SHARD_COUNT=$(ls -1 "${SHARDS_DIR}"/shard_*.txt 2>/dev/null | wc -l)
if [ ${SHARD_COUNT} -eq 0 ]; then
    echo "ERROR: No shard files found in ${SHARDS_DIR}"
    exit 1
fi

if [ -n "${NUM_SHARDS}" ]; then
    if [ ${NUM_SHARDS} -gt ${SHARD_COUNT} ]; then
        echo "WARNING: Requested ${NUM_SHARDS} shards but only ${SHARD_COUNT} found"
        NUM_SHARDS=${SHARD_COUNT}
    fi
else
    NUM_SHARDS=${SHARD_COUNT}
fi

echo "Found ${SHARD_COUNT} shards, using ${NUM_SHARDS}"

# Create log directory
mkdir -p "${LOG_DIR}"

# ==============================================================================
# Determine stage script and resources
# ==============================================================================
case ${START_STAGE} in
    2)
        STAGE_SCRIPT="${SCRIPTS_DIR}/run_stage2_array.sh"
        STAGE_NAME="stage2_metadata"
        RESOURCES="-n 4 -R 'rusage[mem=8000]' -W 4:00"
        ;;
    3)
        STAGE_SCRIPT="${SCRIPTS_DIR}/run_stage3_array.sh"
        STAGE_NAME="stage3_layouts"
        RESOURCES="-n 4 -R 'rusage[mem=8000]' -W 4:00"
        ;;
    4)
        STAGE_SCRIPT="${SCRIPTS_DIR}/run_stage4_array.sh"
        STAGE_NAME="stage4_povs"
        RESOURCES="-n 4 -R 'rusage[mem=8000]' -W 6:00"
        ;;
    5)
        STAGE_SCRIPT="${SCRIPTS_DIR}/run_stage5_array.sh"
        STAGE_NAME="stage5_graphs"
        RESOURCES="-n 2 -R 'rusage[mem=4000]' -W 2:00"
        ;;
    6)
        STAGE_SCRIPT="${SCRIPTS_DIR}/run_stage6_array.sh"
        STAGE_NAME="stage6_manifests"
        RESOURCES="-n 1 -R 'rusage[mem=2000]' -W 1:00"
        ;;
    *)
        echo "ERROR: Invalid stage: ${START_STAGE} (valid: 2-6)"
        exit 1
        ;;
esac

if [ ! -f "${STAGE_SCRIPT}" ]; then
    echo "ERROR: Stage script not found: ${STAGE_SCRIPT}"
    exit 1
fi

echo ""
echo "Will launch: ${STAGE_NAME}[1-${NUM_SHARDS}]"
echo "Script: ${STAGE_SCRIPT}"
echo "Resources: ${RESOURCES}"
echo ""

# ==============================================================================
# Submit job array
# ==============================================================================
BSUB_CMD="bsub -J '${STAGE_NAME}[1-${NUM_SHARDS}]' \
    -o '${LOG_DIR}/${STAGE_NAME}.%J.%I.out' \
    -e '${LOG_DIR}/${STAGE_NAME}.%J.%I.err' \
    ${RESOURCES} \
    -q hpc \
    -env \"CONFIG_FILE=${CONFIG_FILE}\" \
    bash '${STAGE_SCRIPT}'"

if [ "${DRY_RUN}" = true ]; then
    echo "DRY RUN - Would execute:"
    echo "${BSUB_CMD}"
else
    echo "Submitting job array..."
    eval ${BSUB_CMD}
    echo ""
    echo "Job array submitted!"
    echo "Monitor with: bjobs -w"
    echo "Logs in: ${LOG_DIR}"
fi
