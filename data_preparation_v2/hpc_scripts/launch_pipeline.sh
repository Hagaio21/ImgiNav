#!/bin/bash
# ==============================================================================
# Pipeline Launcher
# 
# Launches the data preparation pipeline using paths.yaml configuration.
# Creates shards dynamically from scene_list.
#
# Usage:
#   ./launch_pipeline.sh --config paths.yaml [options]
#
# Options:
#   --config FILE   Path to paths.yaml (required)
#   --stage N       Start from stage N (default: 2)
#   --num-shards N  Number of shards (default: 100)
#   --dry-run       Show what would be submitted
#
# ==============================================================================

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# ==============================================================================
# Parse arguments
# ==============================================================================
CONFIG_FILE="/work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/paths.yaml"
START_STAGE=2
NUM_SHARDS=100
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
HPC_SCRIPTS_DIR="$(get_config hpc_scripts_dir)"
PYTHON_SCRIPTS_DIR="$(get_config python_scripts_dir)"
LOG_DIR="$(get_config log_dir)"
SCENE_LIST="$(get_config scene_list)"

# Make paths absolute if relative
if [[ "${HPC_SCRIPTS_DIR}" != /* ]]; then
    HPC_SCRIPTS_DIR="${BASE_DIR}/${HPC_SCRIPTS_DIR}"
fi
if [[ "${PYTHON_SCRIPTS_DIR}" != /* ]]; then
    PYTHON_SCRIPTS_DIR="${BASE_DIR}/${PYTHON_SCRIPTS_DIR}"
fi
if [[ "${LOG_DIR}" != /* ]]; then
    LOG_DIR="${BASE_DIR}/${LOG_DIR}"
fi
if [[ -n "${SCENE_LIST}" && "${SCENE_LIST}" != /* ]]; then
    SCENE_LIST="${BASE_DIR}/${SCENE_LIST}"
fi

# Fallback for scene_list
if [ -z "${SCENE_LIST}" ] || [ ! -f "${SCENE_LIST}" ]; then
    SCENE_LIST="${BASE_DIR}/valid_scenes.txt"
fi

echo "=========================================="
echo "Pipeline Launcher"
echo "=========================================="
echo "Config file: ${CONFIG_FILE}"
echo "Output Dataset Root: ${OUTPUT_DATASET_ROOT}"
echo "HPC Scripts Dir: ${HPC_SCRIPTS_DIR}"
echo "Python Scripts Dir: ${PYTHON_SCRIPTS_DIR}"
echo "Scene List: ${SCENE_LIST}"
echo "Log Dir: ${LOG_DIR}"
echo "Start Stage: ${START_STAGE}"
echo "Num Shards: ${NUM_SHARDS}"
echo "=========================================="

# ==============================================================================
# Validate scene list and create shards
# ==============================================================================
if [ ! -f "${SCENE_LIST}" ]; then
    echo "ERROR: Scene list not found: ${SCENE_LIST}"
    echo "Set 'scene_list' in paths.yaml or create valid_scenes.txt in base_dir"
    exit 1
fi

TOTAL_SCENES=$(wc -l < "${SCENE_LIST}")
SCENES_PER_SHARD=$(( (TOTAL_SCENES + NUM_SHARDS - 1) / NUM_SHARDS ))

echo "Total scenes: ${TOTAL_SCENES}"
echo "Scenes per shard: ~${SCENES_PER_SHARD}"
echo ""

# Create log directory
mkdir -p "${LOG_DIR}"

# ==============================================================================
# Create shard files
# ==============================================================================
SHARDS_DIR="$(get_config shards_dir)"
if [[ -z "${SHARDS_DIR}" ]]; then
    SHARDS_DIR="${BASE_DIR}/shards"
fi
if [[ "${SHARDS_DIR}" != /* ]]; then
    SHARDS_DIR="${BASE_DIR}/${SHARDS_DIR}"
fi

echo "Creating ${NUM_SHARDS} shard files in ${SHARDS_DIR}..."
mkdir -p "${SHARDS_DIR}"

# Remove old shards
rm -f "${SHARDS_DIR}"/shard_*.txt

# Create new shards
for ((i=1; i<=NUM_SHARDS; i++)); do
    START_LINE=$(( (i - 1) * SCENES_PER_SHARD + 1 ))
    END_LINE=$(( i * SCENES_PER_SHARD ))
    SHARD_FILE="${SHARDS_DIR}/shard_${i}.txt"
    sed -n "${START_LINE},${END_LINE}p" "${SCENE_LIST}" > "${SHARD_FILE}"
done

# Count non-empty shards
ACTUAL_SHARDS=$(find "${SHARDS_DIR}" -name "shard_*.txt" -size +0 | wc -l)
echo "Created ${ACTUAL_SHARDS} non-empty shard files"

if [ ${ACTUAL_SHARDS} -lt ${NUM_SHARDS} ]; then
    echo "Adjusting NUM_SHARDS from ${NUM_SHARDS} to ${ACTUAL_SHARDS}"
    NUM_SHARDS=${ACTUAL_SHARDS}
fi
echo ""

# ==============================================================================
# Determine stage script and resources
# ==============================================================================
case ${START_STAGE} in
    2)
        STAGE_SCRIPT="${HPC_SCRIPTS_DIR}/run_stage2_array.sh"
        STAGE_NAME="stage2_metadata"
        RESOURCES="-n 1 -R 'rusage[mem=1000]' -W 4:00"
        ;;
    3)
        STAGE_SCRIPT="${HPC_SCRIPTS_DIR}/run_stage3_array.sh"
        STAGE_NAME="stage3_layouts"
        RESOURCES="-n 1 -R 'rusage[mem=1000]' -W 4:00"
        ;;
    4)
        STAGE_SCRIPT="${HPC_SCRIPTS_DIR}/run_stage4_array.sh"
        STAGE_NAME="stage4_povs"
        RESOURCES="-n 1 -R 'rusage[mem=1000]' -W 4:00"
        ;;
    5)
        STAGE_SCRIPT="${HPC_SCRIPTS_DIR}/run_stage5_array.sh"
        STAGE_NAME="stage5_graphs"
        RESOURCES="-n 1 -R 'rusage[mem=1000]' -W 2:00"
        ;;
    6)
        STAGE_SCRIPT="${HPC_SCRIPTS_DIR}/run_stage6_array.sh"
        STAGE_NAME="stage6_manifests"
        RESOURCES="-n 1 -R 'rusage[mem=1000]' -W 1:00"
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
    -env \"CONFIG_FILE=${CONFIG_FILE},SCENE_LIST=${SCENE_LIST},NUM_SHARDS=${NUM_SHARDS}\" \
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