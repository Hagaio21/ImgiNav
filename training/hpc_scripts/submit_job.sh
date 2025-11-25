#!/bin/bash
# Master job submission script for HPC
# Generates bsub commands dynamically based on arguments

set -euo pipefail

# Source environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../env_config.sh" 2>/dev/null || {
    # Fallback if env_config.sh not found
    BASE_DIR="${IMGINAV_ROOT:-/work3/s233249/ImgiNav/ImgiNav}"
    export BASE_DIR
}

# Default values
GPU_QUEUE="gpuv100"
NUM_GPUS=1
NUM_CPUS=4
MEMORY=8000
WALLTIME="24:00"
JOB_NAME=""
CONFIG=""
SCRIPT_PATH=""
GENERIC_LAUNCHER="${SCRIPT_DIR}/generic_launcher.sh"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --gpu_queue)
            GPU_QUEUE="$2"
            shift 2
            ;;
        --job_name)
            JOB_NAME="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --num_cpus)
            NUM_CPUS="$2"
            shift 2
            ;;
        --memory)
            MEMORY="$2"
            shift 2
            ;;
        --walltime)
            WALLTIME="$2"
            shift 2
            ;;
        --script_path)
            SCRIPT_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --config <config.yaml> [--gpu_queue <queue>] [--job_name <name>] [--num_gpus <n>] [--script_path <script.py>]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "${CONFIG}" ]; then
    echo "ERROR: --config is required"
    exit 1
fi

if [ -z "${SCRIPT_PATH}" ]; then
    echo "ERROR: --script_path is required"
    exit 1
fi

# Resolve config path
CONFIG_PATH="${BASE_DIR}/${CONFIG}"
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_PATH}"
    exit 1
fi

# Extract experiment name from config if job_name not provided
if [ -z "${JOB_NAME}" ]; then
    JOB_NAME=$(python3 -c "
import yaml
import re
import sys
try:
    with open('${CONFIG_PATH}', 'r') as f:
        config_data = yaml.safe_load(f)
        exp_name = config_data.get('experiment', {}).get('name', 'unnamed')
        exp_name = re.sub(r'[^a-zA-Z0-9_]', '_', exp_name)
        exp_name = re.sub(r'_+', '_', exp_name).strip('_')
        if len(exp_name) > 50:
            exp_name = exp_name[:50]
        print(exp_name)
except Exception as e:
    print('unnamed', file=sys.stderr)
" 2>/dev/null || echo "unnamed")
fi

# Generate log file paths
LOG_SUFFIX=$(echo "${CONFIG}" | sed 's/[^a-zA-Z0-9]/_/g' | sed 's/_\+/_/g')
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"
mkdir -p "${LOG_DIR}"
LOG_OUT="${LOG_DIR}/train_${LOG_SUFFIX}.%J.out"
LOG_ERR="${LOG_DIR}/train_${LOG_SUFFIX}.%J.err"

# Build bsub command
BSUB_CMD="bsub -J \"${JOB_NAME}\""
BSUB_CMD="${BSUB_CMD} -o \"${LOG_OUT}\""
BSUB_CMD="${BSUB_CMD} -e \"${LOG_ERR}\""
BSUB_CMD="${BSUB_CMD} -n ${NUM_CPUS}"
BSUB_CMD="${BSUB_CMD} -R \"rusage[mem=${MEMORY}]\""
BSUB_CMD="${BSUB_CMD} -gpu \"num=${NUM_GPUS}\""
BSUB_CMD="${BSUB_CMD} -W ${WALLTIME}"
BSUB_CMD="${BSUB_CMD} -q ${GPU_QUEUE}"
BSUB_CMD="${BSUB_CMD} bash \"${GENERIC_LAUNCHER}\" python \"${BASE_DIR}/${SCRIPT_PATH}\" --config \"${CONFIG}\""

# Print command and execute
echo "Submitting job: ${JOB_NAME}"
echo "Config: ${CONFIG}"
echo "Queue: ${GPU_QUEUE}"
echo "Command: ${BSUB_CMD}"
echo ""

eval "${BSUB_CMD}"

echo "Job submitted successfully!"

