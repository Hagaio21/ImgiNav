#!/bin/bash
#BSUB -J stage2_metadata[1-10]
#BSUB -o logs/stage2.%J.%I.out
#BSUB -e logs/stage2.%J.%I.err
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -W 4:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Path to paths.yaml - can be overridden with CONFIG_FILE environment variable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/paths.yaml}"

# ==============================================================================
# JOB SETUP
# ==============================================================================
if [ -n "${LSB_JOBINDEX:-}" ]; then
    JOB_ID=${LSB_JOBINDEX}
elif [ $# -ge 1 ]; then
    JOB_ID=$1
else
    echo "ERROR: No job index. Run as array job or provide shard ID as argument."
    exit 1
fi

# ==============================================================================
# LOAD CONFIGURATION
# ==============================================================================
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Parse YAML using Python
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
SHARDS_DIR="$(get_config shards_dir)"
LOG_DIR="$(get_config log_dir)"

# Make paths absolute if relative
if [[ "${HPC_SCRIPTS_DIR}" != /* ]]; then
    HPC_SCRIPTS_DIR="${BASE_DIR}/${HPC_SCRIPTS_DIR}"
fi
if [[ "${PYTHON_SCRIPTS_DIR}" != /* ]]; then
    PYTHON_SCRIPTS_DIR="${BASE_DIR}/${PYTHON_SCRIPTS_DIR}"
fi
if [[ "${SHARDS_DIR}" != /* ]]; then
    SHARDS_DIR="${BASE_DIR}/${SHARDS_DIR}"
fi
if [[ "${LOG_DIR}" != /* ]]; then
    LOG_DIR="${BASE_DIR}/${LOG_DIR}"
fi

SHARD_FILE="${SHARDS_DIR}/shard_${JOB_ID}.txt"

# Ensure directories exist
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Stage 2: Compile Metadata - Shard ${JOB_ID}"
echo "=========================================="
echo "Config file: ${CONFIG_FILE}"
echo "Output Dataset Root: ${OUTPUT_DATASET_ROOT}"
echo "Shard file: ${SHARD_FILE}"
echo "=========================================="

# Verify shard file exists
if [ ! -f "${SHARD_FILE}" ]; then
    echo "ERROR: Shard file not found: ${SHARD_FILE}"
    exit 1
fi

SCENE_COUNT=$(wc -l < "${SHARD_FILE}")
echo "Processing ${SCENE_COUNT} scenes from shard ${JOB_ID}"

if [ ${SCENE_COUNT} -eq 0 ]; then
    echo "No scenes in this shard, skipping"
    exit 0
fi

# ==============================================================================
# CONDA ACTIVATION
# ==============================================================================
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || { echo "Failed to activate conda" >&2; exit 1; }
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate imginav || { echo "Failed to activate conda" >&2; exit 1; }
fi

# ==============================================================================
# RUN STAGE 2
# ==============================================================================
cd "${BASE_DIR}"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"

echo "Running stage 2..."
python "${PYTHON_SCRIPTS_DIR}/stage2_compile_metadata.py" \
    --config "${CONFIG_FILE}" \
    --scene-list "${SHARD_FILE}"

STAGE2_EXIT_CODE=$?

if [ ${STAGE2_EXIT_CODE} -ne 0 ]; then
    echo "ERROR: Stage 2 failed with exit code ${STAGE2_EXIT_CODE}"
    exit ${STAGE2_EXIT_CODE}
fi

echo "Stage 2 completed successfully for shard ${JOB_ID}"

# ==============================================================================
# CHAIN TO STAGE 3
# ==============================================================================
STAGE3_SCRIPT="${HPC_SCRIPTS_DIR}/run_stage3_array.sh"

if [ -f "${STAGE3_SCRIPT}" ]; then
    echo "Submitting stage 3 for shard ${JOB_ID}..."
    
    bsub -J "stage3_shard${JOB_ID}" \
         -o "${LOG_DIR}/stage3_shard${JOB_ID}.%J.out" \
         -e "${LOG_DIR}/stage3_shard${JOB_ID}.%J.err" \
         -n 4 \
         -R "rusage[mem=8000]" \
         -W 4:00 \
         -q hpc \
         -env "CONFIG_FILE=${CONFIG_FILE}" \
         bash "${STAGE3_SCRIPT}" "${JOB_ID}"
    
    echo "Stage 3 job submitted for shard ${JOB_ID}"
else
    echo "WARNING: Stage 3 script not found: ${STAGE3_SCRIPT}"
fi

echo "Stage 2 complete for shard ${JOB_ID}"