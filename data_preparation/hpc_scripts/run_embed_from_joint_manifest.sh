#!/bin/bash
#BSUB -J embed_joint
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/embed_joint.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/embed_joint.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 12:00
#BSUB -q gpul40s

export MKL_INTERFACE_LAYER=LP64
set -euo pipefail

# ----------------------------------------------------------------------
# Create Log Directory
# ----------------------------------------------------------------------
LOG_DIR="/work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs"
mkdir -p "${LOG_DIR}"

# --- Log that the script has started ---
echo "[INFO] LSF Job $LSB_JOBID started on $(hostname)."
echo "[INFO] Log directory ${LOG_DIR} ensured."

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
echo "[INFO] Setting up paths..."

# --- Base Project Paths ---
PROJECT_ROOT="/work3/s233249/ImgiNav"
SCRIPT_DIR="${PROJECT_ROOT}/ImgiNav/data_preparation"
SCRIPT_PATH="${SCRIPT_DIR}/embed_from_joint_manifest.py"
# Store for use in job chaining
export PROJECT_ROOT

# --- Input/Output Paths ---
JOINT_MANIFEST="/work3/s233249/ImgiNav/datasets/joint_manifest.csv"
TAXONOMY="/work3/s233249/ImgiNav/config/taxonomy.json"
OUTPUT_MANIFEST="/work3/s233249/ImgiNav/datasets/joint_manifest_with_embeddings.csv"

# --- Parameters ---
POV_BATCH_SIZE=32
GRAPH_MODEL="all-MiniLM-L6-v2"
FORMAT="pt"

# --- Check that required files exist ---
echo "[INFO] Checking for required files..."
if [ ! -f "${JOINT_MANIFEST}" ]; then
    echo "[ERROR] Joint manifest not found: ${JOINT_MANIFEST}"
    exit 1
fi
if [ ! -f "${TAXONOMY}" ]; then
    echo "[ERROR] Taxonomy file not found: ${TAXONOMY}"
    exit 1
fi
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "[ERROR] Python script not found: ${SCRIPT_PATH}"
    exit 1
fi
echo "[INFO] All required files found."

# ----------------------------------------------------------------------
# Job Start
# ----------------------------------------------------------------------
echo "=============================================================="
echo " Embedding POVs and Graphs from Joint Manifest"
echo "=============================================================="
echo " Joint Manifest: ${JOINT_MANIFEST}"
echo " Taxonomy:       ${TAXONOMY}"
echo " Output:         ${OUTPUT_MANIFEST}"
echo " POV Batch Size: ${POV_BATCH_SIZE}"
echo " Graph Model:    ${GRAPH_MODEL}"
echo " Format:         ${FORMAT}"
echo "=============================================================="

# ----------------------------------------------------------------------
# Conda environment
# ----------------------------------------------------------------------
echo "[INFO] Activating Conda environment..."
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || {
        echo "[ERROR] Failed to activate conda env 'imginav'" >&2
        exit 1
    }
    echo "[INFO] Conda environment 'imginav' activated."
    echo "[INFO] Python path: $(which python)"
else
    echo "[WARN] miniconda3 not found. Assuming environment is already active."
fi
echo "[CONDA] Environment: ${CONDA_DEFAULT_ENV:-none}"
echo ""

# ----------------------------------------------------------------------
# Run script
# ----------------------------------------------------------------------
echo "[INFO] Starting Python script..."
# Use python -u for unbuffered output
python -u "${SCRIPT_PATH}" \
    --joint-manifest "${JOINT_MANIFEST}" \
    --taxonomy "${TAXONOMY}" \
    --output-manifest "${OUTPUT_MANIFEST}" \
    --pov-batch-size "${POV_BATCH_SIZE}" \
    --graph-model "${GRAPH_MODEL}" \
    --format "${FORMAT}"

# ----------------------------------------------------------------------
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "=============================================================="
    echo "[DONE] Embedding completed successfully"
    echo "=============================================================="
    echo " Output manifest: ${OUTPUT_MANIFEST}"
    echo "=============================================================="
    
    # Submit next job: Create ControlNet manifest
    echo ""
    echo "[INFO] Submitting next job: Create ControlNet manifest..."
    # Use PROJECT_ROOT that's already defined in this script
    NEXT_SCRIPT="${PROJECT_ROOT}/ImgiNav/data_preparation/hpc_scripts/run_create_controlnet_manifest_new_layouts.sh"
    if [ -f "${NEXT_SCRIPT}" ]; then
        bsub < "${NEXT_SCRIPT}"
        echo "[INFO] Next job submitted successfully"
    else
        echo "[WARN] Next job script not found: ${NEXT_SCRIPT}"
    fi
else
    echo "=============================================================="
    echo "[ERROR] Embedding failed with exit code: ${EXIT_CODE}"
    echo "=============================================================="
    exit $EXIT_CODE
fi

