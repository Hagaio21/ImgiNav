#!/bin/bash
#BSUB -J recolor_new_layouts
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/recolor_new_layouts.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/recolor_new_layouts.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -W 02:00
#BSUB -q hpc
#
# Job submission script for recolor_new_layouts.py
#
# This script swaps floor and wall colors in taxonomy.json and optionally
# recolors existing layout images.
#
# Usage:
#   bsub < run_recolor_new_layouts.sh
#
# To enable image recoloring, edit the script and set:
#   RECOLOR_IMAGES=true
#   LAYOUTS_DIR="/path/to/layouts"  # OR
#   LAYOUTS_MANIFEST="/path/to/manifest.csv"
#

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
BASE_DIR="${PROJECT_ROOT}/ImgiNav"
SCRIPT_DIR="${BASE_DIR}/data_preparation"
SCRIPT_PATH="${SCRIPT_DIR}/recolor_new_layouts.py"

# --- Input/Output Paths ---
TAXONOMY_FILE="${BASE_DIR}/config/taxonomy.json"

# Optional: Layout directory or manifest for recoloring images
# Set one of these if you want to recolor images:
LAYOUTS_DIR="/work3/s233249/ImgiNav/datasets/controlnet/layouts"
# LAYOUTS_MANIFEST="/work3/s233249/ImgiNav/datasets/layouts_cleaned_with_latents.csv"

# Output directory for recolored images (default: creates "layouts_recolored" next to layouts_dir)
# If not set, will create: /work3/s233249/ImgiNav/datasets/controlnet/layouts_recolored
OUTPUT_DIR="/work3/s233249/ImgiNav/datasets/controlnet/layouts_recolored"

# --- Parameters ---
RECOLOR_IMAGES=true   # Set to true to also recolor existing images
CREATE_BACKUP=true    # Create backup of taxonomy.json

# --- Check that required files exist ---
echo "[INFO] Checking for required files..."
if [ ! -f "${TAXONOMY_FILE}" ]; then
    echo "[ERROR] Taxonomy file not found: ${TAXONOMY_FILE}"
    exit 1
fi

if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "[ERROR] Python script not found: ${SCRIPT_PATH}"
    exit 1
fi

# --- Check optional files if recoloring images ---
if [ "${RECOLOR_IMAGES}" = "true" ]; then
    if [ -n "${LAYOUTS_DIR:-}" ] && [ ! -d "${LAYOUTS_DIR}" ]; then
        echo "[ERROR] Layouts directory not found: ${LAYOUTS_DIR}"
        exit 1
    fi
    
    if [ -n "${LAYOUTS_MANIFEST:-}" ] && [ ! -f "${LAYOUTS_MANIFEST}" ]; then
        echo "[ERROR] Layouts manifest not found: ${LAYOUTS_MANIFEST}"
        exit 1
    fi
    
    if [ -z "${LAYOUTS_DIR:-}" ] && [ -z "${LAYOUTS_MANIFEST:-}" ]; then
        echo "[ERROR] Recoloring images enabled but neither LAYOUTS_DIR nor LAYOUTS_MANIFEST is set"
        exit 1
    fi
fi

# ----------------------------------------------------------------------
# Conda Environment Setup
# ----------------------------------------------------------------------
echo "[INFO] Activating conda environment..."
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || {
        echo "[WARN] Failed to activate conda environment 'imginav', trying 'scenefactor'..." >&2
        conda activate scenefactor || {
            echo "[ERROR] Failed to activate any conda environment" >&2
            exit 1
        }
    }
    echo "[INFO] Conda environment activated: $(conda info --envs | grep '*' | awk '{print $1}')"
else
    echo "[WARN] Conda not found, assuming Python environment is already set up"
fi

# ----------------------------------------------------------------------
# Set Python Path
# ----------------------------------------------------------------------
cd "${BASE_DIR}"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"
echo "[INFO] Working directory: $(pwd)"
echo "[INFO] PYTHONPATH: ${PYTHONPATH}"

# ----------------------------------------------------------------------
# Build Command
# ----------------------------------------------------------------------
echo "[INFO] Building command..."

CMD="python ${SCRIPT_PATH} --taxonomy ${TAXONOMY_FILE}"

# Add backup flag
if [ "${CREATE_BACKUP}" = "true" ]; then
    CMD="${CMD} --backup"
else
    CMD="${CMD} --no-backup"
fi

# Add image recoloring if enabled
if [ "${RECOLOR_IMAGES}" = "true" ]; then
    CMD="${CMD} --recolor-images"
    
    if [ -n "${LAYOUTS_DIR:-}" ]; then
        CMD="${CMD} --layouts-dir ${LAYOUTS_DIR} --recursive"
        if [ -n "${OUTPUT_DIR:-}" ]; then
            CMD="${CMD} --output-dir ${OUTPUT_DIR}"
        fi
    elif [ -n "${LAYOUTS_MANIFEST:-}" ]; then
        CMD="${CMD} --manifest ${LAYOUTS_MANIFEST}"
    fi
fi

# Add overwrite flag (always overwrite when running on HPC)
CMD="${CMD} --overwrite"

# ----------------------------------------------------------------------
# Run Script
# ----------------------------------------------------------------------
echo "[INFO] ========================================="
echo "[INFO] Running recolor_new_layouts.py"
echo "[INFO] ========================================="
echo "[INFO] Command: ${CMD}"
echo "[INFO] ========================================="

${CMD}

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[INFO] ========================================="
    echo "[INFO] ✓ Job completed successfully"
    echo "[INFO] ========================================="
    if [ "${CREATE_BACKUP}" = "true" ]; then
        echo "[INFO] Backup created: ${TAXONOMY_FILE}.backup"
    fi
else
    echo "[ERROR] ========================================="
    echo "[ERROR] ✗ Job failed with exit code ${EXIT_CODE}"
    echo "[ERROR] ========================================="
    exit ${EXIT_CODE}
fi

