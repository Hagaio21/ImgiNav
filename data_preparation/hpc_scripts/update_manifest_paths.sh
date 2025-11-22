#!/bin/bash
#BSUB -J update_manifest_paths
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/update_manifest_paths.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/update_manifest_paths.%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=2000]"
#BSUB -W 00:30
#BSUB -q hpc

export MKL_INTERFACE_LAYER=LP64
set -euo pipefail

# ----------------------------------------------------------------------
# Create Log Directory
# ----------------------------------------------------------------------
LOG_DIR="/work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs"
mkdir -p "${LOG_DIR}"

echo "[INFO] LSF Job $LSB_JOBID started on $(hostname)."

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
SCRIPT_PATH="${BASE_DIR}/data_preparation/update_manifest_paths.py"

# Manifests to update
MANIFEST_SEG="/work3/s233249/ImgiNav/datasets/controlnet/manifest_seg.csv"
MANIFEST_TEX="/work3/s233249/ImgiNav/datasets/controlnet/manifest_tex.csv"

# Paths
OLD_PATH="/work3/s233249/ImgiNav/datasets/controlnet/layouts"
NEW_PATH="/work3/s233249/ImgiNav/datasets/controlnet/layouts_recolored"

# ----------------------------------------------------------------------
# Check files
# ----------------------------------------------------------------------
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "[ERROR] Script not found: ${SCRIPT_PATH}"
    exit 1
fi

# ----------------------------------------------------------------------
# Conda environment
# ----------------------------------------------------------------------
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || {
        echo "[ERROR] Failed to activate conda env 'imginav'" >&2
        exit 1
    }
fi

cd "${BASE_DIR}"

# ----------------------------------------------------------------------
# Update manifests
# ----------------------------------------------------------------------
echo "=============================================================="
echo " Updating Manifest Paths"
echo "=============================================================="
echo " Old path: ${OLD_PATH}"
echo " New path: ${NEW_PATH}"
echo "=============================================================="

# Update manifest_seg.csv
if [ -f "${MANIFEST_SEG}" ]; then
    echo ""
    echo "[INFO] Updating ${MANIFEST_SEG}..."
    python "${SCRIPT_PATH}" \
        --manifest "${MANIFEST_SEG}" \
        --old-path "${OLD_PATH}" \
        --new-path "${NEW_PATH}" \
        --backup
else
    echo "[WARN] Manifest not found: ${MANIFEST_SEG}"
fi

# Update manifest_tex.csv
if [ -f "${MANIFEST_TEX}" ]; then
    echo ""
    echo "[INFO] Updating ${MANIFEST_TEX}..."
    python "${SCRIPT_PATH}" \
        --manifest "${MANIFEST_TEX}" \
        --old-path "${OLD_PATH}" \
        --new-path "${NEW_PATH}" \
        --backup
else
    echo "[WARN] Manifest not found: ${MANIFEST_TEX}"
fi

echo ""
echo "=============================================================="
echo "✓ Manifest path update complete"
echo "=============================================================="

