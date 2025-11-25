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

# Manifests to update (customize as needed)
SHARED_EMBEDDINGS_MANIFEST="/work3/s233249/ImgiNav/experiments/shared_embeddings/manifest_with_embeddings.csv"

# Paths (customize as needed)
OLD_PATH="${OLD_PATH:-/path/to/old/layouts}"
NEW_PATH="${NEW_PATH:-/path/to/new/layouts}"

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

# Update shared embeddings manifest
if [ -f "${SHARED_EMBEDDINGS_MANIFEST}" ]; then
    echo ""
    echo "[INFO] Updating ${SHARED_EMBEDDINGS_MANIFEST}..."
    python "${SCRIPT_PATH}" \
        --manifest "${SHARED_EMBEDDINGS_MANIFEST}" \
        --old-path "${OLD_PATH}" \
        --new-path "${NEW_PATH}" \
        --backup
else
    echo "[WARN] Manifest not found: ${SHARED_EMBEDDINGS_MANIFEST}"
fi

echo ""
echo "=============================================================="
echo "✓ Manifest path update complete"
echo "=============================================================="

