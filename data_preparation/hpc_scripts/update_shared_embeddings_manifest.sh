#!/bin/bash
#BSUB -J update_shared_embeddings_manifest
#BSUB -q normal
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:30
#BSUB -o logs/update_shared_embeddings_manifest_%J.out
#BSUB -e logs/update_shared_embeddings_manifest_%J.err

# Update shared embeddings manifest to use recolored layouts
# This updates the layout_path column to point to layouts_recolored instead of layouts

set -e

# Base directory
BASE_DIR="/work3/s233249/ImgiNav"
cd "${BASE_DIR}"

# Shared embeddings manifest
SHARED_EMBEDDINGS_MANIFEST="/work3/s233249/ImgiNav/experiments/shared_embeddings/manifest_with_embeddings.csv"

# Old and new paths
OLD_PATH="/work3/s233249/ImgiNav/datasets/controlnet/layouts"
NEW_PATH="/work3/s233249/ImgiNav/datasets/controlnet/layouts_recolored"

# Log directory
LOG_DIR="${BASE_DIR}/data_preparation/hpc_scripts/logs"
mkdir -p "${LOG_DIR}"

# Validate manifest exists
if [ ! -f "${SHARED_EMBEDDINGS_MANIFEST}" ]; then
  echo "ERROR: Shared embeddings manifest not found: ${SHARED_EMBEDDINGS_MANIFEST}" >&2
  exit 1
fi

echo "=========================================="
echo "Updating shared embeddings manifest"
echo "=========================================="
echo "Manifest: ${SHARED_EMBEDDINGS_MANIFEST}"
echo "Old path: ${OLD_PATH}"
echo "New path: ${NEW_PATH}"
echo "=========================================="

# Run update script
python3 data_preparation/update_manifest_paths.py \
    --manifest "${SHARED_EMBEDDINGS_MANIFEST}" \
    --old-path "${OLD_PATH}" \
    --new-path "${NEW_PATH}" \
    --backup

echo ""
echo "=========================================="
echo "✓ Shared embeddings manifest updated"
echo "=========================================="

