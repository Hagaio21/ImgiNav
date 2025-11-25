#!/bin/bash
#BSUB -J fix_manifest_csv
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/fix_manifest_csv.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/fix_manifest_csv.%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=4000]"
#BSUB -W 1:00
#BSUB -q gpul40s

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/scripts/fix_manifest_csv.py"

# Shared embeddings manifest
SHARED_EMBEDDINGS_DIR="/work3/s233249/ImgiNav/experiments/shared_embeddings"
INPUT_MANIFEST="${SHARED_EMBEDDINGS_DIR}/manifest_with_embeddings.csv"
OUTPUT_MANIFEST="${SHARED_EMBEDDINGS_DIR}/manifest_with_embeddings_fixed.csv"
BACKUP_MANIFEST="${SHARED_EMBEDDINGS_DIR}/manifest_with_embeddings.csv.backup"

LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# =============================================================================
# MODULES
# =============================================================================
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# CONDA ENV
# =============================================================================
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || {
    echo "Failed to activate conda environment 'imginav'" >&2
    conda activate scenefactor || {
      echo "Failed to activate any conda environment" >&2
      exit 1
    }
  }
fi

# =============================================================================
# VALIDATION
# =============================================================================
if [ ! -f "${INPUT_MANIFEST}" ]; then
  echo "ERROR: Input manifest not found: ${INPUT_MANIFEST}" >&2
  exit 1
fi

if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
  exit 1
fi

# =============================================================================
# RUN
# =============================================================================
echo "=========================================="
echo "Fixing Misaligned CSV Manifest"
echo "=========================================="
echo "Input manifest: ${INPUT_MANIFEST}"
echo "Output manifest: ${OUTPUT_MANIFEST}"
echo "Working directory: ${BASE_DIR}"
echo "Start: $(date)"
echo "=========================================="

cd "${BASE_DIR}"

# Create backup
echo ""
echo "Creating backup..."
cp "${INPUT_MANIFEST}" "${BACKUP_MANIFEST}"
echo "✓ Backup created: ${BACKUP_MANIFEST}"

# Fix CSV
echo ""
echo "Fixing CSV..."
python "${PYTHON_SCRIPT}" \
  --input "${INPUT_MANIFEST}" \
  --output "${OUTPUT_MANIFEST}" \
  --use-default-columns

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "CSV Fix COMPLETE - SUCCESS"
  echo "=========================================="
  echo "Fixed manifest: ${OUTPUT_MANIFEST}"
  echo "Backup saved: ${BACKUP_MANIFEST}"
  echo ""
  echo "To replace the original file, run:"
  echo "  mv ${OUTPUT_MANIFEST} ${INPUT_MANIFEST}"
  echo ""
  echo "End: $(date)"
  echo "=========================================="
  exit 0
else
  echo ""
  echo "=========================================="
  echo "CSV Fix FAILED with exit code: ${EXIT_CODE}"
  echo "=========================================="
  exit $EXIT_CODE
fi

