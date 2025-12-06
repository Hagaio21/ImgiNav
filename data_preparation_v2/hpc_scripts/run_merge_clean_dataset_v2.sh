#!/bin/bash
#BSUB -J merge_clean_dataset_v2
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/merge_clean_dataset_v2.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/merge_clean_dataset_v2.%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=8000]"
#BSUB -W 00:30
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav"
SCRIPTS_DIR="${BASE_DIR}/ImgiNav/data_preparation_v2"
SHARDS_DIR="${BASE_DIR}/dataset_v2/shards_clean_dataset"

# Load config if available
if [ -f "${SHARDS_DIR}/config.sh" ]; then
    source "${SHARDS_DIR}/config.sh"
fi

# Defaults (if config not found)
OUTPUT_PATH="${OUTPUT_PATH:-${BASE_DIR}/dataset_v2/manifests/manifest_seg_cleaned.csv}"

# =============================================================================
# CONDA ENV
# =============================================================================
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    set +u
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || conda activate scenefactor || {
        echo "ERROR: Failed to activate conda environment" >&2
        set -u
        exit 1
    }
    set -u
fi

# =============================================================================
# VALIDATION
# =============================================================================
if [ ! -d "${SHARDS_DIR}" ]; then
    echo "ERROR: Shards directory not found: ${SHARDS_DIR}" >&2
    exit 1
fi

# Find all cleaned shard files
SHARD_FILES=$(ls -1 "${SHARDS_DIR}"/manifest_shard_*_cleaned.csv 2>/dev/null | sort)

if [ -z "${SHARD_FILES}" ]; then
    echo "ERROR: No cleaned shard files found in ${SHARDS_DIR}" >&2
    echo "Looking for pattern: manifest_shard_*_cleaned.csv" >&2
    exit 1
fi

SHARD_COUNT=$(echo "${SHARD_FILES}" | wc -l)
echo "=========================================="
echo "Merge Clean Dataset v2"
echo "=========================================="
echo "Found ${SHARD_COUNT} shard files"
echo "Output: ${OUTPUT_PATH}"
echo "Start: $(date)"
echo "=========================================="

# =============================================================================
# MERGE USING PYTHON SCRIPT
# =============================================================================
cd "${BASE_DIR}"

# Use the merge functionality from clean_dataset.py
python "${SCRIPTS_DIR}/clean_dataset.py" \
    --merge-shards "${SHARDS_DIR}" \
    --output "${OUTPUT_PATH}"

EXIT_CODE=$?

# =============================================================================
# SUMMARY AND CLEANUP
# =============================================================================
echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    if [ -f "${OUTPUT_PATH}" ]; then
        line_count=$(wc -l < "${OUTPUT_PATH}")
        echo "✓ Merge completed successfully"
        echo "  Output: ${OUTPUT_PATH} (${line_count} lines)"
        
        # Count rejected samples if possible
        python3 <<PYTHON_SCRIPT
import pandas as pd
try:
    df = pd.read_csv('${OUTPUT_PATH}', low_memory=False)
    if 'rejected' in df.columns:
        rejected_count = df['rejected'].sum()
        total_count = len(df)
        percent = 100 * rejected_count / total_count if total_count > 0 else 0
        print(f"  Rejected: {rejected_count} / {total_count} ({percent:.1f}%)")
        print(f"  Accepted: {total_count - rejected_count}")
except Exception as e:
    print(f"  Could not read statistics: {e}")
PYTHON_SCRIPT
        
        # Clean up shard files after successful merge
        echo ""
        echo "Cleaning up shard files..."
        SHARD_FILES_TO_REMOVE=$(ls -1 "${SHARDS_DIR}"/manifest_shard_*.csv 2>/dev/null | wc -l)
        if [ "${SHARD_FILES_TO_REMOVE}" -gt 0 ]; then
            rm -f "${SHARDS_DIR}"/manifest_shard_*.csv
            echo "  Removed ${SHARD_FILES_TO_REMOVE} shard files"
            
            # Also remove config and metadata files
            rm -f "${SHARDS_DIR}"/config.sh
            rm -f "${SHARDS_DIR}"/num_shards.txt
            rm -f "${SHARDS_DIR}"/manifest_path.txt
            rm -f "${SHARDS_DIR}"/output_path.txt
            echo "  Removed shard metadata files"
            
            # Try to remove shards directory if empty (may fail if not empty, that's OK)
            rmdir "${SHARDS_DIR}" 2>/dev/null || true
        else
            echo "  No shard files found to remove"
        fi
    else
        echo "✗ Merge completed but output file not found"
        EXIT_CODE=1
    fi
else
    echo "✗ Merge failed with exit code ${EXIT_CODE}"
    echo "  Shard files preserved for debugging"
fi
echo "End: $(date)"
echo "=========================================="

exit ${EXIT_CODE}

