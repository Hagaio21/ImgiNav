#!/bin/bash
#BSUB -J clean_dataset
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/clean_dataset.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/clean_dataset.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -W 12:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/ImgiNav/data_preparation_v2/clean_dataset.py"
LOG_DIR="${BASE_DIR}/ImgiNav/data_preparation_v2/hpc_scripts/logs"

# Dataset paths
DATASET_ROOT="${BASE_DIR}/dataset_v2"

# Manifest paths (can be overridden via environment)
MANIFEST="${MANIFEST:-${DATASET_ROOT}/manifests/manifest_seg.csv}"
OUTPUT="${OUTPUT:-${DATASET_ROOT}/manifests/manifest_seg_cleaned.csv}"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

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
echo "=========================================="
echo "Clean Dataset - Check Layout Quality"
echo "=========================================="
echo "Dataset Root: ${DATASET_ROOT}"
echo "Manifest: ${MANIFEST}"
echo "Output: ${OUTPUT}"
echo "Start: $(date)"
echo "=========================================="

# Validate paths
if [ ! -d "${DATASET_ROOT}" ]; then
    echo "ERROR: Dataset root not found: ${DATASET_ROOT}" >&2
    exit 1
fi

if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: Manifest not found: ${MANIFEST}" >&2
    exit 1
fi

# Validate Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
    exit 1
fi

# =============================================================================
# RUN
# =============================================================================
echo ""
echo "Starting layout quality check..."
echo "=========================================="

cd "${BASE_DIR}"

python "${PYTHON_SCRIPT}" \
    --manifest "${MANIFEST}" \
    --dataset-root "${DATASET_ROOT}" \
    --output "${OUTPUT}"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Dataset cleaning completed successfully"
    if [ -f "${OUTPUT}" ]; then
        line_count=$(wc -l < "${OUTPUT}")
        echo "  Output manifest: ${OUTPUT} (${line_count} lines)"
        
        # Count rejected rows if possible
        if command -v python3 &> /dev/null; then
            rejected_count=$(python3 -c "
import pandas as pd
df = pd.read_csv('${OUTPUT}', low_memory=False)
if 'rejected' in df.columns:
    print(df['rejected'].sum())
else:
    print(0)
" 2>/dev/null || echo "0")
            total_count=$(python3 -c "
import pandas as pd
df = pd.read_csv('${OUTPUT}', low_memory=False)
print(len(df))
" 2>/dev/null || echo "0")
            if [ "${rejected_count}" != "0" ] && [ "${total_count}" != "0" ]; then
                percent=$(python3 -c "print(f'{100*${rejected_count}/${total_count}:.1f}')" 2>/dev/null || echo "0.0")
                echo "  Rejected rows: ${rejected_count} / ${total_count} (${percent}%)"
            fi
        fi
    fi
else
    echo "✗ Dataset cleaning failed with exit code ${EXIT_CODE}"
fi
echo "End: $(date)"
echo "=========================================="

exit ${EXIT_CODE}
