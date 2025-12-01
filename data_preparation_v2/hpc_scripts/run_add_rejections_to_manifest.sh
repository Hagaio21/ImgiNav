#!/bin/bash
#BSUB -J add_rejections_to_manifest
#BSUB -n 1
#BSUB -R "rusage[mem=4000]"
#BSUB -W 01:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav"
SCRIPTS_DIR="${BASE_DIR}/ImgiNav/data_preparation_v2"
MANIFESTS_DIR="${BASE_DIR}/dataset_v2/manifests"
REJECTIONS_DIR="${BASE_DIR}/dataset_v2/rejections"

# Default paths
REJECTIONS_FILE="${REJECTIONS_DIR}/rejections_merged.csv"
ORIGINAL_MANIFEST="${MANIFESTS_DIR}/manifest_seg.csv"

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --rejections)
            REJECTIONS_FILE="$2"
            shift 2
            ;;
        --original-manifest)
            ORIGINAL_MANIFEST="$2"
            shift 2
            ;;
        --manifest-seg)
            MANIFEST_SEG="$2"
            shift 2
            ;;
        --manifest-tex)
            MANIFEST_TEX="$2"
            shift 2
            ;;
        --manifest)
            # Collect all manifest files
            MANIFEST_FILES=()
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MANIFEST_FILES+=("$1")
                shift
            done
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --output)
            # Collect all output files
            OUTPUT_FILES=()
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                OUTPUT_FILES+=("$1")
                shift
            done
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --rejections PATH           Merged rejections CSV (default: rejections_merged.csv)"
            echo "  --original-manifest PATH    Original manifest with sample_id (default: manifest_seg.csv)"
            echo "  --manifest-seg PATH         Input manifest_seg.csv (legacy)"
            echo "  --manifest-tex PATH         Input manifest_tex.csv (legacy)"
            echo "  --manifest PATH [PATH ...]  Input manifest file(s) to update"
            echo "  --output-dir PATH           Output directory (for legacy mode)"
            echo "  --output PATH [PATH ...]    Output file path(s) for --manifest"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Add Rejections to Manifest"
echo "=========================================="
echo "Rejections: ${REJECTIONS_FILE}"
echo "Original Manifest: ${ORIGINAL_MANIFEST}"
echo "Start: $(date)"
echo ""

# =============================================================================
# VALIDATE INPUTS
# =============================================================================
if [ ! -f "${REJECTIONS_FILE}" ]; then
    echo "ERROR: Rejections file not found: ${REJECTIONS_FILE}" >&2
    exit 1
fi

if [ ! -f "${ORIGINAL_MANIFEST}" ]; then
    echo "ERROR: Original manifest not found: ${ORIGINAL_MANIFEST}" >&2
    exit 1
fi

# =============================================================================
# CONDA ENVIRONMENT
# =============================================================================
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || conda activate scenefactor || {
        echo "ERROR: Failed to activate conda environment" >&2
        exit 1
    }
    echo "Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
    echo ""
fi

# =============================================================================
# RUN
# =============================================================================
cd "${BASE_DIR}"

echo "Running add_rejections_to_manifest.py..."
echo ""

# Build command
CMD_ARGS=(
    --rejections "${REJECTIONS_FILE}"
    --original-manifest "${ORIGINAL_MANIFEST}"
)

# Add manifests and outputs if provided
if [ "${MANIFEST_FILES:-}" != "" ]; then
    if [ "${OUTPUT_FILES:-}" = "" ]; then
        echo "ERROR: --output is required when using --manifest" >&2
        exit 1
    fi
    
    if [ ${#MANIFEST_FILES[@]} -ne ${#OUTPUT_FILES[@]} ]; then
        echo "ERROR: Number of manifests (${#MANIFEST_FILES[@]}) must match number of outputs (${#OUTPUT_FILES[@]})" >&2
        exit 1
    fi
    
    for manifest_file in "${MANIFEST_FILES[@]}"; do
        CMD_ARGS+=(--manifest "${manifest_file}")
    done
    
    for output_file in "${OUTPUT_FILES[@]}"; do
        CMD_ARGS+=(--output "${output_file}")
    done
fi

# Legacy mode: manifest-seg/tex with output-dir
if [ "${MANIFEST_SEG:-}" != "" ] || [ "${MANIFEST_TEX:-}" != "" ]; then
    if [ "${OUTPUT_DIR:-}" = "" ]; then
        echo "ERROR: --output-dir is required when using --manifest-seg/--manifest-tex" >&2
        exit 1
    fi
    
    if [ "${MANIFEST_SEG:-}" != "" ]; then
        CMD_ARGS+=(--manifest-seg "${MANIFEST_SEG}")
    fi
    
    if [ "${MANIFEST_TEX:-}" != "" ]; then
        CMD_ARGS+=(--manifest-tex "${MANIFEST_TEX}")
    fi
    
    CMD_ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

python "${SCRIPTS_DIR}/add_rejections_to_manifest.py" "${CMD_ARGS[@]}"

EXIT_CODE=$?

# =============================================================================
# DONE
# =============================================================================
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Add Rejections COMPLETE"
else
    echo "Add Rejections FAILED: ${EXIT_CODE}"
fi
echo "End: $(date)"
echo "=========================================="

exit $EXIT_CODE

