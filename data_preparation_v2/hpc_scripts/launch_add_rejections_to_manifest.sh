#!/bin/bash
#
# Launch Add Rejections to Manifest Job
#
# Submits a job to add rejection columns to manifest files.
#
# Usage:
#   bash launch_add_rejections_to_manifest.sh
#   bash launch_add_rejections_to_manifest.sh --manifest manifest_seg.csv manifest_vae_latent.csv --output manifest_seg_with_rejections.csv manifest_vae_latent_with_rejections.csv

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav"
SCRIPTS_DIR="${BASE_DIR}/ImgiNav/data_preparation_v2"
HPC_SCRIPTS_DIR="${SCRIPTS_DIR}/hpc_scripts"
MANIFESTS_DIR="${BASE_DIR}/dataset_v2/manifests"
REJECTIONS_DIR="${BASE_DIR}/dataset_v2/rejections"

# Default paths
REJECTIONS_FILE="${REJECTIONS_DIR}/rejections_merged.csv"
ORIGINAL_MANIFEST="${MANIFESTS_DIR}/manifest_seg.csv"

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================
MANIFEST_FILES=()
OUTPUT_FILES=()
MANIFEST_SEG=""
MANIFEST_TEX=""
OUTPUT_DIR=""

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
            echo ""
            echo "Examples:"
            echo "  # Update seg and VAE latent manifests"
            echo "  $0 --manifest manifest_seg.csv manifest_vae_latent.csv \\"
            echo "     --output manifest_seg_with_rejections.csv manifest_vae_latent_with_rejections.csv"
            echo ""
            echo "  # Legacy mode"
            echo "  $0 --manifest-seg manifest_seg.csv --manifest-tex manifest_tex.csv --output-dir manifests/"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Launching Add Rejections to Manifest"
echo "=========================================="
echo "Rejections: ${REJECTIONS_FILE}"
echo "Original Manifest: ${ORIGINAL_MANIFEST}"
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
# CREATE TEMPORARY SCRIPT WITH ARGUMENTS
# =============================================================================
TEMP_SCRIPT="${HPC_SCRIPTS_DIR}/temp_add_rejections.sh"
cat > "${TEMP_SCRIPT}" <<EOF
#!/bin/bash
#BSUB -J add_rejections_to_manifest
#BSUB -n 1
#BSUB -R "rusage[mem=4000]"
#BSUB -W 01:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

BASE_DIR="${BASE_DIR}"
SCRIPTS_DIR="${SCRIPTS_DIR}"

cd "\${BASE_DIR}"

if [ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "\$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || conda activate scenefactor || exit 1
fi

# Build command
CMD_ARGS=(
    --rejections "${REJECTIONS_FILE}"
    --original-manifest "${ORIGINAL_MANIFEST}"
)
EOF

# Add manifests and outputs
if [ ${#MANIFEST_FILES[@]} -gt 0 ]; then
    if [ ${#OUTPUT_FILES[@]} -eq 0 ]; then
        echo "ERROR: --output is required when using --manifest" >&2
        rm -f "${TEMP_SCRIPT}"
        exit 1
    fi
    
    if [ ${#MANIFEST_FILES[@]} -ne ${#OUTPUT_FILES[@]} ]; then
        echo "ERROR: Number of manifests (${#MANIFEST_FILES[@]}) must match number of outputs (${#OUTPUT_FILES[@]})" >&2
        rm -f "${TEMP_SCRIPT}"
        exit 1
    fi
    
    for manifest_file in "${MANIFEST_FILES[@]}"; do
        echo "CMD_ARGS+=(--manifest \"${manifest_file}\")" >> "${TEMP_SCRIPT}"
    done
    
    for output_file in "${OUTPUT_FILES[@]}"; do
        echo "CMD_ARGS+=(--output \"${output_file}\")" >> "${TEMP_SCRIPT}"
    done
fi

# Legacy mode
if [ -n "${MANIFEST_SEG}" ] || [ -n "${MANIFEST_TEX}" ]; then
    if [ -z "${OUTPUT_DIR}" ]; then
        echo "ERROR: --output-dir is required when using --manifest-seg/--manifest-tex" >&2
        rm -f "${TEMP_SCRIPT}"
        exit 1
    fi
    
    if [ -n "${MANIFEST_SEG}" ]; then
        echo "CMD_ARGS+=(--manifest-seg \"${MANIFEST_SEG}\")" >> "${TEMP_SCRIPT}"
    fi
    
    if [ -n "${MANIFEST_TEX}" ]; then
        echo "CMD_ARGS+=(--manifest-tex \"${MANIFEST_TEX}\")" >> "${TEMP_SCRIPT}"
    fi
    
    echo "CMD_ARGS+=(--output-dir \"${OUTPUT_DIR}\")" >> "${TEMP_SCRIPT}"
fi

# Add Python command
cat >> "${TEMP_SCRIPT}" <<EOF

python "\${SCRIPTS_DIR}/add_rejections_to_manifest.py" "\${CMD_ARGS[@]}"

exit \$?
EOF

chmod +x "${TEMP_SCRIPT}"

# =============================================================================
# SUBMIT JOB
# =============================================================================
echo "Submitting job..."
JOB_OUTPUT=$(bsub < "${TEMP_SCRIPT}")

JOB_ID=$(echo "${JOB_OUTPUT}" | grep -oP '(?<=Job <)\d+(?=>)' || echo "")

if [ -z "${JOB_ID}" ]; then
    echo "ERROR: Failed to submit job" >&2
    rm -f "${TEMP_SCRIPT}"
    exit 1
fi

echo "  Submitted: Job ${JOB_ID}"
echo "  Check status with: bjobs ${JOB_ID}"
echo ""

# Clean up temp script
rm -f "${TEMP_SCRIPT}"

echo "=========================================="
echo "Job submitted successfully"
echo "=========================================="

