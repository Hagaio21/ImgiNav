#!/bin/bash
# Launch refinement evaluation for multiple checkpoints
#
# Usage:
#   ./launch_eval_refinement.sh checkpoints.txt
#   ./launch_eval_refinement.sh --checkpoint /path/to/single/checkpoint.pt
#
# Options:
#   --num-samples N       Number of samples to evaluate (default: 50)
#   --max-povs N          Max POVs per sample (default: 5)
#   --noise-strengths     Noise strengths to test (default: "0.3 0.5")
#   --queue Q             Queue to submit to (default: gpul40s)
#   --no-images           Don't save progression images
#   --dry-run             Print commands without submitting

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
RUN_SCRIPT="${SCRIPT_DIR}/eval/run_eval_refinement.sh"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Defaults
NUM_SAMPLES=50
MAX_POVS=5
NOISE_STRENGTHS="0.3 0.5"
GUIDANCE_SCALE=7.5
QUEUE="gpul40s"
DRY_RUN=false
SAVE_IMAGES=true
MANIFEST="/work3/s233249/ImgiNav/experiments/diffusion/v2/manifest_val.csv"
TAXONOMY="${BASE_DIR}/data_preparation_v2/taxonomy.json"
OUTPUT_DIR="${BASE_DIR}/refinement_results"

# Parse arguments
CHECKPOINTS=()
CHECKPOINT_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --max-povs)
            MAX_POVS="$2"
            shift 2
            ;;
        --noise-strengths)
            NOISE_STRENGTHS="$2"
            shift 2
            ;;
        --guidance-scale)
            GUIDANCE_SCALE="$2"
            shift 2
            ;;
        --queue)
            QUEUE="$2"
            shift 2
            ;;
        --manifest)
            MANIFEST="$2"
            shift 2
            ;;
        --taxonomy)
            TAXONOMY="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-images)
            SAVE_IMAGES=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --checkpoint)
            CHECKPOINTS+=("$2")
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            CHECKPOINT_FILE="$1"
            shift
            ;;
    esac
done

# Load from file
if [ -n "${CHECKPOINT_FILE}" ] && [ -f "${CHECKPOINT_FILE}" ]; then
    while IFS= read -r line; do
        line=$(echo "$line" | xargs)
        if [ -n "$line" ] && [[ ! "$line" =~ ^# ]]; then
            CHECKPOINTS+=("$line")
        fi
    done < "${CHECKPOINT_FILE}"
fi

# Validate
if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "No checkpoints specified!"
    echo ""
    echo "Usage:"
    echo "  $0 checkpoints.txt"
    echo "  $0 --checkpoint /path/to/checkpoint.pt"
    echo ""
    echo "Options:"
    echo "  --num-samples N       Samples per checkpoint (default: 50)"
    echo "  --max-povs N          Max POVs (default: 5)"
    echo "  --noise-strengths S   Noise strengths (default: \"0.3 0.5\")"
    echo "  --queue Q             LSF queue (default: gpul40s)"
    echo "  --no-images           Don't save progression images"
    echo "  --dry-run             Print without submitting"
    exit 1
fi

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "Launching Refinement Evaluation Jobs"
echo "=============================================="
echo "Checkpoints: ${#CHECKPOINTS[@]}"
echo "Samples per checkpoint: ${NUM_SAMPLES}"
echo "Max POVs: ${MAX_POVS}"
echo "Noise strengths: ${NOISE_STRENGTHS}"
echo "Save images: ${SAVE_IMAGES}"
echo "Queue: ${QUEUE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "=============================================="
echo ""

SUBMITTED=0
SKIPPED=0

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    # Validate checkpoint exists
    if [ ! -f "${CHECKPOINT}" ]; then
        echo "SKIP: Checkpoint not found: ${CHECKPOINT}"
        ((SKIPPED++))
        continue
    fi
    
    # Extract experiment name
    EXP_NAME=$(basename "$(dirname "${CHECKPOINT}")")
    if [ "${EXP_NAME}" == "checkpoints" ]; then
        EXP_NAME=$(basename "$(dirname "$(dirname "${CHECKPOINT}")")")
    fi
    EXP_NAME=$(echo "${EXP_NAME}" | sed 's/[^a-zA-Z0-9_]/_/g')
    
    # Check if already evaluated
    RESULT_PATTERN="${OUTPUT_DIR}/${EXP_NAME}_refinement_*.json"
    if ls ${RESULT_PATTERN} 1>/dev/null 2>&1; then
        echo "SKIP: Already evaluated: ${EXP_NAME}"
        ((SKIPPED++))
        continue
    fi
    
    JOB_NAME="refine_${EXP_NAME}"
    
    echo "Submitting: ${JOB_NAME}"
    echo "  Checkpoint: ${CHECKPOINT}"
    
    if [ "${DRY_RUN}" = true ]; then
        echo "  [DRY RUN] Would submit job"
    else
        bsub -J "${JOB_NAME}" \
             -o "${LOG_DIR}/refine_${EXP_NAME}.%J.out" \
             -e "${LOG_DIR}/refine_${EXP_NAME}.%J.err" \
             -q "${QUEUE}" \
             -env "CHECKPOINT=${CHECKPOINT},MANIFEST=${MANIFEST},TAXONOMY=${TAXONOMY},OUTPUT_DIR=${OUTPUT_DIR},NUM_SAMPLES=${NUM_SAMPLES},MAX_POVS=${MAX_POVS},NOISE_STRENGTHS=${NOISE_STRENGTHS},GUIDANCE_SCALE=${GUIDANCE_SCALE},SAVE_IMAGES=${SAVE_IMAGES}" \
             "${RUN_SCRIPT}"
        
        ((SUBMITTED++))
    fi
    echo ""
done

echo "=============================================="
echo "Summary"
echo "=============================================="
echo "Submitted: ${SUBMITTED}"
echo "Skipped: ${SKIPPED}"
echo "=============================================="
