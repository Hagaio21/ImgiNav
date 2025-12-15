#!/bin/bash
# Launch baseline evaluation for multiple checkpoints
#
# Usage:
#   ./launch_eval_baseline.sh checkpoints.txt
#   ./launch_eval_baseline.sh --checkpoint /path/to/single/checkpoint.pt
#   ./launch_eval_baseline.sh --find-best /path/to/experiments/dir
#
# Options:
#   --num-samples N     Number of samples to evaluate (default: 500)
#   --guidance-scale G  CFG guidance scale (default: 7.5)
#   --fov F             Camera field of view in degrees (default: 80)
#   --empty-threshold N Furniture count below which room is empty (default: 3)
#   --eval-mode M       Evaluation mode: "all", "empty", or "furnished" (default: all)
#   --queue Q           Queue to submit to (default: gpul40s)
#   --dry-run           Print commands without submitting
#
# Evaluation modes:
#   all       - Evaluate both empty and furnished rooms together (default)
#   empty     - Evaluate only empty rooms
#   furnished - Evaluate only furnished rooms

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
RUN_SCRIPT="${SCRIPT_DIR}/eval/run_eval_baseline.sh"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Defaults
NUM_SAMPLES=500
GUIDANCE_SCALE=7.5
FOV=80.0
EMPTY_THRESHOLD=3
EVAL_MODE="all"
QUEUE="gpul40s"
DRY_RUN=false
MANIFEST="/work3/s233249/ImgiNav/dataset_v2/manifests/manifest_val.csv"
TAXONOMY="${BASE_DIR}/data_preparation_v2/taxonomy.json"
OUTPUT_DIR="${BASE_DIR}/evaluation_results"

# Parse arguments
CHECKPOINTS=()
CHECKPOINT_FILE=""
FIND_BEST_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --guidance-scale)
            GUIDANCE_SCALE="$2"
            shift 2
            ;;
        --fov)
            FOV="$2"
            shift 2
            ;;
        --empty-threshold)
            EMPTY_THRESHOLD="$2"
            shift 2
            ;;
        --eval-mode)
            EVAL_MODE="$2"
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
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --checkpoint)
            CHECKPOINTS+=("$2")
            shift 2
            ;;
        --find-best)
            FIND_BEST_DIR="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            # Assume it's a checkpoint file
            CHECKPOINT_FILE="$1"
            shift
            ;;
    esac
done

# Validate eval mode
case "${EVAL_MODE}" in
    all|empty|furnished)
        ;;
    *)
        echo "ERROR: Invalid eval-mode '${EVAL_MODE}'. Must be: all, empty, or furnished"
        exit 1
        ;;
esac

# Find best checkpoints in directory
if [ -n "${FIND_BEST_DIR}" ]; then
    echo "Finding best checkpoints in: ${FIND_BEST_DIR}"
    while IFS= read -r -d '' ckpt; do
        CHECKPOINTS+=("$ckpt")
    done < <(find "${FIND_BEST_DIR}" -name "best_checkpoint.pt" -print0 2>/dev/null)
    
    if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
        # Try finding any checkpoint
        while IFS= read -r -d '' ckpt; do
            CHECKPOINTS+=("$ckpt")
        done < <(find "${FIND_BEST_DIR}" -name "*.pt" -print0 2>/dev/null | head -20)
    fi
fi

# Load from file
if [ -n "${CHECKPOINT_FILE}" ] && [ -f "${CHECKPOINT_FILE}" ]; then
    while IFS= read -r line; do
        line=$(echo "$line" | xargs)  # Trim whitespace
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
    echo "  $0 --find-best /path/to/experiments/"
    echo ""
    echo "Options:"
    echo "  --num-samples N      Number of samples (default: 500)"
    echo "  --guidance-scale G   CFG scale (default: 7.5)"
    echo "  --fov F              Camera FOV in degrees (default: 80)"
    echo "  --empty-threshold N  Furniture count threshold for empty (default: 3)"
    echo "  --eval-mode M        Mode: all, empty, furnished (default: all)"
    echo "  --queue Q            LSF queue (default: gpul40s)"
    echo "  --dry-run            Print without submitting"
    exit 1
fi

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "Launching Baseline Evaluation Jobs"
echo "=============================================="
echo "Checkpoints: ${#CHECKPOINTS[@]}"
echo "Samples per checkpoint: ${NUM_SAMPLES}"
echo "Guidance scale: ${GUIDANCE_SCALE}"
echo "FOV: ${FOV}°"
echo "Empty threshold: furniture_count < ${EMPTY_THRESHOLD}"
echo "Eval mode: ${EVAL_MODE}"
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
    
    # Extract experiment name from path
    EXP_NAME=$(basename "$(dirname "${CHECKPOINT}")")
    if [ "${EXP_NAME}" == "checkpoints" ]; then
        EXP_NAME=$(basename "$(dirname "$(dirname "${CHECKPOINT}")")")
    fi
    EXP_NAME=$(echo "${EXP_NAME}" | sed 's/[^a-zA-Z0-9_]/_/g')
    
    # Add eval mode suffix for result checking
    RESULT_SUFFIX=""
    if [ "${EVAL_MODE}" == "empty" ]; then
        RESULT_SUFFIX="_empty"
    elif [ "${EVAL_MODE}" == "furnished" ]; then
        RESULT_SUFFIX="_furnished"
    fi
    
    # Check if already evaluated (per-experiment folder structure)
    EXP_FOLDER="${OUTPUT_DIR}/${EXP_NAME}${RESULT_SUFFIX}"
    if [ -d "${EXP_FOLDER}" ] && ls "${EXP_FOLDER}"/results_*.json 1>/dev/null 2>&1; then
        echo "SKIP: Already evaluated: ${EXP_NAME}${RESULT_SUFFIX}"
        ((SKIPPED++))
        continue
    fi
    
    JOB_NAME="eval_${EXP_NAME}"
    if [ -n "${RESULT_SUFFIX}" ]; then
        JOB_NAME="${JOB_NAME}${RESULT_SUFFIX}"
    fi
    
    echo "Submitting: ${JOB_NAME}"
    echo "  Checkpoint: ${CHECKPOINT}"
    echo "  Eval mode: ${EVAL_MODE}"
    
    if [ "${DRY_RUN}" = true ]; then
        echo "  [DRY RUN] Would submit job"
    else
        bsub -J "${JOB_NAME}" \
            -o "${LOG_DIR}/eval_${EXP_NAME}${RESULT_SUFFIX}.%J.out" \
            -e "${LOG_DIR}/eval_${EXP_NAME}${RESULT_SUFFIX}.%J.err" \
            -q "${QUEUE}" \
            -n 4 \
            -R "rusage[mem=4000]" \
            -gpu "num=1" \
            -W 2:00 \
            -env "CHECKPOINT=${CHECKPOINT},MANIFEST=${MANIFEST},TAXONOMY=${TAXONOMY},OUTPUT_DIR=${OUTPUT_DIR},NUM_SAMPLES=${NUM_SAMPLES},GUIDANCE_SCALE=${GUIDANCE_SCALE},FOV=${FOV},EMPTY_THRESHOLD=${EMPTY_THRESHOLD},EVAL_MODE=${EVAL_MODE}" \
            bash "${RUN_SCRIPT}"
        
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