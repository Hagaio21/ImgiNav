#!/bin/bash
# Launch refinement evaluation for multiple checkpoints
#
# Usage:
#   ./launch_eval_refinement.sh                                    # Uses default checkpoint file
#   ./launch_eval_refinement.sh /path/to/checkpoints.txt           # Custom checkpoint file
#   ./launch_eval_refinement.sh --checkpoint /path/to/single.pt    # Single checkpoint
#
# Options:
#   --num-samples N       Number of samples to evaluate (default: 50)
#   --max-povs N          Max POVs per sample (default: 7)
#   --noise-strengths     Noise strengths to test (default: "0.3 0.5")
#   --queue Q             Queue to submit to (default: gpul40s)
#   --no-images           Don't save progression images
#   --dry-run             Print commands without submitting

set -euo pipefail

# ============================================
# CONFIGURATION - Defaults
# ============================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
DEFAULT_CHECKPOINT_FILE="/work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/eval/checkpoints_to_eval_refinement.txt"
MANIFEST="/work3/s233249/ImgiNav/dataset_v2/refinement_eval/manifest_refinement.csv"
TAXONOMY="${BASE_DIR}/data_preparation_v2/taxonomy.json"
OUTPUT_DIR="${BASE_DIR}/refinement_results"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"
PYTHON_SCRIPT="${BASE_DIR}/scripts/evaluate_refinement.py"

NUM_SAMPLES=200
MAX_POVS=7
NOISE_STRENGTHS="0.5 0.8"
GUIDANCE_SCALE=7.5
NUM_STEPS=50
QUEUE="gpul40s"
DRY_RUN=false
SAVE_IMAGES=true

POV_COLUMNS="pov_emb_step0_center pov_emb_step1_left pov_emb_step1_center pov_emb_step1_right pov_emb_step2_left pov_emb_step2_center pov_emb_step2_right"

# ============================================
# Parse arguments
# ============================================
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

# Use default checkpoint file if none specified
if [ ${#CHECKPOINTS[@]} -eq 0 ] && [ -z "${CHECKPOINT_FILE}" ]; then
    CHECKPOINT_FILE="${DEFAULT_CHECKPOINT_FILE}"
fi

# Load checkpoints from file
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
    echo "No checkpoints found!"
    echo ""
    echo "Usage:"
    echo "  $0                                        # Uses default: ${DEFAULT_CHECKPOINT_FILE}"
    echo "  $0 /path/to/checkpoints.txt              # Custom checkpoint file"
    echo "  $0 --checkpoint /path/to/checkpoint.pt   # Single checkpoint"
    echo ""
    echo "Options:"
    echo "  --num-samples N       Samples per checkpoint (default: 50)"
    echo "  --max-povs N          Max POVs (default: 7)"
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
        SKIPPED=$((SKIPPED + 1))
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
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    
    JOB_NAME="refine_${EXP_NAME}"
    
    echo "Submitting: ${JOB_NAME}"
    echo "  Checkpoint: ${CHECKPOINT}"
    
    # Build the python command
    CMD="python ${PYTHON_SCRIPT} \
        --checkpoint ${CHECKPOINT} \
        --manifest ${MANIFEST} \
        --taxonomy ${TAXONOMY} \
        --output-dir ${OUTPUT_DIR} \
        --max-samples ${NUM_SAMPLES} \
        --max-povs ${MAX_POVS} \
        --noise-strengths ${NOISE_STRENGTHS} \
        --guidance-scale ${GUIDANCE_SCALE} \
        --num-steps ${NUM_STEPS} \
        --pov-columns ${POV_COLUMNS}"
    
    if [ "${SAVE_IMAGES}" = "true" ]; then
        CMD="${CMD} --save-images"
    fi
    
    if [ "${DRY_RUN}" = true ]; then
        echo "  [DRY RUN] Would submit job"
        echo "  Command: ${CMD}"
    else
        # Submit job with proper resource specs
        bsub -J "${JOB_NAME}" \
             -o "${LOG_DIR}/refine_${EXP_NAME}.%J.out" \
             -e "${LOG_DIR}/refine_${EXP_NAME}.%J.err" \
             -n 4 \
             -R "rusage[mem=16GB]" \
             -R "span[hosts=1]" \
             -gpu "num=1" \
             -W 8:00 \
             -q "${QUEUE}" \
             <<EOF
#!/bin/bash
set -euo pipefail

# Load modules
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate conda
if [ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "\$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || conda activate scenefactor
fi

cd "${BASE_DIR}"

echo "Running: ${CMD}"
${CMD}

echo "Refinement evaluation complete!"
EOF
        
        SUBMITTED=$((SUBMITTED + 1))
    fi
    echo ""
done

echo "=============================================="
echo "Summary"
echo "=============================================="
echo "Submitted: ${SUBMITTED}"
echo "Skipped: ${SKIPPED}"
echo "=============================================="