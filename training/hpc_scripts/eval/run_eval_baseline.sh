#!/bin/bash
#BSUB -J eval_baseline
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/eval_baseline.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/eval_baseline.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 4:00
#BSUB -q gpul40s

set -euo pipefail

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/scripts/evaluate_baseline.py"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Parameters (can be overridden via bsub -env)
CHECKPOINT="${CHECKPOINT:-}"
MANIFEST="${MANIFEST:-/work3/s233249/ImgiNav/experiments/diffusion/v2/manifest_val.csv}"
TAXONOMY="${TAXONOMY:-${BASE_DIR}/data_preparation_v2/taxonomy.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/evaluation_results}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
NUM_STEPS="${NUM_STEPS:-50}"
FOV="${FOV:-80.0}"

# Evaluation mode: "all", "empty", or "furnished"
# Default is "all" which evaluates both empty and furnished rooms together
EVAL_MODE="${EVAL_MODE:-all}"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Validate checkpoint
if [ -z "${CHECKPOINT}" ]; then
    echo "ERROR: CHECKPOINT not specified. Use: bsub -env \"CHECKPOINT=/path/to/checkpoint.pt\" ..." >&2
    exit 1
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi

echo "=============================================="
echo "Baseline Evaluation (Supercategory Metrics)"
echo "=============================================="
echo "Checkpoint: ${CHECKPOINT}"
echo "Manifest: ${MANIFEST}"
echo "Taxonomy: ${TAXONOMY}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Num samples: ${NUM_SAMPLES}"
echo "Guidance scale: ${GUIDANCE_SCALE}"
echo "Num steps: ${NUM_STEPS}"
echo "FOV: ${FOV}°"
echo "Eval mode: ${EVAL_MODE}"
echo "=============================================="

# Load modules
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate conda
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

cd "${BASE_DIR}"

# Build eval mode arguments
EVAL_MODE_ARGS=""
case "${EVAL_MODE}" in
    empty)
        EVAL_MODE_ARGS="--empty-only"
        ;;
    furnished)
        EVAL_MODE_ARGS="--furnished-only"
        ;;
    all)
        # Default: evaluate both empty and furnished together
        EVAL_MODE_ARGS=""
        ;;
    *)
        echo "WARNING: Unknown EVAL_MODE '${EVAL_MODE}', using 'all'"
        EVAL_MODE_ARGS=""
        ;;
esac

# Run evaluation
python "${PYTHON_SCRIPT}" \
    --checkpoint "${CHECKPOINT}" \
    --manifest "${MANIFEST}" \
    --taxonomy "${TAXONOMY}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-samples "${NUM_SAMPLES}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    --num-steps "${NUM_STEPS}" \
    --fov "${FOV}" \
    --save-images \
    ${EVAL_MODE_ARGS}

echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
