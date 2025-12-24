#!/bin/bash
#BSUB -J eval_refine
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/eval_refine.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/eval_refine.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 8:00
#BSUB -q gpul40s

set -euo pipefail

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/scripts/evaluate_refinement.py"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Parameters (can be overridden via bsub -env)
CHECKPOINT="${CHECKPOINT:-}"
MANIFEST="${MANIFEST:-/work3/s233249/ImgiNav/dataset_v2/refinement_eval/manifest_refinement.csv}"
TAXONOMY="${TAXONOMY:-${BASE_DIR}/data_preparation_v2/taxonomy.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/refinement_results}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
MAX_POVS="${MAX_POVS:-7}"
NOISE_STRENGTHS="${NOISE_STRENGTHS:-0.3 0.5}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
NUM_STEPS="${NUM_STEPS:-50}"
SAVE_IMAGES="${SAVE_IMAGES:-true}"
# POV columns for column-based manifest format (refinement manifest)
POV_COLUMNS="${POV_COLUMNS:-pov_emb_step0_center pov_emb_step1_left pov_emb_step1_center pov_emb_step1_right pov_emb_step2_left pov_emb_step2_center pov_emb_step2_right}"

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
echo "Refinement Evaluation"
echo "=============================================="
echo "Checkpoint: ${CHECKPOINT}"
echo "Manifest: ${MANIFEST}"
echo "Taxonomy: ${TAXONOMY}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Num samples: ${NUM_SAMPLES}"
echo "Max POVs: ${MAX_POVS}"
echo "Noise strengths: ${NOISE_STRENGTHS}"
echo "Guidance scale: ${GUIDANCE_SCALE}"
echo "Num steps: ${NUM_STEPS}"
echo "Save images: ${SAVE_IMAGES}"
echo "POV columns: ${POV_COLUMNS}"
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

# Build command
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

# Run
eval ${CMD}

echo "Refinement evaluation complete!"