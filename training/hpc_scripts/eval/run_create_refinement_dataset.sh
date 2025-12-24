#!/bin/bash
#BSUB -J prep_refine_ds
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/prep_refine_ds.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/prep_refine_ds.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -W 6:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
SCRIPT="${BASE_DIR}/data_preparation_v2/prepare_refinement_dataset.py"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Parameters (override via bsub -env)
MANIFEST="${MANIFEST:-/work3/s233249/ImgiNav/experiments/diffusion/v2/manifest_val.csv}"
DATASET_ROOT="${DATASET_ROOT:-/work3/s233249/ImgiNav/dataset_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-/work3/s233249/ImgiNav/dataset_v2/refinement_eval}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
SEED="${SEED:-42}"
STEP_DISTANCES="${STEP_DISTANCES:-0.5 1.0}"
SWEEP_ANGLES="${SWEEP_ANGLES:--30 0 30}"
POV_WIDTH="${POV_WIDTH:-1280}"
POV_HEIGHT="${POV_HEIGHT:-720}"
POV_FOV="${POV_FOV:-80.0}"

mkdir -p "${LOG_DIR}"

echo "=============================================="
echo "Prepare Refinement Evaluation Dataset"
echo "=============================================="
echo "Manifest: ${MANIFEST}"
echo "Dataset root: ${DATASET_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Num samples: ${NUM_SAMPLES}"
echo "Seed: ${SEED}"
echo "Step distances: ${STEP_DISTANCES}"
echo "Sweep angles: ${SWEEP_ANGLES}"
echo "POV size: ${POV_WIDTH}x${POV_HEIGHT}"
echo "POV FOV: ${POV_FOV}°"
echo "=============================================="

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

# Build command - use CPU and osmesa for software rendering
CMD="python ${SCRIPT} \
    --manifest ${MANIFEST} \
    --dataset-root ${DATASET_ROOT} \
    --output-dir ${OUTPUT_DIR} \
    --num-samples ${NUM_SAMPLES} \
    --seed ${SEED} \
    --step-distances ${STEP_DISTANCES} \
    --sweep-angles ${SWEEP_ANGLES} \
    --pov-width ${POV_WIDTH} \
    --pov-height ${POV_HEIGHT} \
    --pov-fov ${POV_FOV} \
    --device cpu \
    --hpc --backend osmesa"

echo ""
echo "Running: ${CMD}"
echo ""

eval ${CMD}

echo "=============================================="
echo "Refinement dataset preparation complete!"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="