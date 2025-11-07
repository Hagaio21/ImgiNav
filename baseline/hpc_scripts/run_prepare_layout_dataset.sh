#!/bin/bash
#BSUB -J prepare_sd_dataset
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/baseline/hpc_scripts/logs/prepare_sd_dataset.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/baseline/hpc_scripts/logs/prepare_sd_dataset.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -W 2:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
SCRIPT_PATH="${BASE_DIR}/baseline/prepare_layout_dataset.py"
LOG_DIR="${BASE_DIR}/baseline/hpc_scripts/logs"

# Input manifest (augmented dataset)
INPUT_MANIFEST="/work3/s233249/ImgiNav/datasets/augmented/manifest.csv"

# Output directory for layout images
OUTPUT_DIR="/work3/s233249/ImgiNav/datasets/sd_finetuning_images"

# Number of samples to extract
NUM_SAMPLES=5000

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "========================================="
echo "Preparing Layout Dataset for SD Fine-tuning"
echo "========================================="
echo "Script: ${SCRIPT_PATH}"
echo "Input manifest: ${INPUT_MANIFEST}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Number of samples: ${NUM_SAMPLES}"
echo ""

# Check required files
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "ERROR: Script not found: ${SCRIPT_PATH}" >&2
    exit 1
fi
if [ ! -f "${INPUT_MANIFEST}" ]; then
    echo "ERROR: Input manifest not found: ${INPUT_MANIFEST}" >&2
    exit 1
fi

# ----------------------------------------------------------------------
# Conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || conda activate scenefactor || exit 1
fi

# ----------------------------------------------------------------------
# Run script
cd "${BASE_DIR}"
python "${SCRIPT_PATH}" \
    --manifest "${INPUT_MANIFEST}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_samples "${NUM_SAMPLES}" \
    --filter_empty \
    --filter_augmented

echo ""
echo "========================================="
echo "Dataset Preparation Completed"
echo "========================================="
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Next step: Fine-tune SD using:"
echo "  bsub < baseline/hpc_scripts/run_finetune_sd.sh"

