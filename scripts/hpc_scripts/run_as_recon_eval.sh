#!/bin/bash
#BSUB -J ae_compare
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/ae_compare.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/ae_compare.%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=32000]"
#BSUB -W 4:00
#BSUB -q hpc

export MKL_INTERFACE_LAYER=LP64

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/scripts/analyze_reconstruction.py"
EXPERIMENTS_DIR="/work3/s233249/ImgiNav/experiments/ae_diffusion_sweep"
OUTPUT_DIR="/work3/s233249/ImgiNav/experiments/ae_model_comparison"
TAXONOMY_PATH="/work3/s233249/ImgiNav/ImgiNav/config/taxonomy.json"

# Test images - add paths to your test images here
TEST_IMAGE_1="/work3/s233249/ImgiNav/datasets/scenes/ffab746b-2e40-4bb7-8e69-51bb20f09ce1/layouts/ffab746b-2e40-4bb7-8e69-51bb20f09ce1_scene_layout.png"
TEST_IMAGE_2="/work3/s233249/ImgiNav/datasets/scenes/2befd25c-4ad6-4d8d-a817-85f69d9e1197/layouts/2befd25c-4ad6-4d8d-a817-85f69d9e1197_scene_layout.png"
TEST_IMAGE_3="/work3/s233249/ImgiNav/datasets/scenes/ffed9e6c-5e6d-49aa-ba90-83927369ff47/layouts/ffed9e6c-5e6d-49aa-ba90-83927369ff47_scene_layout.png"

# Number of top models to compare (based on training loss)
NUM_MODELS=10

# Device for inference
DEVICE="cpu"

# =============================================================================
# SETUP
# =============================================================================
echo "=========================================="
echo "MULTI-MODEL RECONSTRUCTION COMPARISON"
echo "=========================================="
echo "Job ID: ${LSB_JOBID}"
echo "Start time: $(date)"
echo ""
echo "[CONFIGURATION]"
echo "  Python script: ${PYTHON_SCRIPT}"
echo "  Experiments directory: ${EXPERIMENTS_DIR}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Number of models: ${NUM_MODELS}"
echo "  Device: ${DEVICE}"
echo "=========================================="
echo ""

mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname "$0")/logs"

# =============================================================================
# CONDA ACTIVATION
# =============================================================================
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || {
    echo "Failed to activate conda environment 'imginav'" >&2
    echo "Trying fallback environment 'scenefactor'..." >&2
    conda activate scenefactor || {
      echo "Failed to activate any conda environment" >&2
      exit 1
    }
  }
fi

echo "[CONDA] Environment: ${CONDA_DEFAULT_ENV:-none}"
echo ""

# =============================================================================
# ENVIRONMENT VERIFICATION
# =============================================================================
echo "[ENVIRONMENT] Python info:"
echo "  Version: $(python --version)"
echo "  Path: $(which python)"
echo ""

echo "[DEPENDENCIES] Checking packages..."
python -c "
import sys
packages = ['torch', 'torchvision', 'PIL', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'yaml']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print('  ✓ ' + pkg)
    except ImportError:
        print('  ✗ ' + pkg)
        missing.append(pkg)
if missing:
    print('\nMissing: ' + ', '.join(missing))
    sys.exit(1)
"
echo ""

# =============================================================================
# VALIDATE INPUTS
# =============================================================================
echo "[VALIDATION] Checking inputs..."

if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "  ✗ ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
  exit 1
fi
echo "  ✓ Python script exists"

if [ ! -d "${EXPERIMENTS_DIR}" ]; then
  echo "  ✗ ERROR: Experiments directory not found: ${EXPERIMENTS_DIR}" >&2
  exit 1
fi
NUM_EXPERIMENTS=$(find "${EXPERIMENTS_DIR}" -name "best.pt" 2>/dev/null | wc -l)
echo "  ✓ Experiments directory exists (${NUM_EXPERIMENTS} models with checkpoints)"

# Validate test images and build list
TEST_IMAGES_ARG=""
VALID_COUNT=0

if [ -f "${TEST_IMAGE_1}" ]; then
  TEST_IMAGES_ARG="${TEST_IMAGES_ARG} ${TEST_IMAGE_1}"
  VALID_COUNT=$((VALID_COUNT + 1))
  echo "  ✓ Test image 1: ${TEST_IMAGE_1}"
else
  echo "  ⚠ WARNING: Image 1 not found: ${TEST_IMAGE_1}"
fi

if [ -f "${TEST_IMAGE_2}" ]; then
  TEST_IMAGES_ARG="${TEST_IMAGES_ARG} ${TEST_IMAGE_2}"
  VALID_COUNT=$((VALID_COUNT + 1))
  echo "  ✓ Test image 2: ${TEST_IMAGE_2}"
else
  echo "  ⚠ WARNING: Image 2 not found: ${TEST_IMAGE_2}"
fi

if [ -f "${TEST_IMAGE_3}" ]; then
  TEST_IMAGES_ARG="${TEST_IMAGES_ARG} ${TEST_IMAGE_3}"
  VALID_COUNT=$((VALID_COUNT + 1))
  echo "  ✓ Test image 3: ${TEST_IMAGE_3}"
else
  echo "  ⚠ WARNING: Image 3 not found: ${TEST_IMAGE_3}"
fi

if [ ${VALID_COUNT} -eq 0 ]; then
  echo "  ✗ ERROR: No valid test images found" >&2
  exit 1
fi

echo "  Total valid test images: ${VALID_COUNT}"
echo ""

# =============================================================================
# RUN COMPARISON
# =============================================================================
echo "=========================================="
echo "RUNNING MODEL COMPARISON"
echo "=========================================="
echo ""

python "${PYTHON_SCRIPT}" \
  --experiments_dir "${EXPERIMENTS_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_models ${NUM_MODELS} \
  --device ${DEVICE} \
  --test_images ${TEST_IMAGES_ARG} || {
  echo "" >&2
  echo "==========================================" >&2
  echo "ERROR: Model comparison failed" >&2
  echo "==========================================" >&2
  exit 1
}

# =============================================================================
# COMPLETION
# =============================================================================
echo ""
echo "=========================================="
echo "COMPARISON COMPLETED SUCCESSFULLY"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""

if [ -d "${OUTPUT_DIR}" ]; then
  echo "[OUTPUT FILES]"
  ls -lh "${OUTPUT_DIR}" 2>/dev/null | tail -n +2 | awk '{printf "  %s  %s\n", $5, $9}'
  echo ""
  
  PNG_COUNT=$(find "${OUTPUT_DIR}" -name "*.png" 2>/dev/null | wc -l)
  CSV_COUNT=$(find "${OUTPUT_DIR}" -name "*.csv" 2>/dev/null | wc -l)
  
  echo "[SUMMARY]"
  echo "  PNG files: ${PNG_COUNT}"
  echo "  CSV files: ${CSV_COUNT}"
  echo ""
  
  if [ -f "${OUTPUT_DIR}/comparison_metrics.csv" ]; then
    echo "[TOP 5 MODELS BY OVERALL MAE]"
    head -n 1 "${OUTPUT_DIR}/comparison_metrics.csv"
    tail -n +2 "${OUTPUT_DIR}/comparison_metrics.csv" | sort -t',' -k7 -n | head -n 5
    echo ""
  fi
fi

echo "=========================================="
echo "Job ${LSB_JOBID} finished"
echo "=========================================="