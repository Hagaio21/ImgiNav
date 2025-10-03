#!/bin/bash
#BSUB -J ae_compare
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/ae_compare.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/ae_compare.%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=64000]"
#BSUB -W 4:00
#BSUB -q gpul40s

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

# Dataset manifest for sampling test images
LAYOUT_MANIFEST="/work3/s233249/ImgiNav/datasets/layouts.csv"

# Percentage of dataset to use (0.1 = 10%)
SAMPLE_RATIO=0.01

# Number of top models to compare (based on training loss)
NUM_MODELS=27

# Device for inference
DEVICE="cuda"
BATCH_SIZE=100

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
echo "  Layout manifest: ${LAYOUT_MANIFEST}"
echo "  Sample ratio: ${SAMPLE_RATIO} ($(echo "${SAMPLE_RATIO} * 100" | bc)%)"
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
# VALIDATE INPUTS & SAMPLE TEST IMAGES
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

if [ ! -f "${LAYOUT_MANIFEST}" ]; then
  echo "  ✗ ERROR: Layout manifest not found: ${LAYOUT_MANIFEST}" >&2
  exit 1
fi
echo "  ✓ Layout manifest exists"

# Sample test images from the dataset
echo ""
echo "[SAMPLING] Extracting test images from dataset..."

TEST_IMAGES_FILE="${OUTPUT_DIR}/sampled_test_images.txt"

python -c "
import pandas as pd
import sys

manifest_path = '${LAYOUT_MANIFEST}'
sample_ratio = ${SAMPLE_RATIO}
output_file = '${TEST_IMAGES_FILE}'

try:
    df = pd.read_csv(manifest_path)
    
    # Filter non-empty layouts
    df = df[df['is_empty'] == False]
    
    # Sample
    n_samples = max(1, int(len(df) * sample_ratio))
    sampled = df.sample(n=n_samples, random_state=42)
    
    # Extract paths
    image_paths = sampled['layout_path'].tolist()
    
    # Save to file
    with open(output_file, 'w') as f:
        for path in image_paths:
            f.write(path + '\n')
    
    print(f'  Total layouts: {len(df)}')
    print(f'  Sampled: {n_samples} ({sample_ratio*100:.1f}%)')
    print(f'  Saved to: {output_file}')
    
except Exception as e:
    print(f'  ✗ ERROR: Failed to sample images: {e}', file=sys.stderr)
    sys.exit(1)
" || exit 1

# Read sampled images into array
if [ ! -f "${TEST_IMAGES_FILE}" ]; then
  echo "  ✗ ERROR: Failed to generate test images file" >&2
  exit 1
fi

# Build command-line arguments for test images
TEST_IMAGES_ARG=""
VALID_COUNT=0

while IFS= read -r img_path; do
  if [ -f "${img_path}" ]; then
    TEST_IMAGES_ARG="${TEST_IMAGES_ARG} ${img_path}"
    VALID_COUNT=$((VALID_COUNT + 1))
  else
    echo "  ⚠ WARNING: Image not found: ${img_path}"
  fi
done < "${TEST_IMAGES_FILE}"

if [ ${VALID_COUNT} -eq 0 ]; then
  echo "  ✗ ERROR: No valid test images found" >&2
  exit 1
fi

echo "  ✓ Valid test images: ${VALID_COUNT}"
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
  --taxonomy_path "${TAXONOMY_PATH}" \
  --batch_size ${BATCH_SIZE} \
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
  SVG_COUNT=$(find "${OUTPUT_DIR}" -name "*.svg" 2>/dev/null | wc -l)
  CSV_COUNT=$(find "${OUTPUT_DIR}" -name "*.csv" 2>/dev/null | wc -l)
  
  echo "[SUMMARY]"
  echo "  PNG files: ${PNG_COUNT}"
  echo "  SVG files: ${SVG_COUNT}"
  echo "  CSV files: ${CSV_COUNT}"
  echo "  Test images used: ${VALID_COUNT}"
  echo ""
  
  if [ -f "${OUTPUT_DIR}/comparison_metrics.csv" ]; then
    echo "[TOP 5 MODELS BY OVERALL MAE]"
    head -n 1 "${OUTPUT_DIR}/comparison_metrics.csv"
    tail -n +2 "${OUTPUT_DIR}/comparison_metrics.csv" | sort -t',' -k7 -n | head -n 5
    echo ""
  fi
  
  echo "[GENERATED VISUALIZATIONS]"
  echo "  - 3 family comparison plots (vanilla, medium, deep)"
  echo "  - Aggregate performance comparison"
  echo "  - Color channel analysis"
  echo "  - Model ranking"
  echo "  - Class color preservation (by architecture and channels)"
  echo ""
fi

echo "=========================================="
echo "Job ${LSB_JOBID} finished"
echo "=========================================="