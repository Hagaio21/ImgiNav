#!/bin/bash
#BSUB -J ae_eval
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/ae_eval.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/ae_eval.%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=16000]"
#BSUB -W 2:00
#BSUB -q hpc

# Set MKL interface layer BEFORE set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

set -euo pipefail

# =============================================================================
# CONFIGURATION - EDIT THESE VARIABLES
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/scripts/evaluate_ae_sweep.py"
EXPERIMENTS_DIR="/work3/s233249/ImgiNav/experiments/ae_diffusion_sweep"
OUTPUT_DIR="/work3/s233249/ImgiNav/experiments/ae_evaluation_results"

# Test image for reconstruction analysis (set to empty string to skip reconstruction analysis)
TEST_IMAGE_PATH="work3/s233249/ImgiNav/datasets/scenes/2befd25c-4ad6-4d8d-a817-85f69d9e1197/layouts/2befd25c-4ad6-4d8d-a817-85f69d9e1197_scene_layout.png"

# Device for model inference (cpu, cuda, or mps)
DEVICE="cpu"

# =============================================================================
# SETUP
# =============================================================================
echo "=========================================="
echo "AUTOENCODER SWEEP EVALUATION"
echo "=========================================="
echo "Job ID: ${LSB_JOBID}"
echo "Start time: $(date)"
echo ""
echo "[CONFIGURATION]"
echo "  Python script: ${PYTHON_SCRIPT}"
echo "  Experiments directory: ${EXPERIMENTS_DIR}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Test image: ${TEST_IMAGE_PATH}"
echo "  Device: ${DEVICE}"
echo "=========================================="
echo ""

# Create necessary directories
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

echo "[DEPENDENCIES] Checking required packages..."
python -c "
import sys

packages = {
    'pandas': 'pandas',
    'matplotlib': 'matplotlib', 
    'seaborn': 'seaborn',
    'yaml': 'pyyaml',
    'numpy': 'numpy',
    'torch': 'torch',
    'torchvision': 'torchvision',
    'PIL': 'pillow'
}

missing = []
for module, pkg in packages.items():
    try:
        mod = __import__(module)
        if hasattr(mod, '__version__'):
            print(f'  ✓ {module}: {mod.__version__}')
        else:
            print(f'  ✓ {module}: installed')
    except ImportError:
        print(f'  ✗ {module}: not found')
        missing.append(pkg)

if missing:
    print('\nMissing packages: ' + ', '.join(missing))
    sys.exit(1)
" || {
  echo ""
  echo "Installing missing packages..."
  pip install --no-cache-dir pandas matplotlib seaborn pyyaml numpy torch torchvision pillow
  echo ""
  python -c "import pandas, matplotlib, seaborn, yaml, numpy, torch, torchvision, PIL; print('✓ All packages installed')"
}
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
NUM_EXPERIMENTS=$(find "${EXPERIMENTS_DIR}" -maxdepth 1 -type d -name "*ae_*" | wc -l)
echo "  ✓ Experiments directory exists (${NUM_EXPERIMENTS} experiment dirs)"

if [ -n "${TEST_IMAGE_PATH}" ] && [ ! -f "${TEST_IMAGE_PATH}" ]; then
  echo "  ⚠ WARNING: Test image not found: ${TEST_IMAGE_PATH}"
  echo "  → Skipping reconstruction analysis"
  TEST_IMAGE_PATH=""
fi

if [ -n "${TEST_IMAGE_PATH}" ]; then
  echo "  ✓ Test image exists"
fi
echo ""

# =============================================================================
# BUILD COMMAND
# =============================================================================
CMD="python ${PYTHON_SCRIPT} \
  --experiments_dir ${EXPERIMENTS_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --device ${DEVICE}"

# Add test image if provided
if [ -n "${TEST_IMAGE_PATH}" ]; then
  CMD="${CMD} --test_image ${TEST_IMAGE_PATH}"
fi

# =============================================================================
# RUN EVALUATION
# =============================================================================
echo "=========================================="
echo "RUNNING EVALUATION"
echo "=========================================="
echo ""
echo "[COMMAND]"
echo "${CMD}"
echo ""

eval ${CMD} || {
  echo "" >&2
  echo "==========================================" >&2
  echo "ERROR: Evaluation failed" >&2
  echo "==========================================" >&2
  exit 1
}

# =============================================================================
# COMPLETION
# =============================================================================
echo ""
echo "=========================================="
echo "EVALUATION COMPLETED SUCCESSFULLY"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""

# List generated files
if [ -d "${OUTPUT_DIR}" ]; then
  echo "[OUTPUT FILES]"
  ls -lh "${OUTPUT_DIR}" | tail -n +2 | awk '{printf "  %s  %s\n", $5, $9}'
  echo ""
  
  # Count files by type
  PNG_COUNT=$(find "${OUTPUT_DIR}" -name "*.png" | wc -l)
  SVG_COUNT=$(find "${OUTPUT_DIR}" -name "*.svg" | wc -l)
  CSV_COUNT=$(find "${OUTPUT_DIR}" -name "*.csv" | wc -l)
  
  echo "[SUMMARY]"
  echo "  PNG files: ${PNG_COUNT}"
  echo "  SVG files: ${SVG_COUNT}"
  echo "  CSV files: ${CSV_COUNT}"
  echo ""
  
  # Show summary stats if available
  if [ -f "${OUTPUT_DIR}/summary_table.csv" ]; then
    echo "[TOP 5 CONFIGURATIONS]"
    head -n 6 "${OUTPUT_DIR}/summary_table.csv" | column -t -s ','
    echo ""
  fi
fi

echo "=========================================="
echo "Job ${LSB_JOBID} finished"
echo "=========================================="