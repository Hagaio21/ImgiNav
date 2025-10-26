#!/bin/bash
#BSUB -J ae_identity
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/scripts/hpc_scripts/logs/ae_identity.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/scripts/hpc_scripts/logs/ae_identity.%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=16000]"
#BSUB -W 2:00
#BSUB -q hpc

export MKL_INTERFACE_LAYER=LP64
set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/scripts/test_autoencoder_identity.py"
CONFIG_PATH="/work3/s233249/ImgiNav/experiments/ae_configs/config_diff_4ch_32x32_vanilla.yml"
CHECKPOINT_PATH="/work3/s233249/ImgiNav/experiments/ae_diffusion_sweep/20250930-222728_ae_diff_4ch_32x32_vanilla/best.pt"
LAYOUT_MANIFEST="/work3/s233249/ImgiNav/datasets/layouts.csv"
DEVICE="cpu"
BATCH_SIZE=64

# =============================================================================
# LOGGING
# =============================================================================
echo "=========================================="
echo "AUTOENCODER IDENTITY CHECK"
echo "=========================================="
echo "Job ID: ${LSB_JOBID}"
echo "Start time: $(date)"
echo ""
echo "[CONFIGURATION]"
echo "  Config: ${CONFIG_PATH}"
echo "  Checkpoint: ${CHECKPOINT_PATH}"
echo "  Manifest: ${LAYOUT_MANIFEST}"
echo "  Device: ${DEVICE}"
echo "  Batch size: ${BATCH_SIZE}"
echo "=========================================="
echo ""

mkdir -p "$(dirname "$0")/logs"

# =============================================================================
# CONDA ACTIVATION
# =============================================================================
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || {
    echo "Failed to activate conda env 'imginav'" >&2
    exit 1
  }
fi
echo "[CONDA] Environment: ${CONDA_DEFAULT_ENV:-none}"
echo ""

# =============================================================================
# VERIFY PYTHON AND DEPENDENCIES
# =============================================================================
echo "[ENVIRONMENT] Python info:"
echo "  $(python --version)"
echo "  $(which python)"
echo ""

echo "[DEPENDENCIES] Checking required packages..."
python - <<'PY'
import importlib, sys
req = ["torch","torchvision","yaml","numpy","matplotlib","seaborn","pandas","sklearn","PIL"]
missing = [p for p in req if importlib.util.find_spec(p) is None]
if missing:
    print("Missing:", ", ".join(missing))
    sys.exit(1)
else:
    print("✓ All required packages found")
PY
echo ""

# =============================================================================
# RUN TEST
# =============================================================================
CMD="python ${PYTHON_SCRIPT} \
  --config ${CONFIG_PATH} \
  --checkpoint ${CHECKPOINT_PATH} \
  --layout_manifest ${LAYOUT_MANIFEST} \
  --device ${DEVICE} \
  --batch_size ${BATCH_SIZE}"

echo "[COMMAND]"
echo "${CMD}"
echo "=========================================="
echo ""

eval ${CMD}

# =============================================================================
# COMPLETION
# =============================================================================
echo ""
echo "=========================================="
echo "IDENTITY TEST COMPLETED"
echo "=========================================="
echo "End time: $(date)"
echo ""
