#!/bin/bash
#BSUB -J create_new_layouts_scenes
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/create_new_layouts_scenes.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/create_new_layouts_scenes.%J.err
#BSUB -n 10
#BSUB -R "rusage[mem=4000]"
#BSUB -W 10:00
#BSUB -q hpc

set -euo pipefail

# Configuration
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
SCENES_ROOT="/work3/s233249/ImgiNav/datasets/scenes"
TAXONOMY_FILE="${BASE_DIR}/config/taxonomy.json"
PYTHON_SCRIPT="${BASE_DIR}/data_preparation/create_new_layouts.py"
OUTPUT_DIR="/work3/s233249/ImgiNav/datasets"
LOG_DIR="${BASE_DIR}/data_preparation/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

echo "Running new scene layout creation"
echo "Job ID: ${LSB_JOBID:-unknown}"

# Verify Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}"
  exit 1
fi

# Verify taxonomy file exists
if [ ! -f "${TAXONOMY_FILE}" ]; then
  echo "ERROR: Taxonomy file not found: ${TAXONOMY_FILE}"
  exit 1
fi

# Verify scenes directory exists
if [ ! -d "${SCENES_ROOT}" ]; then
  echo "ERROR: Scenes directory not found: ${SCENES_ROOT}"
  exit 1
fi

# Conda activation
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

# Change to base directory and set PYTHONPATH so Python can find common module
cd "${BASE_DIR}"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"

# Run create_new_layouts for scenes only
# Scenes use smaller point sizes: point-size=3, object-multiplier=1.0
python "${PYTHON_SCRIPT}" \
  --in_root "${SCENES_ROOT}" \
  --taxonomy "${TAXONOMY_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --mode "scene" \
  --res 512 \
  --hmin 0.1 \
  --hmax 1.8 \
  --point-size 5 \
  --scene-point-size 3 \
  --object-point-size-multiplier 1.5 \
  --scene-object-point-size-multiplier 1.0 \
  --min-colors 4 \
  --max-whiteness 0.95 \
  --color-mode "category"

echo "New scene layout creation completed successfully"

