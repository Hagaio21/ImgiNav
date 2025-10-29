#!/bin/bash
#BSUB -J create_zmap
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/visualization/hpc_scripts/logs/create_zmap.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/visualization/hpc_scripts/logs/create_zmap.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=32000]"
#BSUB -W 10:00
#BSUB -q hpc
set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
# --- Base project directory
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"

# --- The python script to run
PYTHON_SCRIPT="${BASE_DIR}/visualization/lifting_utils.py"

# --- Directory containing all your scene/room parquet files
# !!! PLEASE VERIFY THIS PATH !!!
PARQUET_ROOT_DIR="/work3/s233249/ImgiNav/datasets/scenes" 

# --- Input Taxonomy and Output Z-Map
CONFIG_DIR="${BASE_DIR}/config"
TAXONOMY_FILE="${CONFIG_DIR}/taxonomy.json"
OUTPUT_ZMAP_FILE="${CONFIG_DIR}/zmap.json"

# =============================================================================
# MODULES
# =============================================================================
# Load modules required by the conda environment
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# CONDA ENV
# =============================================================================
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

# =============================================================================
# RUN
# =============================================================================
echo "=========================================="
echo "Running Z-Map Creation"
echo "Taxonomy: ${TAXONOMY_FILE}"
echo "Source: ${PARQUET_ROOT_DIR}"
echo "Output: ${OUTPUT_ZMAP_FILE}"
echo "Start: $(date)"
echo "=========================================="

python "${PYTHON_SCRIPT}" create-zmap \
    --taxonomy "${TAXONOMY_FILE}" \
    --parquets "${PARQUET_ROOT_DIR}/**/*.parquet" \
    --output "${OUTPUT_ZMAP_FILE}" \
    --hmin 0.1 \
    --hmax 1.8

echo "=========================================="
echo "Z-Map creation COMPLETE"
echo "End: $(date)"
echo "=========================================="