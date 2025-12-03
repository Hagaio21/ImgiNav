#!/bin/bash
# Launch Stage 4 v2: POV rendering + layout rotation + graph generation
# ======================================================================
# This script:
# 1. Discovers all room metadata files
# 2. Creates temporary shards
# 3. Submits array job
#
# Usage:
#   ./launch_stage4_v2.sh /path/to/dataset [num_shards] [extra_args...]
#
# Examples:
#   ./launch_stage4_v2.sh /data/structured3d 500
#   ./launch_stage4_v2.sh /data/structured3d 500 --only-graphs
#   ./launch_stage4_v2.sh /data/structured3d 100 --no-layouts

set -e

# Parse arguments
DATASET_ROOT="${1:?Usage: $0 <dataset_root> [num_shards] [extra_args...]}"
NUM_SHARDS="${2:-500}"
shift 2 || shift 1 || true
EXTRA_ARGS="$@"

# Paths
SCRIPTS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
METADATA_DIR="${DATASET_ROOT}/metadata/rooms"
SHARDS_DIR="${DATASET_ROOT}/shards_stage4v2"
LOGS_DIR="${DATASET_ROOT}/logs/stage4v2"

# Validate
if [[ ! -d "$METADATA_DIR" ]]; then
    echo "ERROR: Room metadata directory not found: $METADATA_DIR"
    exit 1
fi

# Create directories
mkdir -p "$SHARDS_DIR" "$LOGS_DIR"

# Clean old shards
rm -f "${SHARDS_DIR}"/shard_*.txt

echo "========================================"
echo "Stage 4 v2 Launch"
echo "========================================"
echo "Dataset:     $DATASET_ROOT"
echo "Scripts:     $SCRIPTS_DIR"
echo "Shards:      $NUM_SHARDS"
echo "Extra args:  $EXTRA_ARGS"
echo "========================================"

# Step 1: Discover room metadata files
echo "Discovering room metadata files..."
ROOM_FILES=$(find "$METADATA_DIR" -name "*.json" -type f | sort)
TOTAL_ROOMS=$(echo "$ROOM_FILES" | wc -l)

if [[ $TOTAL_ROOMS -eq 0 ]]; then
    echo "ERROR: No room metadata files found in $METADATA_DIR"
    exit 1
fi

echo "Found $TOTAL_ROOMS room metadata files"

# Step 2: Create shards
echo "Creating $NUM_SHARDS shards..."

# Calculate rooms per shard
ROOMS_PER_SHARD=$(( (TOTAL_ROOMS + NUM_SHARDS - 1) / NUM_SHARDS ))

# Split into shards
echo "$ROOM_FILES" | split -l "$ROOMS_PER_SHARD" -d -a 4 - "${SHARDS_DIR}/shard_"

# Rename to .txt
for f in "${SHARDS_DIR}"/shard_*; do
    if [[ ! "$f" =~ \.txt$ ]]; then
        mv "$f" "${f}.txt"
    fi
done

# Count actual shards created
ACTUAL_SHARDS=$(ls "${SHARDS_DIR}"/shard_*.txt 2>/dev/null | wc -l)
echo "Created $ACTUAL_SHARDS shards (~$ROOMS_PER_SHARD rooms each)"

# Step 3: Create job script
JOB_SCRIPT="${SHARDS_DIR}/job_stage4v2.sh"

cat > "$JOB_SCRIPT" << 'JOBSCRIPT'
#!/bin/bash
#BSUB -J stage4v2[1-NUM_SHARDS_PLACEHOLDER]
#BSUB -o LOGS_DIR_PLACEHOLDER/stage4v2_%J_%I.out
#BSUB -e LOGS_DIR_PLACEHOLDER/stage4v2_%J_%I.err
#BSUB -q hpc
#BSUB -W 4:00
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# ==============================================================================
# HEADLESS RENDERING SETUP (must be before Python imports OpenGL)
# ==============================================================================
# Fix XDG_RUNTIME_DIR error
export XDG_RUNTIME_DIR="${HOME}/.cache/xdg-runtime-$$"
mkdir -p "${XDG_RUNTIME_DIR}"

# Use OSMesa (software rendering) - most reliable on CPU nodes
export PYOPENGL_PLATFORM=osmesa

# Mesa software rendering settings
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATASET_ROOT="DATASET_ROOT_PLACEHOLDER"
SCRIPTS_DIR="SCRIPTS_DIR_PLACEHOLDER"
SHARDS_DIR="SHARDS_DIR_PLACEHOLDER"
EXTRA_ARGS="EXTRA_ARGS_PLACEHOLDER"

# Get shard for this task (LSB_JOBINDEX is 1-based)
SHARD_ID=$(printf "%04d" $((LSB_JOBINDEX - 1)))
SHARD_FILE="${SHARDS_DIR}/shard_${SHARD_ID}.txt"

if [[ ! -f "$SHARD_FILE" ]]; then
    echo "Shard file not found: $SHARD_FILE (OK if fewer shards exist)"
    rm -rf "${XDG_RUNTIME_DIR}" 2>/dev/null || true
    exit 0
fi

ROOM_COUNT=$(wc -l < "$SHARD_FILE")
echo "=========================================="
echo "Stage 4 v2: Shard ${SHARD_ID}"
echo "=========================================="
echo "Dataset: ${DATASET_ROOT}"
echo "Rooms: ${ROOM_COUNT}"
echo "Extra args: ${EXTRA_ARGS}"
echo "PYOPENGL_PLATFORM: ${PYOPENGL_PLATFORM}"
echo "=========================================="

# ==============================================================================
# CONDA ACTIVATION
# ==============================================================================
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || { echo "Failed to activate conda" >&2; exit 1; }
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate imginav || { echo "Failed to activate conda" >&2; exit 1; }
fi

# ==============================================================================
# RUN STAGE 4 V2
# ==============================================================================
cd "${SCRIPTS_DIR}"
export PYTHONPATH="${SCRIPTS_DIR}:${PYTHONPATH:-}"

python "${SCRIPTS_DIR}/stage4_render_povs_v2.py" \
    --dataset-root "${DATASET_ROOT}" \
    --room-list "${SHARD_FILE}" \
    --shard-id "${SHARD_ID}" \
    --hpc \
    --backend osmesa \
    --skip-existing \
    ${EXTRA_ARGS}

EXIT_CODE=$?

# Cleanup runtime dir
rm -rf "${XDG_RUNTIME_DIR}" 2>/dev/null || true

echo "=========================================="
echo "Stage 4 v2 shard ${SHARD_ID} completed with exit code: ${EXIT_CODE}"
echo "Output: povs/pov_info_shard_${SHARD_ID}.json"
echo "=========================================="

exit ${EXIT_CODE}
JOBSCRIPT

# Replace placeholders
sed -i "s|NUM_SHARDS_PLACEHOLDER|${ACTUAL_SHARDS}|g" "$JOB_SCRIPT"
sed -i "s|LOGS_DIR_PLACEHOLDER|${LOGS_DIR}|g" "$JOB_SCRIPT"
sed -i "s|DATASET_ROOT_PLACEHOLDER|${DATASET_ROOT}|g" "$JOB_SCRIPT"
sed -i "s|SCRIPTS_DIR_PLACEHOLDER|${SCRIPTS_DIR}|g" "$JOB_SCRIPT"
sed -i "s|SHARDS_DIR_PLACEHOLDER|${SHARDS_DIR}|g" "$JOB_SCRIPT"
sed -i "s|EXTRA_ARGS_PLACEHOLDER|${EXTRA_ARGS}|g" "$JOB_SCRIPT"

echo ""
echo "Job script created: $JOB_SCRIPT"
echo ""

# Step 4: Submit
echo "Submitting array job..."
bsub < "$JOB_SCRIPT"

echo ""
echo "========================================"
echo "Submitted! Monitor with:"
echo "  bjobs -w"
echo ""
echo "After completion, merge POV info:"
echo "  python ${SCRIPTS_DIR}/merge_pov_info_shards.py --dataset-root ${DATASET_ROOT}"
echo "========================================"
