#!/bin/bash
# Launch Stage 5 v2: POV-normalized graph generation
# ====================================================
# This script:
# 1. Discovers all scene metadata files
# 2. Creates shards
# 3. Submits array job
#
# Usage:
#   ./launch_stage5_v2.sh /path/to/dataset [num_shards]
#
# Examples:
#   ./launch_stage5_v2.sh /work3/s233249/ImgiNav/dataset_v2 100

set -e

# Parse arguments
DATASET_ROOT="${1:?Usage: $0 <dataset_root> [num_shards]}"
NUM_SHARDS="${2:-100}"

# Paths
SCRIPTS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
METADATA_DIR="${DATASET_ROOT}/metadata/scenes"
SHARDS_DIR="${DATASET_ROOT}/shards_stage5v2"
LOGS_DIR="${DATASET_ROOT}/logs/stage5v2"

# Validate
if [[ ! -d "$METADATA_DIR" ]]; then
    echo "ERROR: Scene metadata directory not found: $METADATA_DIR"
    exit 1
fi

# Create directories
mkdir -p "$SHARDS_DIR" "$LOGS_DIR"

# Clean old shards
rm -f "${SHARDS_DIR}"/shard_*.txt

echo "========================================"
echo "Stage 5 v2 Launch"
echo "========================================"
echo "Dataset:     $DATASET_ROOT"
echo "Scripts:     $SCRIPTS_DIR"
echo "Shards:      $NUM_SHARDS"
echo "========================================"

# Step 1: Discover scene metadata files
echo "Discovering scene metadata files..."
SCENE_FILES=$(find "$METADATA_DIR" -name "*.json" -type f | sort)
TOTAL_SCENES=$(echo "$SCENE_FILES" | wc -l)

if [[ $TOTAL_SCENES -eq 0 ]]; then
    echo "ERROR: No scene metadata files found in $METADATA_DIR"
    exit 1
fi

echo "Found $TOTAL_SCENES scene metadata files"

# Step 2: Create shards (extract scene IDs from filenames)
echo "Creating $NUM_SHARDS shards..."

# Extract scene IDs from file paths
SCENE_IDS=$(echo "$SCENE_FILES" | xargs -n1 basename | sed 's/\.json$//' | sort)

# Calculate scenes per shard
SCENES_PER_SHARD=$(( (TOTAL_SCENES + NUM_SHARDS - 1) / NUM_SHARDS ))

# Split into shards
echo "$SCENE_IDS" | split -l "$SCENES_PER_SHARD" -d -a 4 - "${SHARDS_DIR}/shard_"

# Rename to .txt
for f in "${SHARDS_DIR}"/shard_*; do
    if [[ ! "$f" =~ \.txt$ ]]; then
        mv "$f" "${f}.txt"
    fi
done

# Count actual shards created
ACTUAL_SHARDS=$(ls "${SHARDS_DIR}"/shard_*.txt 2>/dev/null | wc -l)
echo "Created $ACTUAL_SHARDS shards (~$SCENES_PER_SHARD scenes each)"

# Step 3: Create job script
JOB_SCRIPT="${SHARDS_DIR}/job_stage5v2.sh"

cat > "$JOB_SCRIPT" << 'JOBSCRIPT'
#!/bin/bash
#BSUB -J stage5v2[1-NUM_SHARDS_PLACEHOLDER]
#BSUB -o LOGS_DIR_PLACEHOLDER/stage5v2_%J_%I.out
#BSUB -e LOGS_DIR_PLACEHOLDER/stage5v2_%J_%I.err
#BSUB -q hpc
#BSUB -W 4:00
#BSUB -n 2
#BSUB -R "rusage[mem=2000]"

# Configuration
DATASET_ROOT="DATASET_ROOT_PLACEHOLDER"
SCRIPTS_DIR="SCRIPTS_DIR_PLACEHOLDER"
SHARDS_DIR="SHARDS_DIR_PLACEHOLDER"

# Get shard for this task (LSB_JOBINDEX is 1-based)
SHARD_ID=$(printf "%04d" $((LSB_JOBINDEX - 1)))
SHARD_FILE="${SHARDS_DIR}/shard_${SHARD_ID}.txt"

if [[ ! -f "$SHARD_FILE" ]]; then
    echo "Shard file not found: $SHARD_FILE (OK if fewer shards exist)"
    exit 0
fi

SCENE_COUNT=$(wc -l < "$SHARD_FILE")
echo "========================================"
echo "Stage 5 v2: Shard ${SHARD_ID}"
echo "Scenes: ${SCENE_COUNT}"
echo "========================================"

# Load modules
module load python/3.10 2>/dev/null || true

# Activate venv if exists
if [[ -f "${SCRIPTS_DIR}/venv/bin/activate" ]]; then
    source "${SCRIPTS_DIR}/venv/bin/activate"
fi

# Activate conda if available
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav 2>/dev/null || conda activate scenefactor 2>/dev/null || true
fi

# Run
cd "${SCRIPTS_DIR}"
export PYTHONPATH="${SCRIPTS_DIR}:${PYTHONPATH:-}"

python "${SCRIPTS_DIR}/stage5_build_graphs_v2.py" \
    --dataset-root "${DATASET_ROOT}" \
    --scene-list "${SHARD_FILE}" \
    --skip-existing

echo "Shard ${SHARD_ID} completed with exit code: $?"
JOBSCRIPT

# Replace placeholders
sed -i "s|NUM_SHARDS_PLACEHOLDER|${ACTUAL_SHARDS}|g" "$JOB_SCRIPT"
sed -i "s|LOGS_DIR_PLACEHOLDER|${LOGS_DIR}|g" "$JOB_SCRIPT"
sed -i "s|DATASET_ROOT_PLACEHOLDER|${DATASET_ROOT}|g" "$JOB_SCRIPT"
sed -i "s|SCRIPTS_DIR_PLACEHOLDER|${SCRIPTS_DIR}|g" "$JOB_SCRIPT"
sed -i "s|SHARDS_DIR_PLACEHOLDER|${SHARDS_DIR}|g" "$JOB_SCRIPT"

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
echo "After completion, collect manifest:"
echo "  bsub < ${SCRIPTS_DIR}/hpc_scripts/run_collect_manifest_pov_normalized.sh"
echo "========================================"

