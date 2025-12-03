#!/bin/bash
#BSUB -J stage5_pov_graphs[1-10]
#BSUB -o logs/stage5_pov_%J_%I.out
#BSUB -e logs/stage5_pov_%J_%I.err
#BSUB -q hpc
#BSUB -W 4:00
#BSUB -n 2
#BSUB -R "rusage[mem=2000]"

# Stage 5 v2: POV-Normalized Graph Generation
# Generates POV-specific graphs with:
# - Descriptive object naming (no numerical suffixes)
# - POV-relative spatial relations ("ahead", "to your left")
# - Second-person perspective descriptions

set -e

# Configuration
CONFIG_FILE="${CONFIG_FILE:-paths.yaml}"
SHARD_INDEX="${LSB_JOBINDEX:-1}"
SHARDS_DIR="${SHARDS_DIR:-shards}"
SCENE_LIST="${SHARDS_DIR}/shard_${SHARD_INDEX}.txt"

# Load configuration
if [ -f "$CONFIG_FILE" ]; then
    DATASET_ROOT=$(grep "output_dataset_root:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
else
    echo "ERROR: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

METADATA_DIR="${DATASET_ROOT}/metadata"
OUTPUT_DIR="${DATASET_ROOT}/graphs"
SCRIPTS_DIR="$(dirname "$0")/.."

echo "=============================================="
echo "Stage 5 v2: POV-Normalized Graph Generation"
echo "=============================================="
echo "Shard Index: ${SHARD_INDEX}"
echo "Scene List: ${SCENE_LIST}"
echo "Dataset Root: ${DATASET_ROOT}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "=============================================="

# Check scene list exists
if [ ! -f "$SCENE_LIST" ]; then
    echo "ERROR: Scene list not found: $SCENE_LIST"
    exit 1
fi

SCENE_COUNT=$(wc -l < "$SCENE_LIST")
echo "Processing ${SCENE_COUNT} scenes from ${SCENE_LIST}"

# Create output directories
mkdir -p "${OUTPUT_DIR}/jsons"
mkdir -p "${OUTPUT_DIR}/texts"
mkdir -p logs

# Run POV-normalized graph generation
python "${SCRIPTS_DIR}/stage5_build_graphs_v2.py" \
    --dataset-root "$DATASET_ROOT" \
    --scene-list "$SCENE_LIST" \
    --skip-existing

echo ""
echo "=============================================="
echo "Stage 5 v2 Complete"
echo "=============================================="

# Count outputs
SCENE_GRAPHS=$(find "${OUTPUT_DIR}/jsons" -name "*_scene_graph.json" 2>/dev/null | wc -l)
ROOM_GRAPHS=$(find "${OUTPUT_DIR}/jsons" -name "*_room_graph.json" 2>/dev/null | wc -l)
POV_GRAPHS=$(find "${OUTPUT_DIR}/jsons" -name "*door*_room_graph.json" -o -name "*window*_room_graph.json" 2>/dev/null | wc -l)

echo "Scene graphs: ${SCENE_GRAPHS}"
echo "Total room graphs: ${ROOM_GRAPHS}"
echo "POV-specific graphs: ${POV_GRAPHS}"
