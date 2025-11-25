#!/bin/bash
# Test script to run the full pipeline v2 on a single scene
# Discovers scenes and tests on the first one found
# Usage: ./test_pipeline.sh [optional_scene_id]

set -euo pipefail

# Configuration
SCENES_ROOT="/dtu/datasets2/ScanNet/FutureFront3D/3D-FUTUR_FRONT"
MODEL_DIR="/dtu/datasets2/ScanNet/FutureFront3D/3D-FUTURE-model"
TAXONOMY_FILE="/work3/s233249/ImgiNav/ImgiNav/config/taxonomy.json"
OUTPUT_DIR="/work3/s233249/ImgiNav/datasets/dataset_v2_test"
PROJECT_ROOT="/work3/s233249/ImgiNav"

echo "=========================================="
echo "Pipeline v2 Test - Scene Discovery"
echo "=========================================="
echo "Scenes root: ${SCENES_ROOT}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Discover scenes
if [ $# -ge 1 ]; then
    # User provided scene ID
    SCENE_ID=$1
    echo "Using provided scene ID: ${SCENE_ID}"
    SCENE_JSON=$(find "${SCENES_ROOT}" -type f -name "${SCENE_ID}.json" | head -1)
    
    if [ -z "${SCENE_JSON}" ]; then
        echo "ERROR: Scene JSON file not found for ${SCENE_ID}"
        echo "Searched in: ${SCENES_ROOT}"
        exit 1
    fi
else
    # Discover scenes automatically
    echo "Discovering scenes in ${SCENES_ROOT}..."
    if [ ! -d "${SCENES_ROOT}" ]; then
        echo "ERROR: SCENES_ROOT directory does not exist: ${SCENES_ROOT}" >&2
        exit 1
    fi
    
    # Find all JSON files
    SCENE_FILES=$(find "${SCENES_ROOT}" -type f -name '*.json' 2>/dev/null | sort | head -1)
    
    if [ -z "${SCENE_FILES}" ]; then
        echo "ERROR: No scene JSON files found in ${SCENES_ROOT}" >&2
        exit 1
    fi
    
    SCENE_JSON="${SCENE_FILES}"
    SCENE_ID=$(basename "${SCENE_JSON}" .json)
    echo "Found scene: ${SCENE_ID}"
fi

echo "Scene JSON: ${SCENE_JSON}"
echo "Scene ID: ${SCENE_ID}"
echo ""

# Change to project directory
cd "${PROJECT_ROOT}/ImgiNav" || {
    echo "ERROR: Failed to change to project directory"
    exit 1
}

# Step 1: Export geometry
echo "=========================================="
echo "Step 1: Exporting geometry"
echo "=========================================="
python data_preparation/pipeline_v2/export_geometry.py \
    --scene_json "${SCENE_JSON}" \
    --future_root "${MODEL_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --taxonomy "${TAXONOMY_FILE}" || {
    echo "ERROR: Geometry export failed"
    exit 1
}

echo ""

# Steps 2-4: Run in parallel (layouts, POVs, graphs)
echo "=========================================="
echo "Steps 2-4: Running layouts, POVs, and graphs in parallel"
echo "=========================================="

# Create log files for each parallel process
LOG_DIR="${OUTPUT_DIR}/test_logs"
mkdir -p "${LOG_DIR}"

LAYOUT_LOG="${LOG_DIR}/${SCENE_ID}_layouts.log"
POV_LOG="${LOG_DIR}/${SCENE_ID}_povs.log"
GRAPH_LOG="${LOG_DIR}/${SCENE_ID}_graphs.log"

# Start all three processes in background
echo "Starting layout rendering..."
python data_preparation/pipeline_v2/render_layouts.py \
    --scene_id "${SCENE_ID}" \
    --output_dir "${OUTPUT_DIR}" \
    --taxonomy "${TAXONOMY_FILE}" \
    --seed 42 > "${LAYOUT_LOG}" 2>&1 &
LAYOUT_PID=$!

echo "Starting POV rendering..."
python data_preparation/pipeline_v2/render_povs.py \
    --scene_id "${SCENE_ID}" \
    --output_dir "${OUTPUT_DIR}" \
    --taxonomy "${TAXONOMY_FILE}" \
    --num_povs 6 \
    --seed 42 > "${POV_LOG}" 2>&1 &
POV_PID=$!

echo "Starting graph building..."
python data_preparation/pipeline_v2/build_graphs_from_metadata.py \
    --scene_id "${SCENE_ID}" \
    --output_dir "${OUTPUT_DIR}" \
    --taxonomy "${TAXONOMY_FILE}" > "${GRAPH_LOG}" 2>&1 &
GRAPH_PID=$!

echo "All three processes started (PIDs: layouts=${LAYOUT_PID}, povs=${POV_PID}, graphs=${GRAPH_PID})"
echo "Waiting for all processes to complete..."

# Wait for all background processes
FAILED=0

wait ${LAYOUT_PID}
LAYOUT_EXIT=$?
if [ ${LAYOUT_EXIT} -ne 0 ]; then
    echo "ERROR: Layout rendering failed (exit code: ${LAYOUT_EXIT})"
    echo "Check log: ${LAYOUT_LOG}"
    FAILED=1
else
    echo "✓ Layout rendering completed"
fi

wait ${POV_PID}
POV_EXIT=$?
if [ ${POV_EXIT} -ne 0 ]; then
    echo "ERROR: POV rendering failed (exit code: ${POV_EXIT})"
    echo "Check log: ${POV_LOG}"
    FAILED=1
else
    echo "✓ POV rendering completed"
fi

wait ${GRAPH_PID}
GRAPH_EXIT=$?
if [ ${GRAPH_EXIT} -ne 0 ]; then
    echo "ERROR: Graph building failed (exit code: ${GRAPH_EXIT})"
    echo "Check log: ${GRAPH_LOG}"
    FAILED=1
else
    echo "✓ Graph building completed"
fi

echo ""

if [ ${FAILED} -eq 1 ]; then
    echo "=========================================="
    echo "Pipeline test completed with ERRORS"
    echo "=========================================="
    echo "Check logs in: ${LOG_DIR}"
    exit 1
fi

echo "=========================================="
echo "Pipeline test completed successfully!"
echo "=========================================="
echo "Scene ID: ${SCENE_ID}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
echo "  Geometry: ${OUTPUT_DIR}/geometry/${SCENE_ID}.obj"
echo "  Layouts: ${OUTPUT_DIR}/layouts/rgb/ and layouts/seg/"
echo "  POVs: ${OUTPUT_DIR}/povs/rgb/ and povs/seg/"
echo "  Graphs: ${OUTPUT_DIR}/graphs/"
echo ""
echo "Logs: ${LOG_DIR}"

