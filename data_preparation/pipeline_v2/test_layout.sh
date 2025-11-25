#!/bin/bash
# Simple test script for layout rendering

SCENE_ID="002c110c-9bbc-4ab4-affa-4225fb127bad"
SCENE_JSON="${1:-test_dataset/${SCENE_ID}.json}"  # Default or first argument
FUTURE_ROOT="${2:-/path/to/3D-FUTURE-model}"  # Default or second argument
OUTPUT_DIR="test_dataset/test_output"
TAXONOMY="config/taxonomy.json"

echo "Testing layout rendering for scene: ${SCENE_ID}"
echo "Scene JSON: ${SCENE_JSON}"
echo "3D-FUTURE root: ${FUTURE_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

python data_preparation/pipeline_v2/render_layouts.py \
    --scene_json "${SCENE_JSON}" \
    --future_root "${FUTURE_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --taxonomy "${TAXONOMY}" \
    --seed 42

