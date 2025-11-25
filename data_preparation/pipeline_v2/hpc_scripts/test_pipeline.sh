#!/bin/bash
#BSUB -J test_pipeline
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/pipeline_v2/hpc_scripts/logs/test_pipeline_%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/pipeline_v2/hpc_scripts/logs/test_pipeline_%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=8000]"
#BSUB -W 4:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# Configure for CPU-only software rendering (no GPU)
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330

# Pipeline v2: Test full pipeline on a single scene
# Discovers scenes and tests on the first one found

# =============================================================================
# CONFIGURATION - YOUR PATHS
# =============================================================================
SCENES_ROOT="/dtu/datasets2/ScanNet/FutureFront3D/3D-FUTUR_FRONT"
MODEL_DIR="/dtu/datasets2/ScanNet/FutureFront3D/3D-FUTURE-model"
TAXONOMY_FILE="/work3/s233249/ImgiNav/ImgiNav/config/taxonomy.json"
OUTPUT_DIR="/work3/s233249/ImgiNav/datasets/dataset_v2_test"
PROJECT_ROOT="/work3/s233249/ImgiNav"
# =============================================================================

# Create logs directory
mkdir -p "${PROJECT_ROOT}/ImgiNav/data_preparation/pipeline_v2/hpc_scripts/logs"

# Runtime fixes (like old pipeline)
export XDG_RUNTIME_DIR=/tmp/$USER
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

echo "=========================================="
echo "Pipeline v2 Test - Scene Discovery"
echo "=========================================="
echo "Scenes root: ${SCENES_ROOT}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Change to project directory
cd "${PROJECT_ROOT}/ImgiNav" || {
    echo "ERROR: Failed to change to project directory" >&2
    exit 1
}

# Discover scenes
echo "Discovering scenes in ${SCENES_ROOT}..."
if [ ! -d "${SCENES_ROOT}" ]; then
    echo "ERROR: SCENES_ROOT directory does not exist: ${SCENES_ROOT}" >&2
    exit 1
fi

# Find first JSON file (handle SIGPIPE from head gracefully)
# Use set +e temporarily around the pipe to avoid SIGPIPE errors
set +e
FIND_RESULT=$(find "${SCENES_ROOT}" -type f -name '*.json' 2>/dev/null)
FIND_EXIT=$?
set -e

if [ ${FIND_EXIT} -ne 0 ]; then
    echo "ERROR: find command failed with exit code ${FIND_EXIT}" >&2
    exit 1
fi

# Sort and get first result (handle pipe errors - head may cause SIGPIPE)
set +e
SCENE_JSON=$(echo "${FIND_RESULT}" | sort | head -1)
HEAD_EXIT=$?
set -e

# Exit code 141 is SIGPIPE from head, which is OK - we got the result
if [ ${HEAD_EXIT} -ne 0 ] && [ ${HEAD_EXIT} -ne 141 ]; then
    echo "WARNING: head command exited with code ${HEAD_EXIT}" >&2
fi

if [ -z "${SCENE_JSON}" ] || [ ! -f "${SCENE_JSON}" ]; then
    echo "ERROR: No scene JSON files found in ${SCENES_ROOT}" >&2
    exit 1
fi

if [ -z "${SCENE_JSON}" ]; then
    echo "ERROR: No scene JSON files found in ${SCENES_ROOT}" >&2
    exit 1
fi

SCENE_ID=$(basename "${SCENE_JSON}" .json)
echo "Found scene: ${SCENE_ID}"
echo "Scene JSON: ${SCENE_JSON}"
echo ""

# Robust conda activation
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || {
        echo "WARNING: Failed to activate imginav, trying scenefactor..." >&2
        conda activate scenefactor || {
            echo "WARNING: Failed to activate scenefactor environment" >&2
        }
    }
elif [ -x "$HOME/miniconda3/bin/conda" ]; then
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda activate imginav || {
        echo "WARNING: Failed to activate imginav, trying scenefactor..." >&2
        conda activate scenefactor || {
            echo "WARNING: Failed to activate scenefactor environment" >&2
        }
    }
fi

# Check dependencies
echo "Checking Python dependencies..."
python -c "import trimesh, open3d, numpy, PIL, json" || {
    echo "ERROR: Required Python packages not available" >&2
    exit 1
}
echo "All dependencies available"
echo ""

# Step 1: Export geometry
echo "=========================================="
echo "Step 1: Exporting geometry"
echo "=========================================="
python data_preparation/pipeline_v2/export_geometry.py \
    --scene_json "${SCENE_JSON}" \
    --future_root "${MODEL_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --taxonomy "${TAXONOMY_FILE}" || {
    echo "ERROR: Geometry export failed" >&2
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
    --seed 42 \
    --hpc > "${LAYOUT_LOG}" 2>&1 &
LAYOUT_PID=$!

echo "Starting POV rendering..."
python data_preparation/pipeline_v2/render_povs.py \
    --scene_id "${SCENE_ID}" \
    --output_dir "${OUTPUT_DIR}" \
    --taxonomy "${TAXONOMY_FILE}" \
    --num_povs 6 \
    --seed 42 \
    --hpc > "${POV_LOG}" 2>&1 &
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
    echo "ERROR: Layout rendering failed (exit code: ${LAYOUT_EXIT})" >&2
    echo "Check log: ${LAYOUT_LOG}" >&2
    tail -50 "${LAYOUT_LOG}" >&2
    FAILED=1
else
    echo "✓ Layout rendering completed"
fi

wait ${POV_PID}
POV_EXIT=$?
if [ ${POV_EXIT} -ne 0 ]; then
    echo "ERROR: POV rendering failed (exit code: ${POV_EXIT})" >&2
    echo "Check log: ${POV_LOG}" >&2
    tail -50 "${POV_LOG}" >&2
    FAILED=1
else
    echo "✓ POV rendering completed"
fi

wait ${GRAPH_PID}
GRAPH_EXIT=$?
if [ ${GRAPH_EXIT} -ne 0 ]; then
    echo "ERROR: Graph building failed (exit code: ${GRAPH_EXIT})" >&2
    echo "Check log: ${GRAPH_LOG}" >&2
    tail -50 "${GRAPH_LOG}" >&2
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

