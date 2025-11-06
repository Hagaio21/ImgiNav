#!/bin/bash
#BSUB -J controlnet_manifest
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs/controlnet_manifest.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs/controlnet_manifest.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -W 04:00
#BSUB -q gpuv100

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SCRIPT_PATH="/work3/s233249/ImgiNav/ImgiNav/data_preparation/create_controlnet_manifest.py"

# Input manifests
LAYOUTS_MANIFEST="/work3/s233249/ImgiNav/datasets/layouts.csv"
POV_EMBEDDINGS_MANIFEST="/work3/s233249/ImgiNav/datasets/povs_with_embeddings.csv"
GRAPH_EMBEDDINGS_MANIFEST="/work3/s233249/ImgiNav/datasets/graphs_with_embeddings.csv"

# Output manifest
OUTPUT_MANIFEST="/work3/s233249/ImgiNav/datasets/controlnet_training_manifest.csv"

# Optional: Create layout embeddings if missing
# Set to "true" to create layout embeddings, "false" to skip
CREATE_LAYOUT_EMBEDDINGS="false"

# Autoencoder config (required if CREATE_LAYOUT_EMBEDDINGS=true)
AUTOENCODER_CONFIG="/work3/s233249/ImgiNav/experiments/autoencoders/phase1/config.yaml"
AUTOENCODER_CHECKPOINT="/work3/s233249/ImgiNav/checkpoints/autoencoder.pt"

# Embedding creation parameters (if creating embeddings)
BATCH_SIZE=32
NUM_WORKERS=8
DEVICE="cuda"

echo "=============================================================="
echo " Creating ControlNet Training Manifest"
echo "=============================================================="
echo " Layouts manifest:        ${LAYOUTS_MANIFEST}"
echo " POV embeddings manifest:  ${POV_EMBEDDINGS_MANIFEST}"
echo " Graph embeddings manifest: ${GRAPH_EMBEDDINGS_MANIFEST}"
echo " Output manifest:          ${OUTPUT_MANIFEST}"
echo " Create layout embeddings: ${CREATE_LAYOUT_EMBEDDINGS}"
echo "=============================================================="

# ----------------------------------------------------------------------
# Conda environment
# ----------------------------------------------------------------------
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || true
fi

# ----------------------------------------------------------------------
# Run script
# ----------------------------------------------------------------------
if [ "${CREATE_LAYOUT_EMBEDDINGS}" = "true" ]; then
    echo "[INFO] Creating manifest with layout embedding creation enabled"
    python "${SCRIPT_PATH}" \
        --layouts-manifest "${LAYOUTS_MANIFEST}" \
        --pov-embeddings-manifest "${POV_EMBEDDINGS_MANIFEST}" \
        --graph-embeddings-manifest "${GRAPH_EMBEDDINGS_MANIFEST}" \
        --output "${OUTPUT_MANIFEST}" \
        --create-layout-embeddings \
        --autoencoder-config "${AUTOENCODER_CONFIG}" \
        --autoencoder-checkpoint "${AUTOENCODER_CHECKPOINT}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --device "${DEVICE}"
else
    echo "[INFO] Creating manifest using existing embeddings"
    python "${SCRIPT_PATH}" \
        --layouts-manifest "${LAYOUTS_MANIFEST}" \
        --pov-embeddings-manifest "${POV_EMBEDDINGS_MANIFEST}" \
        --graph-embeddings-manifest "${GRAPH_EMBEDDINGS_MANIFEST}" \
        --output "${OUTPUT_MANIFEST}"
fi

echo "=============================================================="
echo " ControlNet manifest creation completed successfully"
echo "=============================================================="

