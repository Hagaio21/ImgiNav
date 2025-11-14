#!/bin/bash
# Phase 2.1 Pipeline Launcher
# Runs 3 pipelines for different UNet sizes: 32, 64, 128
# Each pipeline: Autoencoder training -> Embedding -> Diffusion training

set -e  # Exit on error

# Base paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AE_CONFIG="${PROJECT_ROOT}/experiments/autoencoders/phase2/phase2_1_AE_64x64_structural.yaml"

# Diffusion configs
DIFFUSION_CONFIG_32="${PROJECT_ROOT}/experiments/diffusion/phase2/phase2_1_diffusion_64x64_bottleneck_attn_unet32.yaml"
DIFFUSION_CONFIG_64="${PROJECT_ROOT}/experiments/diffusion/phase2/phase2_1_diffusion_64x64_bottleneck_attn_unet64.yaml"
DIFFUSION_CONFIG_128="${PROJECT_ROOT}/experiments/diffusion/phase2/phase2_1_diffusion_64x64_bottleneck_attn_unet128.yaml"

# Change to project root
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Phase 2.1 Pipeline Launcher"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo "Autoencoder config: $AE_CONFIG"
echo ""
echo "Diffusion configs:"
echo "  - UNet32: $DIFFUSION_CONFIG_32"
echo "  - UNet64: $DIFFUSION_CONFIG_64"
echo "  - UNet128: $DIFFUSION_CONFIG_128"
echo "=========================================="
echo ""

# Check configs exist
if [ ! -f "$AE_CONFIG" ]; then
    echo "ERROR: Autoencoder config not found: $AE_CONFIG"
    exit 1
fi

for config in "$DIFFUSION_CONFIG_32" "$DIFFUSION_CONFIG_64" "$DIFFUSION_CONFIG_128"; do
    if [ ! -f "$config" ]; then
        echo "ERROR: Diffusion config not found: $config"
        exit 1
    fi
done

# Function to run a single pipeline
run_pipeline() {
    local diffusion_config=$1
    local config_name=$(basename "$diffusion_config" .yaml)
    
    echo ""
    echo "=========================================="
    echo "Starting pipeline: $config_name"
    echo "=========================================="
    echo "Autoencoder config: $AE_CONFIG"
    echo "Diffusion config: $diffusion_config"
    echo "=========================================="
    echo ""
    
    python training/train_pipeline_phase2.py \
        --ae-config "$AE_CONFIG" \
        --diffusion-config "$diffusion_config"
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Pipeline failed for $config_name"
        exit 1
    fi
    
    echo ""
    echo "=========================================="
    echo "Pipeline completed: $config_name"
    echo "=========================================="
    echo ""
}

# Run pipelines sequentially
# Note: Autoencoder training happens only once (first pipeline)
# Subsequent pipelines will reuse the same autoencoder checkpoint

echo "Starting UNet32 pipeline..."
run_pipeline "$DIFFUSION_CONFIG_32"

echo "Starting UNet64 pipeline..."
run_pipeline "$DIFFUSION_CONFIG_64"

echo "Starting UNet128 pipeline..."
run_pipeline "$DIFFUSION_CONFIG_128"

echo ""
echo "=========================================="
echo "All pipelines completed successfully!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - UNet32: experiments/phase2/phase2_1_diffusion_64x64_bottleneck_attn_unet32/"
echo "  - UNet64: experiments/phase2/phase2_1_diffusion_64x64_bottleneck_attn_unet64/"
echo "  - UNet128: experiments/phase2/phase2_1_diffusion_64x64_bottleneck_attn_unet128/"
echo "=========================================="

