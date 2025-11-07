#!/bin/bash
# Launch complete baseline pipeline: prepare dataset -> fine-tune -> sample

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Launching Baseline Pipeline"
echo "Stable Diffusion Fine-tuning for Comparison"
echo "=========================================="
echo ""

echo "Step 1: Preparing layout dataset..."
bsub < "${SCRIPT_DIR}/run_prepare_layout_dataset.sh"
sleep 2

echo ""
echo "Step 2: Fine-tuning Stable Diffusion..."
echo "  (Wait for Step 1 to complete first, then run manually:)"
echo "  bsub < ${SCRIPT_DIR}/run_finetune_sd.sh"
echo ""

echo "Step 3: Sampling from fine-tuned model..."
echo "  (Wait for Step 2 to complete first, then run manually:)"
echo "  bsub < ${SCRIPT_DIR}/run_sample_finetuned_sd.sh"
echo ""

echo "=========================================="
echo "Pipeline Launched!"
echo "=========================================="
echo ""
echo "Workflow:"
echo "  1. Dataset preparation (Step 1) - Running now"
echo "  2. Fine-tuning SD (Step 2) - Run after Step 1 completes"
echo "  3. Sampling (Step 3) - Run after Step 2 completes"
echo ""
echo "Monitor with: bjobs"
echo "Check logs: ${SCRIPT_DIR}/logs/*.out"
echo ""

