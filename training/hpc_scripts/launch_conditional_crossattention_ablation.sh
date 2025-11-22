#!/bin/bash
# Launch script for Conditional Cross-Attention Diffusion Ablation Experiments
# This script submits all ablation experiment jobs to the HPC cluster
#
# Ablation experiments test different attention placements:
# - downs only
# - bottleneck only
# - ups only
# - downs + bottleneck
# - downs + ups
# - bottleneck + ups
# - all (downs + bottleneck + ups)
#
# For both small (48 base_channels) and large (96 base_channels) models

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# =============================================================================
# CONFIGURATION
# =============================================================================
# Small model ablation experiments (48 base_channels, depth 3, 2 attention heads)
SMALL_EXPERIMENTS=(
    "conditional_crossattention_diffusion_downs"
    "conditional_crossattention_diffusion_bottleneck"
    "conditional_crossattention_diffusion_ups"
    "conditional_crossattention_diffusion_downs_bottleneck"
    "conditional_crossattention_diffusion_downs_ups"
    "conditional_crossattention_diffusion_bottleneck_ups"
    "conditional_crossattention_diffusion_all"
)

# Large model ablation experiments (96 base_channels, depth 4, 8 attention heads)
LARGE_EXPERIMENTS=(
    "conditional_crossattention_diffusion_large_downs"
    "conditional_crossattention_diffusion_large_bottleneck"
    "conditional_crossattention_diffusion_large_ups"
    "conditional_crossattention_diffusion_large_downs_bottleneck"
    "conditional_crossattention_diffusion_large_downs_ups"
    "conditional_crossattention_diffusion_large_bottleneck_ups"
    "conditional_crossattention_diffusion_large_all"
)

ABLATION_SCRIPT="${SCRIPT_DIR}/run_conditional_crossattention_pipeline_ablation.sh"

# =============================================================================
# VALIDATION
# =============================================================================
if [ ! -f "${ABLATION_SCRIPT}" ]; then
    echo "ERROR: Ablation script not found: ${ABLATION_SCRIPT}" >&2
    exit 1
fi

# Make script executable
chmod +x "${ABLATION_SCRIPT}"

# =============================================================================
# MAIN
# =============================================================================
echo "=============================================================================="
echo "Launching Conditional Cross-Attention Diffusion Ablation Experiments"
echo "=============================================================================="
echo ""
echo "Script directory: ${SCRIPT_DIR}"
echo "Base directory: ${BASE_DIR}"
echo ""
echo "Small Model Experiments (48 base_channels, depth 3, 2 attention heads):"
for exp in "${SMALL_EXPERIMENTS[@]}"; do
    echo "  - ${exp}"
done
echo ""
echo "Large Model Experiments (96 base_channels, depth 4, 8 attention heads):"
for exp in "${LARGE_EXPERIMENTS[@]}"; do
    echo "  - ${exp}"
done
echo ""
echo "Total experiments: $((${#SMALL_EXPERIMENTS[@]} + ${#LARGE_EXPERIMENTS[@]}))"
echo "=============================================================================="
echo ""

# Prompt for confirmation
read -p "Submit all ablation experiments? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# =============================================================================
# SUBMIT JOBS
# =============================================================================
JOB_IDS=()
SUBMITTED_COUNT=0
FAILED_COUNT=0

echo ""
echo "Submitting jobs..."
echo "=============================================================================="

# Submit small model experiments
echo ""
echo "Submitting SMALL model experiments..."
for exp in "${SMALL_EXPERIMENTS[@]}"; do
    echo -n "  Submitting ${exp}... "
    
    # Submit job and capture job ID
    JOB_OUTPUT=$(bsub -J "ablation_${exp}" \
        -o "${BASE_DIR}/training/hpc_scripts/logs/ablation_${exp}.%J.out" \
        -e "${BASE_DIR}/training/hpc_scripts/logs/ablation_${exp}.%J.err" \
        -n 8 \
        -R "rusage[mem=16000]" \
        -gpu "num=1" \
        -W 24:00 \
        -q gpuv100 \
        "${ABLATION_SCRIPT}" "${exp}" "small" 2>&1)
    
    if [ $? -eq 0 ]; then
        # Extract job ID from bsub output (format: "Job <12345> is submitted to queue <gpuv100>")
        JOB_ID=$(echo "${JOB_OUTPUT}" | grep -oP 'Job <\K[0-9]+(?=>)')
        if [ -n "${JOB_ID}" ]; then
            JOB_IDS+=("${JOB_ID}")
            echo "SUCCESS (Job ID: ${JOB_ID})"
            SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
        else
            echo "SUBMITTED (could not extract job ID)"
            SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
        fi
    else
        echo "FAILED"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
    
    # Small delay to avoid overwhelming the scheduler
    sleep 1
done

# Submit large model experiments
echo ""
echo "Submitting LARGE model experiments..."
for exp in "${LARGE_EXPERIMENTS[@]}"; do
    echo -n "  Submitting ${exp}... "
    
    # Submit job and capture job ID
    JOB_OUTPUT=$(bsub -J "ablation_${exp}" \
        -o "${BASE_DIR}/training/hpc_scripts/logs/ablation_${exp}.%J.out" \
        -e "${BASE_DIR}/training/hpc_scripts/logs/ablation_${exp}.%J.err" \
        -n 8 \
        -R "rusage[mem=24000]" \
        -gpu "num=1" \
        -W 24:00 \
        -q gpuv100 \
        "${ABLATION_SCRIPT}" "${exp}" "large" 2>&1)
    
    if [ $? -eq 0 ]; then
        # Extract job ID from bsub output
        JOB_ID=$(echo "${JOB_OUTPUT}" | grep -oP 'Job <\K[0-9]+(?=>)')
        if [ -n "${JOB_ID}" ]; then
            JOB_IDS+=("${JOB_ID}")
            echo "SUCCESS (Job ID: ${JOB_ID})"
            SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
        else
            echo "SUBMITTED (could not extract job ID)"
            SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
        fi
    else
        echo "FAILED"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
    
    # Small delay to avoid overwhelming the scheduler
    sleep 1
done

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=============================================================================="
echo "Submission Summary"
echo "=============================================================================="
echo "Successfully submitted: ${SUBMITTED_COUNT} jobs"
echo "Failed to submit: ${FAILED_COUNT} jobs"
echo ""

if [ ${#JOB_IDS[@]} -gt 0 ]; then
    echo "Job IDs:"
    for job_id in "${JOB_IDS[@]}"; do
        echo "  - ${job_id}"
    done
    echo ""
    echo "Monitor jobs with: bjobs"
    echo "Check logs in: ${BASE_DIR}/training/hpc_scripts/logs/"
fi

echo "=============================================================================="

