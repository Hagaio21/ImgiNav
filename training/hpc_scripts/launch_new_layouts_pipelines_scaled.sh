#!/bin/bash
# Launch script for New Layouts pipelines with Latent Scaling
# This script submits all new_layouts pipeline jobs with scale_factor to the HPC cluster
#
# Uses cleaned layouts manifest: /work3/s233249/ImgiNav/datasets/layouts_cleaned.csv
# All models use cfg_dropout_rate=1.0 (100% dropout = always unconditional training)
# scale_factor is automatically calculated from dataset latents during training

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# =============================================================================
# CONFIGURATION
# =============================================================================
PIPELINE_SCRIPTS=(
    "run_new_layouts_pipeline_unet48_256_unconditional_scaled.sh"
    "run_new_layouts_pipeline_unet64_256_unconditional_scaled.sh"
)

# =============================================================================
# MAIN
# =============================================================================
echo "=============================================================================="
echo "Launching New Layouts Pipeline Jobs (256x256 input, Unconditional, Scaled)"
echo "=============================================================================="
echo ""
echo "Script directory: ${SCRIPT_DIR}"
echo "Base directory: ${BASE_DIR}"
echo ""
echo "Jobs to launch:"
for script in "${PIPELINE_SCRIPTS[@]}"; do
    echo "  - ${script}"
done
echo ""
echo "Using cleaned layouts manifest: /work3/s233249/ImgiNav/datasets/layouts_cleaned.csv"
echo "=============================================================================="
echo ""

# Launch each pipeline job
JOB_IDS=()

for script in "${PIPELINE_SCRIPTS[@]}"; do
    script_path="${SCRIPT_DIR}/${script}"
    
    if [ ! -f "${script_path}" ]; then
        echo "ERROR: Script not found: ${script_path}" >&2
        exit 1
    fi
    
    echo "Submitting job: ${script}"
    OUTPUT=$(bsub < "${script_path}" 2>&1)
    
    if [ $? -eq 0 ]; then
        # Extract job ID from bsub output (format: "Job <12345> is submitted to queue <gpul40s>")
        JOB_ID=$(echo "${OUTPUT}" | grep -oP 'Job <\K[0-9]+')
        if [ -n "${JOB_ID}" ]; then
            JOB_IDS+=("${JOB_ID}")
            echo "  ✓ Job submitted successfully (Job ID: ${JOB_ID})"
        else
            echo "  ✓ Job submitted successfully (could not extract job ID)"
        fi
    else
        echo "  ✗ Failed to submit job: ${OUTPUT}" >&2
        exit 1
    fi
    echo ""
    sleep 2
done

# Summary
echo "=============================================================================="
echo "Summary"
echo "=============================================================================="
echo "Total jobs submitted: ${#JOB_IDS[@]}"
if [ ${#JOB_IDS[@]} -gt 0 ]; then
    echo "Job IDs: ${JOB_IDS[*]}"
    echo ""
    echo "To check job status, use:"
    echo "  bjobs ${JOB_IDS[0]}"
    echo ""
    echo "To check all your jobs:"
    echo "  bjobs"
    echo ""
    echo "To cancel a job:"
    echo "  bkill <job_id>"
fi
echo ""
echo "Pipeline breakdown:"
echo "  - UNet48_256_scaled: base_channels=48, batch_size=128, unconditional, with latent scaling"
echo "  - UNet64_256_scaled: base_channels=64, batch_size=96, unconditional, with latent scaling"
echo ""
echo "All pipelines use:"
echo "  - 1000 noise steps with LinearScheduler"
echo "  - cfg_dropout_rate=1.0 (100% unconditional training)"
echo "  - Same VAE autoencoder (trained once, reused)"
echo "  - 256×256 input resolution"
echo "  - Cleaned layouts (density >= 0.1, floor color present)"
echo "  - Latent clamping: [-6.0, 6.0]"
echo "  - AUTOMATIC LATENT SCALING: scale_factor calculated from ~100 random latents"
echo "    This normalizes VAE latents to unit variance for the diffusion scheduler"
echo "    Expected to fix 'neon' or static image issues caused by latent scaling mismatch"
echo "=============================================================================="

