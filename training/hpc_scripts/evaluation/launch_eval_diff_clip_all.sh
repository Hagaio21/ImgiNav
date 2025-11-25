#!/bin/bash
# Launch script for all CLIP Diffusion evaluation jobs
# Submits all 6 evaluation jobs (small/medium/large x rooms/scenes)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"

EVAL_SCRIPTS=(
    "${SCRIPT_DIR}/eval_diff_clip_small_rooms.sh"
    "${SCRIPT_DIR}/eval_diff_clip_small_scenes.sh"
    "${SCRIPT_DIR}/eval_diff_clip_medium_rooms.sh"
    "${SCRIPT_DIR}/eval_diff_clip_medium_scenes.sh"
    "${SCRIPT_DIR}/eval_diff_clip_large_rooms.sh"
    "${SCRIPT_DIR}/eval_diff_clip_large_scenes.sh"
)

CONFIG_NAMES=(
    "small_rooms"
    "small_scenes"
    "medium_rooms"
    "medium_scenes"
    "large_rooms"
    "large_scenes"
)

echo "=============================================================================="
echo "Launching CLIP Diffusion Evaluation Jobs"
echo "=============================================================================="
echo "Submitting ${#EVAL_SCRIPTS[@]} evaluation jobs..."
echo ""
echo "Jobs to submit:"
for i in "${!EVAL_SCRIPTS[@]}"; do
    echo "  $((i+1)). ${CONFIG_NAMES[$i]}"
done
echo ""

# Create logs directory if it doesn't exist
mkdir -p "${BASE_DIR}/training/hpc_scripts/logs"

for i in "${!EVAL_SCRIPTS[@]}"; do
    script="${EVAL_SCRIPTS[$i]}"
    config_name="${CONFIG_NAMES[$i]}"
    
    if [ ! -f "${script}" ]; then
        echo "WARNING: Script not found: ${script} (will be skipped)"
        continue
    fi
    
    # Make script executable
    chmod +x "${script}"
    
    job_name="eval_clip_${config_name}"
    log_suffix="eval_clip_${config_name}"
    
    echo "[$((i+1))/${#EVAL_SCRIPTS[@]}] Submitting: ${config_name}"
    echo "  Script: ${script}"
    echo "  Job name: ${job_name}"
    
    bsub -J "${job_name}" \
        -o "${BASE_DIR}/training/hpc_scripts/logs/${log_suffix}.%J.out" \
        -e "${BASE_DIR}/training/hpc_scripts/logs/${log_suffix}.%J.err" \
        -n 4 \
        -R "rusage[mem=8000]" \
        -gpu "num=1" \
        -W 12:00 \
        -q gpul40s \
        bash "${script}"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Submitted successfully"
    else
        echo "  ✗ Failed to submit"
    fi
    
    sleep 1
    echo ""
done

echo "=============================================================================="
echo "Done! Submitted ${#EVAL_SCRIPTS[@]} evaluation jobs to gpul40s queue"
echo "=============================================================================="
echo ""
echo "Check job status with: bjobs"
echo "Check logs in: ${BASE_DIR}/training/hpc_scripts/logs/"
echo ""

