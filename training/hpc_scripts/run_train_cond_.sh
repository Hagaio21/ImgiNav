#!/bin/bash
#BSUB -J cond_diffusion_mixer
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/cond_diffusion_mixer.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/cond_diffusion_mixer.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=32000]"
#BSUB -gpu "num=1"
#BSUB -W 24:00
#BSUB -q gpul40s

# --- Strict Mode ---
# Exit immediately if a command exits with a non-zero status.
# Treat unset variables as an error.
# Pipeline errors are propagated.
set -euo pipefail

# =============================================================================
# Configuration Variables
# =============================================================================
## --- Project Paths ---
readonly BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
readonly PYTHON_SCRIPT="${BASE_DIR}/training/train_conditioned_diffusion.py"

## --- Experiment Setup ---
readonly EXP_CONFIG="${BASE_DIR}/config/experiments/cond_exp_small_200_2.yml"
readonly ROOM_MANIFEST="/work3/s233249/ImgiNav/datasets/room_dataset_with_emb.csv"
readonly SCENE_MANIFEST="/work3/s233249/ImgiNav/datasets/scene_dataset_with_emb.csv"

## --- Model & Training Options ---
# Choose mixer type: "LinearConcat" or "NonLinearConcat"
readonly MIXER_TYPE="NonLinearConcat"

# POV Mode: "seg", "tex", or "" (empty string) for all types
readonly POV_MODE="seg"

# Resume training: "true" or "false"
readonly RESUME_JOB="false" # Changed from commenting/uncommenting

# =============================================================================
# Environment Setup
# =============================================================================
echo "--- Loading Modules ---"
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64
echo "Modules loaded."

echo "--- Activating Conda Environment ---"
# N.B.: Using $HOME here assumes miniconda3 is in the user's home directory.
# Adjust if your miniconda installation path is different.
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    # Try primary environment
    if conda activate imginav; then
        echo "Activated conda environment 'imginav'."
    # Try fallback environment
    elif conda activate scenefactor; then
        echo "Activated fallback conda environment 'scenefactor'."
    # Fail if neither works
    else
        echo "ERROR: Failed to activate 'imginav' or 'scenefactor' conda environment." >&2
        exit 1
    fi
else
    echo "WARNING: Conda initialization script not found at '$HOME/miniconda3/etc/profile.d/conda.sh'." >&2
    echo "         Attempting to run Python directly, assuming environment is already sourced." >&2
fi

# =============================================================================
# Run Training Script
# =============================================================================
echo "=========================================="
echo "Starting Conditioned Diffusion Training"
echo "  Config:        ${EXP_CONFIG}"
echo "  Room Manifest: ${ROOM_MANIFEST}"
echo "  Scene Manifest:${SCENE_MANIFEST}"
echo "  Mixer Type:    ${MIXER_TYPE}"
echo "  POV Mode:      ${POV_MODE:-all}" # Print "all" if POV_MODE is empty
echo "  Resume:        ${RESUME_JOB}"
echo "  Start Time:    $(date)"
echo "=========================================="

# Set PYTHONPATH to include the base directory for module imports
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"

# Change to the base directory to ensure relative paths in Python script work
cd "${BASE_DIR}"

# Build the command arguments array
# N.B.: Using an array handles spaces in paths correctly.
CMD_ARGS=(
    --exp_config "${EXP_CONFIG}"
    --room_manifest "${ROOM_MANIFEST}"
    --scene_manifest "${SCENE_MANIFEST}"
    --mixer_type "${MIXER_TYPE}"
)

# Add optional arguments
if [[ -n "$POV_MODE" ]]; then
    CMD_ARGS+=(--pov_type "${POV_MODE}")
fi

if [[ "$RESUME_JOB" == "true" ]]; then
    CMD_ARGS+=(--resume)
fi

# Execute the Python script
echo "--- Running Python Script ---"
python "${PYTHON_SCRIPT}" "${CMD_ARGS[@]}"
EXIT_CODE=$? # Capture the exit code immediately

# =============================================================================
# Final Output & Cleanup
# =============================================================================
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training COMPLETE"
else
    echo "❌ Training FAILED with exit code $EXIT_CODE"
fi
echo "  End Time:      $(date)"
echo "=========================================="

# --- Display Output Paths (Best Effort) ---
# N.B.: Parsing YAML with grep/awk is fragile. It's better if the Python script
#       prints the definitive experiment directory path at the end.
echo ""
echo "--- Locating Experiment Directory (Best Effort) ---"
EXP_DIR_GUESS=$(grep -A 1 "^experiment:" "${EXP_CONFIG}" | grep "exp_dir:" | awk '{print $2}')

if [[ -n "$EXP_DIR_GUESS" ]] && [[ -d "$EXP_DIR_GUESS" ]]; then
    echo "Experiment output likely saved to: ${EXP_DIR_GUESS}"
    echo "  - Checkpoints: ${EXP_DIR_GUESS}/checkpoints/"
    echo "  - Samples:     ${EXP_DIR_GUESS}/samples/"
    echo "  - Logs:        ${EXP_DIR_GUESS}/logs/"
else
    echo "Could not reliably determine experiment directory from config file."
    echo "Please check the Python script output or the config file manually."
fi

exit $EXIT_CODE