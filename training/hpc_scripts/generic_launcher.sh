#!/bin/bash
# Generic launcher script for running Python scripts inside HPC jobs
# This script handles environment setup and executes the target Python script

set -euo pipefail

# Source environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../env_config.sh" 2>/dev/null || {
    # Fallback if env_config.sh not found
    BASE_DIR="${IMGINAV_ROOT:-/work3/s233249/ImgiNav}"
    export BASE_DIR
}

# Activate conda environment
# Try to detect conda environment from imginav_env.yml or use IMGINAV_ENV
if [ -n "${IMGINAV_ENV:-}" ]; then
    CONDA_ENV="${IMGINAV_ENV}"
elif [ -f "${BASE_DIR}/imginav_env.yml" ]; then
    # Try to extract environment name from yml file
    CONDA_ENV=$(grep -E "^name:" "${BASE_DIR}/imginav_env.yml" | head -1 | sed 's/name:[[:space:]]*//' | tr -d '"' | tr -d "'")
    if [ -z "${CONDA_ENV}" ]; then
        CONDA_ENV="imginav"  # Default fallback
    fi
else
    CONDA_ENV="imginav"  # Default fallback
fi

# Activate conda if available
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}" || {
        echo "WARNING: Failed to activate conda environment '${CONDA_ENV}'"
        echo "Continuing without conda activation..."
    }
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}" || {
        echo "WARNING: Failed to activate conda environment '${CONDA_ENV}'"
        echo "Continuing without conda activation..."
    }
fi

# Change to project directory
cd "${BASE_DIR}" || {
    echo "ERROR: Failed to change to BASE_DIR: ${BASE_DIR}"
    exit 1
}

# Execute the command passed as arguments
# All arguments after the script name are passed to the Python script
exec "$@"

