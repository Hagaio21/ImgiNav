#!/bin/bash
# Convenience wrapper that automatically launches unfinished experiments
# This is the main script you should use

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
BASE_DIR="${1:-/work3/s233249/ImgiNav/experiments/clip}"
REPO_DIR="${2:-/work3/s233249/ImgiNav/ImgiNav}"

echo "================================================================================"
echo "Auto-Launch Unfinished Experiments"
echo "================================================================================"
echo ""
echo "This script will:"
echo "  1. Scan for unfinished experiments"
echo "  2. Find their config files"
echo "  3. Launch them automatically"
echo ""
echo "Base directory: ${BASE_DIR}"
echo "Repository directory: ${REPO_DIR}"
echo ""

# First, show what will be launched
echo "Step 1: Scanning for unfinished experiments..."
echo ""

bash "${SCRIPT_DIR}/list_unfinished_experiments.sh" "${BASE_DIR}" 1000 | tail -n +10

echo ""
read -p "Do you want to launch these experiments? (yes/no): " confirm

if [ "${confirm}" != "yes" ] && [ "${confirm}" != "y" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Step 2: Launching experiments..."
echo ""

# Launch them
bash "${SCRIPT_DIR}/launch_unfinished_experiments.sh" "${BASE_DIR}" 1000 "${REPO_DIR}" false

