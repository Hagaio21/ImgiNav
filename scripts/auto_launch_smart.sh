#!/bin/bash
# Smart auto-launch: Update unfinished list, then launch prioritizing almost-done experiments

# Default values
BASE_DIR="${1:-/work3/s233249/ImgiNav/experiments/clip}"
REPO_DIR="${2:-/work3/s233249/ImgiNav/ImgiNav}"
MIN_COMPLETION="${3:-90}"  # Launch experiments that are at least 90% complete
MAX_JOBS="${4:-5}"  # Launch max 5 at a time

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================================================"
echo "Smart Auto-Launch System"
echo "================================================================================"
echo "This script will:"
echo "  1. Update unfinished.txt and statistics.txt files"
echo "  2. Launch experiments that are ${MIN_COMPLETION}%+ complete"
echo "  3. Limit to ${MAX_JOBS} jobs at a time"
echo ""
echo "Base directory: ${BASE_DIR}"
echo "Repository directory: ${REPO_DIR}"
echo ""

# Step 1: Update unfinished list and registry
echo "Step 1: Updating unfinished experiments list and registry..."
echo ""
bash "${SCRIPT_DIR}/update_unfinished_list.sh" "${BASE_DIR}" 1000 "${REPO_DIR}"

if [ $? -ne 0 ]; then
    echo "Error updating unfinished list. Aborting."
    exit 1
fi

echo ""
echo "Registry updated. Current experiment status:"
REGISTRY_FILE="${BASE_DIR}/experiment_registry.txt"
if [ -f "${REGISTRY_FILE}" ]; then
    echo "  Running:   $(grep -c "|running|" "${REGISTRY_FILE}" 2>/dev/null || echo "0")"
    echo "  Pending:   $(grep -c "|pending|" "${REGISTRY_FILE}" 2>/dev/null || echo "0")"
    echo "  Unfinished: $(grep -c "|unfinished|" "${REGISTRY_FILE}" 2>/dev/null || echo "0")"
    echo "  Not started: $(grep -c "|not_started|" "${REGISTRY_FILE}" 2>/dev/null || echo "0")"
fi

echo ""
echo "Step 2: Launching high-priority experiments (${MIN_COMPLETION}%+ complete)..."
echo ""

# Step 2: Launch from unfinished.txt
bash "${SCRIPT_DIR}/launch_from_unfinished.sh" "${BASE_DIR}" "${REPO_DIR}" "${MIN_COMPLETION}" "${MAX_JOBS}" false

echo ""
echo "Done! Check unfinished.txt for remaining experiments."
echo "Run this script again later to launch more as they progress."

