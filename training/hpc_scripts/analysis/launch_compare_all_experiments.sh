#!/bin/bash
# Launch script to compare all experiment metrics

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
COMPARE_SCRIPT="${SCRIPT_DIR}/run_compare_all_experiments.sh"

echo "=============================================================================="
echo "Launching Experiment Comparison"
echo "=============================================================================="
echo ""
echo "This will:"
echo "  - Find all experiment metrics CSV files"
echo "  - Create comparison plots for train_loss and val_loss"
echo "  - Generate a summary table"
echo ""
echo "Results will be saved to:"
echo "  /work3/s233249/ImgiNav/experiments/clip/comparison_summary"
echo ""
echo "=============================================================================="

# Make script executable
chmod +x "${COMPARE_SCRIPT}"

# Submit job
bsub -J "compare_experiments" \
    -o "${BASE_DIR}/training/hpc_scripts/logs/compare_experiments.%J.out" \
    -e "${BASE_DIR}/training/hpc_scripts/logs/compare_experiments.%J.err" \
    -n 1 \
    -R "rusage[mem=4000]" \
    -W 2:00 \
    -q hpc \
    bash "${COMPARE_SCRIPT}"

echo ""
echo "Job submitted! Check logs with:"
echo "  bjobs"
echo "  tail -f ${BASE_DIR}/training/hpc_scripts/logs/compare_experiments.*.out"

