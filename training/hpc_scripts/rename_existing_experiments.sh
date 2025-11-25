#!/bin/bash
#BSUB -J rename_experiments
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/rename_experiments.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/rename_experiments.%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=2000]"
#BSUB -W 1:00
#BSUB -q normal

set -euo pipefail

# Base directory where experiments are stored
BASE_DIR="/work3/s233249/ImgiNav/experiments/clip"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Renaming Existing Experiments"
echo "=========================================="
echo "Base directory: ${BASE_DIR}"
echo ""

# Check if base directory exists
if [ ! -d "${BASE_DIR}" ]; then
    echo -e "${RED}ERROR: Base directory does not exist: ${BASE_DIR}${NC}" >&2
    exit 1
fi

# Mapping of old names to new names
declare -A RENAME_MAP=(
    ["diff_clip_regular_medium_all"]="diff_clip_regular_both_medium_all"
    ["diff_clip_regular_rooms_large_bottleneck"]="diff_clip_regular_rooms_large_down_bottleneck"
    ["diff_clip_regular_rooms_medium_bottleneck"]="diff_clip_regular_rooms_medium_down_bottleneck"
    ["diff_clip_regular_rooms_small_bottleneck"]="diff_clip_regular_rooms_small_down_bottleneck"
    ["diff_clip_regular_scenes_large_bottleneck"]="diff_clip_regular_scenes_large_down_bottleneck"
    ["diff_clip_regular_scenes_medium_bottleneck"]="diff_clip_regular_scenes_medium_down_bottleneck"
    ["diff_clip_regular_scenes_small_bottleneck"]="diff_clip_regular_scenes_small_down_bottleneck"
    ["diff_clip_regular_small_all"]="diff_clip_regular_both_small_all"
)

renamed_count=0
file_renamed_count=0

# Function to rename files inside a directory
rename_files_in_dir() {
    local dir="$1"
    local old_name="$2"
    local new_name="$3"
    
    # Find and rename checkpoint files
    if [ -d "${dir}/checkpoints" ]; then
        for checkpoint in "${dir}"/checkpoints/${old_name}_checkpoint_*.pt; do
            if [ -f "${checkpoint}" ]; then
                new_checkpoint=$(echo "${checkpoint}" | sed "s/${old_name}/${new_name}/g")
                echo -e "    Renaming checkpoint: $(basename ${checkpoint}) -> $(basename ${new_checkpoint})"
                mv "${checkpoint}" "${new_checkpoint}"
            fi
        done
    fi
    
    # Rename metrics CSV
    old_metrics="${dir}/${old_name}_metrics.csv"
    if [ -f "${old_metrics}" ]; then
        new_metrics="${dir}/${new_name}_metrics.csv"
        echo -e "    Renaming metrics: $(basename ${old_metrics}) -> $(basename ${new_metrics})"
        mv "${old_metrics}" "${new_metrics}"
    fi
    
    # Rename metric plot files
    for plot in "${dir}/${old_name}_metric_"*.png; do
        if [ -f "${plot}" ]; then
            new_plot=$(echo "${plot}" | sed "s/${old_name}/${new_name}/g")
            echo -e "    Renaming plot: $(basename ${plot}) -> $(basename ${new_plot})"
            mv "${plot}" "${new_plot}"
        fi
    done
    
    # Rename any other files with the old name pattern (maxdepth 1 to avoid subdirectories)
    while IFS= read -r file; do
        if [ -f "${file}" ]; then
            new_file=$(echo "${file}" | sed "s/${old_name}/${new_name}/g")
            if [ "${file}" != "${new_file}" ]; then
                echo -e "    Renaming file: $(basename ${file}) -> $(basename ${new_file})"
                mv "${file}" "${new_file}"
            fi
        fi
    done < <(find "${dir}" -maxdepth 1 -type f -name "*${old_name}*" 2>/dev/null || true)
    
    # Update any references in text files (configs, logs, etc.)
    echo -e "  Updating references in text files..."
    while IFS= read -r text_file; do
        if grep -q "${old_name}" "${text_file}" 2>/dev/null; then
            echo -e "    Updating references in: $(basename ${text_file})"
            sed -i "s/${old_name}/${new_name}/g" "${text_file}"
        fi
    done < <(find "${dir}" -type f \( -name "*.txt" -o -name "*.log" -o -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name "*.csv" \) 2>/dev/null || true)
}

# Process each experiment
for old_name in "${!RENAME_MAP[@]}"; do
    new_name="${RENAME_MAP[$old_name]}"
    
    old_dir="${BASE_DIR}/${old_name}"
    new_dir="${BASE_DIR}/${new_name}"
    
    # Check if old directory exists
    if [ -d "${old_dir}" ]; then
        echo -e "${YELLOW}Found: ${old_name}${NC}"
        
        # Check if new directory already exists
        if [ -d "${new_dir}" ]; then
            echo -e "${RED}  WARNING: Target directory already exists: ${new_name}${NC}"
            echo -e "${RED}  Skipping to avoid overwriting...${NC}"
            echo ""
            continue
        fi
        
        # Rename the directory
        echo -e "  Renaming directory: ${old_name} -> ${new_name}"
        mv "${old_dir}" "${new_dir}"
        renamed_count=$((renamed_count + 1))
        
        # Now rename files inside that reference the old experiment name
        if [ -d "${new_dir}" ]; then
            echo -e "  Renaming files inside directory..."
            rename_files_in_dir "${new_dir}" "${old_name}" "${new_name}"
            echo -e "${GREEN}  ✓ Completed: ${new_name}${NC}"
        fi
    else
        echo -e "  (Not found: ${old_name})"
    fi
    echo ""
done

echo "=========================================="
echo "Summary"
echo "=========================================="
echo -e "${GREEN}Directories renamed: ${renamed_count}${NC}"
echo ""
echo "Done!"

