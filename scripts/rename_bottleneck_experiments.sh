#!/bin/bash
# Script to rename experiment directories and files:
# 1. *_bottleneck -> *_down_bottleneck (for experiments that were trained with old naming)
# 2. diff_clip_(regular|spatial)_(size)_* -> diff_clip_(regular|spatial)_both_(size)_* (add missing "both")
# This fixes experiments that were trained with the old naming convention

set -e  # Exit on error

# Base directory where experiments are stored
BASE_DIR="/work3/s233249/ImgiNav/experiments/clip"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Renaming Experiments"
echo "=========================================="
echo "Base directory: ${BASE_DIR}"
echo "This script will:"
echo "  1. Rename *_bottleneck -> *_down_bottleneck"
echo "  2. Add '_both_' to experiments missing rooms/scenes/both"
echo ""

# Check if base directory exists
if [ ! -d "${BASE_DIR}" ]; then
    echo -e "${RED}ERROR: Base directory does not exist: ${BASE_DIR}${NC}" >&2
    exit 1
fi

# Patterns to rename: (variant)_(size)_bottleneck -> (variant)_(size)_down_bottleneck
# Variants: regular_rooms, regular_scenes, spatial_rooms, spatial_scenes
# Sizes: small, medium, large

VARIANTS=("regular_rooms" "regular_scenes" "spatial_rooms" "spatial_scenes")
SIZES=("small" "medium" "large")

renamed_count=0
file_renamed_count=0

# Function to rename files inside a directory
rename_files_in_dir() {
    local dir="$1"
    local old_name="$2"
    local new_name="$3"
    
    # Find and rename checkpoint files
    for checkpoint in "${dir}"/checkpoints/${old_name}_checkpoint_*.pt; do
        if [ -f "${checkpoint}" ]; then
            new_checkpoint=$(echo "${checkpoint}" | sed "s/${old_name}/${new_name}/g")
            echo -e "    Renaming checkpoint: $(basename ${checkpoint}) -> $(basename ${new_checkpoint})"
            mv "${checkpoint}" "${new_checkpoint}"
            file_renamed_count=$((file_renamed_count + 1))
        fi
    done
    
    # Rename metrics CSV
    old_metrics="${dir}/${old_name}_metrics.csv"
    if [ -f "${old_metrics}" ]; then
        new_metrics="${dir}/${new_name}_metrics.csv"
        echo -e "    Renaming metrics: $(basename ${old_metrics}) -> $(basename ${new_metrics})"
        mv "${old_metrics}" "${new_metrics}"
        file_renamed_count=$((file_renamed_count + 1))
    fi
    
    # Rename metric plot files
    for plot in "${dir}/${old_name}_metric_"*.png; do
        if [ -f "${plot}" ]; then
            new_plot=$(echo "${plot}" | sed "s/${old_name}/${new_name}/g")
            echo -e "    Renaming plot: $(basename ${plot}) -> $(basename ${new_plot})"
            mv "${plot}" "${new_plot}"
            file_renamed_count=$((file_renamed_count + 1))
        fi
    done
    
    # Rename any other files with the old name pattern
    find "${dir}" -type f -name "*${old_name}*" | while read -r file; do
        if [ -f "${file}" ]; then
            new_file=$(echo "${file}" | sed "s/${old_name}/${new_name}/g")
            if [ "${file}" != "${new_file}" ]; then
                echo -e "    Renaming file: $(basename ${file}) -> $(basename ${new_file})"
                mv "${file}" "${new_file}"
                file_renamed_count=$((file_renamed_count + 1))
            fi
        fi
    done
    
    # Update any references in text files (configs, logs, etc.)
    echo -e "  Updating references in text files..."
    find "${dir}" -type f \( -name "*.txt" -o -name "*.log" -o -name "*.yaml" -o -name "*.yml" -o -name "*.json" \) | while read -r text_file; do
        if grep -q "${old_name}" "${text_file}" 2>/dev/null; then
            echo -e "    Updating references in: $(basename ${text_file})"
            sed -i "s/${old_name}/${new_name}/g" "${text_file}"
        fi
    done
}

# Part 1: Rename bottleneck -> down_bottleneck
echo "=========================================="
echo "Part 1: Renaming *_bottleneck -> *_down_bottleneck"
echo "=========================================="
echo ""

for variant in "${VARIANTS[@]}"; do
    for size in "${SIZES[@]}"; do
        old_dir_name="diff_clip_${variant}_${size}_bottleneck"
        new_dir_name="diff_clip_${variant}_${size}_down_bottleneck"
        
        old_dir="${BASE_DIR}/${old_dir_name}"
        new_dir="${BASE_DIR}/${new_dir_name}"
        
        # Check if old directory exists
        if [ -d "${old_dir}" ]; then
            echo -e "${YELLOW}Found: ${old_dir_name}${NC}"
            
            # Check if new directory already exists
            if [ -d "${new_dir}" ]; then
                echo -e "${RED}  WARNING: Target directory already exists: ${new_dir_name}${NC}"
                echo -e "${RED}  Skipping to avoid overwriting...${NC}"
                continue
            fi
            
            # Rename the directory
            echo -e "  Renaming directory: ${old_dir_name} -> ${new_dir_name}"
            mv "${old_dir}" "${new_dir}"
            renamed_count=$((renamed_count + 1))
            
            # Now rename files inside that reference the old experiment name
            if [ -d "${new_dir}" ]; then
                echo -e "  Renaming files inside directory..."
                rename_files_in_dir "${new_dir}" "${old_dir_name}" "${new_dir_name}"
                echo -e "${GREEN}  ✓ Completed: ${new_dir_name}${NC}"
            fi
        else
            echo -e "  (Not found: ${old_dir_name})"
        fi
        echo ""
    done
done

# Part 2: Add _both_ to experiments missing rooms/scenes/both
echo ""
echo "=========================================="
echo "Part 2: Adding '_both_' to experiments"
echo "=========================================="
echo ""

# Pattern: diff_clip_(regular|spatial)_(small|medium|large)_(attention)
# Should become: diff_clip_(regular|spatial)_both_(small|medium|large)_(attention)

for clip_type in "regular" "spatial"; do
    for size in "${SIZES[@]}"; do
        # Find directories matching pattern without _both_, _rooms_, or _scenes_
        for old_dir in "${BASE_DIR}"/diff_clip_${clip_type}_${size}_*; do
            if [ -d "${old_dir}" ]; then
                old_dir_name=$(basename "${old_dir}")
                
                # Skip if already has _both_, _rooms_, or _scenes_
                if [[ "${old_dir_name}" == *"_both_"* ]] || \
                   [[ "${old_dir_name}" == *"_rooms_"* ]] || \
                   [[ "${old_dir_name}" == *"_scenes_"* ]]; then
                    continue
                fi
                
                # Extract attention part (everything after size_)
                attention_part=$(echo "${old_dir_name}" | sed "s/diff_clip_${clip_type}_${size}_//")
                
                # Build new name
                new_dir_name="diff_clip_${clip_type}_both_${size}_${attention_part}"
                new_dir="${BASE_DIR}/${new_dir_name}"
                
                echo -e "${YELLOW}Found: ${old_dir_name}${NC}"
                
                # Check if new directory already exists
                if [ -d "${new_dir}" ]; then
                    echo -e "${RED}  WARNING: Target directory already exists: ${new_dir_name}${NC}"
                    echo -e "${RED}  Skipping to avoid overwriting...${NC}"
                    continue
                fi
                
                # Rename the directory
                echo -e "  Renaming directory: ${old_dir_name} -> ${new_dir_name}"
                mv "${old_dir}" "${new_dir}"
                renamed_count=$((renamed_count + 1))
                
                # Rename files inside
                if [ -d "${new_dir}" ]; then
                    echo -e "  Renaming files inside directory..."
                    rename_files_in_dir "${new_dir}" "${old_dir_name}" "${new_dir_name}"
                    echo -e "${GREEN}  ✓ Completed: ${new_dir_name}${NC}"
                fi
                echo ""
            fi
        done
    done
done

echo "=========================================="
echo "Summary"
echo "=========================================="
echo -e "${GREEN}Directories renamed: ${renamed_count}${NC}"
echo -e "${GREEN}Files renamed: ${file_renamed_count}${NC}"
echo ""
echo "Done!"

