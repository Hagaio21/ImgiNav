#!/bin/bash
# Update unfinished.txt with current list of unfinished experiments
# Also updates statistics.txt in each experiment directory

# Default values
BASE_DIR="${1:-/work3/s233249/ImgiNav/experiments/clip}"
DEFAULT_TARGET_EPOCHS="${2:-1000}"
REPO_DIR="${3:-/work3/s233249/ImgiNav/ImgiNav}"
OUTPUT_FILE="${BASE_DIR}/unfinished.txt"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGISTRY_SCRIPT="${SCRIPT_DIR}/experiment_registry.sh"
REGISTRY_FILE="${BASE_DIR}/experiment_registry.txt"

echo "Updating unfinished experiments list..."
echo "Base directory: ${BASE_DIR}"
echo "Output file: ${OUTPUT_FILE}"
echo ""

# Also update experiment registry
echo "Updating experiment registry..."
bash "${REGISTRY_SCRIPT}" "${REGISTRY_FILE}" "${BASE_DIR}" "${REPO_DIR}" >/dev/null 2>&1
echo ""

# Array to store unfinished experiments with details
declare -a unfinished_data

# Find all experiment directories
exp_dirs=($(find "${BASE_DIR}" -maxdepth 1 -type d -name "diff_clip_*" | sort))

if [ ${#exp_dirs[@]} -eq 0 ]; then
    echo "No experiment directories found."
    echo "" > "${OUTPUT_FILE}"
    exit 0
fi

echo "Scanning ${#exp_dirs[@]} experiments..."

# Process each experiment
for exp_dir in "${exp_dirs[@]}"; do
    exp_name=$(basename "${exp_dir}")
    
    # Look for metrics CSV file
    metrics_csv="${exp_dir}/${exp_name}_metrics.csv"
    stats_file="${exp_dir}/statistics.txt"
    
    # Initialize stats
    current_epoch=0
    target_epochs="${DEFAULT_TARGET_EPOCHS}"
    status="not_started"
    completion_pct=0.0
    
    # Try to get target epochs from config file
    config_dirs=(
        "experiments/diffusion/clip"
        "experiments/diffusion/clip/regular"
        "experiments/diffusion/clip/regular_rooms"
        "experiments/diffusion/clip/regular_scenes"
        "experiments/diffusion/clip/spatial"
        "experiments/diffusion/clip/spatial_rooms"
        "experiments/diffusion/clip/spatial_scenes"
    )
    
    config_path=""
    for config_dir in "${config_dirs[@]}"; do
        full_dir="${REPO_DIR}/${config_dir}"
        if [ -d "${full_dir}" ]; then
            config_file=$(grep -l "name: ${exp_name}" "${full_dir}"/*.yaml 2>/dev/null | head -n 1)
            if [ -n "${config_file}" ] && [ -f "${config_file}" ]; then
                config_path="${config_file#${REPO_DIR}/}"
                # Extract epochs_target or epochs from config
                config_target=$(grep -E "epochs_target:|epochs:" "${config_file}" | head -n 1 | awk '{print $2}' | tr -d ' ')
                if [ -n "${config_target}" ] && [[ "${config_target}" =~ ^[0-9]+$ ]]; then
                    target_epochs="${config_target}"
                fi
                break
            fi
        fi
    done
    
    # Get current epoch from metrics CSV
    if [ -f "${metrics_csv}" ]; then
        line_count=$(wc -l < "${metrics_csv}" | tr -d ' ')
        if [ "${line_count}" -gt 1 ]; then
            # Get last epoch from CSV
            first_line=$(head -n 1 "${metrics_csv}")
            has_header=false
            epoch_col=1
            
            if echo "${first_line}" | grep -q "epoch"; then
                has_header=true
                epoch_col=$(echo "${first_line}" | tr ',' '\n' | grep -n "^epoch$" | cut -d: -f1)
                if [ -z "${epoch_col}" ]; then
                    epoch_col=$(echo "${first_line}" | tr ',' '\n' | grep -ni "^epoch$" | cut -d: -f1)
                fi
                if [ -z "${epoch_col}" ]; then
                    epoch_col=1
                fi
            fi
            
            if [ "${has_header}" = true ]; then
                current_epoch=$(tail -n +2 "${metrics_csv}" | tail -n 1 | awk -F',' -v col="${epoch_col}" '{print $col}' | tr -d ' ' | tr -d '"')
            else
                current_epoch=$(tail -n 1 "${metrics_csv}" | awk -F',' -v col="${epoch_col}" '{print $col}' | tr -d ' ' | tr -d '"')
            fi
            
            if ! [[ "${current_epoch}" =~ ^[0-9]+$ ]]; then
                current_epoch=0
            fi
        fi
    fi
    
    # Calculate completion percentage (using awk for portability)
    if [ "${target_epochs}" -gt 0 ]; then
        completion_pct=$(awk "BEGIN {printf \"%.2f\", ${current_epoch} * 100 / ${target_epochs}}")
    else
        completion_pct="0.00"
    fi
    
    # Determine status
    if [ "${current_epoch}" -eq 0 ]; then
        status="not_started"
    elif [ "${current_epoch}" -ge "${target_epochs}" ]; then
        status="finished"
    else
        status="unfinished"
    fi
    
    # Write statistics.txt
    {
        echo "experiment_name=${exp_name}"
        echo "status=${status}"
        echo "current_epoch=${current_epoch}"
        echo "target_epochs=${target_epochs}"
        echo "completion_percentage=${completion_pct}"
        echo "config_path=${config_path}"
        echo "last_updated=$(date -u +"%Y-%m-%d %H:%M:%S UTC")"
    } > "${stats_file}"
    
    # Add to unfinished list if not finished
    if [ "${status}" != "finished" ]; then
        unfinished_data+=("${exp_name}|${current_epoch}|${target_epochs}|${completion_pct}|${status}|${config_path}")
    fi
done

# Sort by completion percentage (descending) - almost done first
IFS=$'\n' sorted_data=($(printf '%s\n' "${unfinished_data[@]}" | sort -t'|' -k4 -rn))
unset IFS

# Write unfinished.txt
{
    echo "# Unfinished experiments list"
    echo "# Format: exp_name|current_epoch|target_epochs|completion_pct|status|config_path"
    echo "# Updated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
    echo "#"
    for line in "${sorted_data[@]}"; do
        echo "${line}"
    done
} > "${OUTPUT_FILE}"

echo ""
echo "Updated ${#unfinished_data[@]} unfinished experiments"
echo "Output written to: ${OUTPUT_FILE}"
echo ""
echo "Top 5 unfinished experiments (by completion %):"
head -n 8 "${OUTPUT_FILE}" | tail -n 5 | while IFS='|' read -r name current target pct status config; do
    printf "  %-50s %4d/%4d (%5.1f%%) [%s]\n" "${name}" "${current}" "${target}" "${pct}" "${status}"
done

