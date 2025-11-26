#!/bin/bash
# List unfinished experiments by checking metrics CSV files for last epoch

# Default values
BASE_DIR="${1:-/work3/s233249/ImgiNav/experiments/clip}"
DEFAULT_TARGET_EPOCHS="${2:-1000}"

echo "================================================================================"
echo "Unfinished Experiments Scanner"
echo "================================================================================"
echo "Base directory: ${BASE_DIR}"
echo "Default target epochs: ${DEFAULT_TARGET_EPOCHS}"
echo ""

if [ ! -d "${BASE_DIR}" ]; then
    echo "Error: Base directory does not exist: ${BASE_DIR}"
    exit 1
fi

# Array to store unfinished experiments
unfinished=()

# Find all experiment directories
exp_dirs=($(find "${BASE_DIR}" -maxdepth 1 -type d -name "diff_clip_*" | sort))

if [ ${#exp_dirs[@]} -eq 0 ]; then
    echo "No experiment directories found."
    exit 0
fi

echo "Found ${#exp_dirs[@]} experiment directories"
echo ""

# Process each experiment
for exp_dir in "${exp_dirs[@]}"; do
    exp_name=$(basename "${exp_dir}")
    
    # Look for metrics CSV file
    metrics_csv="${exp_dir}/${exp_name}_metrics.csv"
    
    if [ ! -f "${metrics_csv}" ]; then
        echo "  ${exp_name}: No metrics CSV (not started)"
        continue
    fi
    
    # Check if CSV has data (more than just header)
    line_count=$(wc -l < "${metrics_csv}" | tr -d ' ')
    if [ "${line_count}" -le 1 ]; then
        echo "  ${exp_name}: Metrics CSV is empty or has only header (not started)"
        continue
    fi
    
    # Get last epoch from CSV
    # First, check if CSV has a header row
    first_line=$(head -n 1 "${metrics_csv}")
    has_header=false
    epoch_col=1
    
    if echo "${first_line}" | grep -q "epoch"; then
        # Has header - find epoch column index
        has_header=true
        epoch_col=$(echo "${first_line}" | tr ',' '\n' | grep -n "^epoch$" | cut -d: -f1)
        if [ -z "${epoch_col}" ]; then
            # Try case-insensitive
            epoch_col=$(echo "${first_line}" | tr ',' '\n' | grep -ni "^epoch$" | cut -d: -f1)
        fi
        if [ -z "${epoch_col}" ]; then
            # Fallback to first column
            epoch_col=1
        fi
    fi
    
    # Get last epoch from last data row
    if [ "${has_header}" = true ]; then
        # Skip header, get last line
        last_epoch=$(tail -n +2 "${metrics_csv}" | tail -n 1 | awk -F',' -v col="${epoch_col}" '{print $col}' | tr -d ' ' | tr -d '"')
    else
        # No header, get last line
        last_epoch=$(tail -n 1 "${metrics_csv}" | awk -F',' -v col="${epoch_col}" '{print $col}' | tr -d ' ' | tr -d '"')
    fi
    
    # Check if last_epoch is a valid number
    if ! [[ "${last_epoch}" =~ ^[0-9]+$ ]]; then
        echo "  ${exp_name}: Could not parse epoch from CSV (got: '${last_epoch}')"
        continue
    fi
    
    # Try to get target epochs from config file
    target_epochs="${DEFAULT_TARGET_EPOCHS}"
    
    # Look for config in common locations
    config_dirs=(
        "experiments/diffusion/clip"
        "experiments/diffusion/clip/regular"
        "experiments/diffusion/clip/regular_rooms"
        "experiments/diffusion/clip/regular_scenes"
        "experiments/diffusion/clip/spatial"
        "experiments/diffusion/clip/spatial_rooms"
        "experiments/diffusion/clip/spatial_scenes"
    )
    
    for config_dir in "${config_dirs[@]}"; do
        if [ -d "${config_dir}" ]; then
            # Find YAML file with matching experiment name
            config_file=$(grep -l "name: ${exp_name}" "${config_dir}"/*.yaml 2>/dev/null | head -n 1)
            if [ -n "${config_file}" ] && [ -f "${config_file}" ]; then
                # Try to extract epochs_target or epochs from config
                config_target=$(grep -E "epochs_target:|epochs:" "${config_file}" | head -n 1 | awk '{print $2}' | tr -d ' ')
                if [ -n "${config_target}" ] && [[ "${config_target}" =~ ^[0-9]+$ ]]; then
                    target_epochs="${config_target}"
                    break
                fi
            fi
        fi
    done
    
    # Check if unfinished
    if [ "${last_epoch}" -lt "${target_epochs}" ]; then
        unfinished+=("${exp_name} - ${last_epoch}/${target_epochs}")
        echo "  ${exp_name}: ${last_epoch}/${target_epochs} (unfinished)"
    else
        echo "  ${exp_name}: ${last_epoch}/${target_epochs} (complete)"
    fi
done

echo ""
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo "Total unfinished experiments: ${#unfinished[@]}"
echo ""

if [ ${#unfinished[@]} -gt 0 ]; then
    echo "Unfinished experiments:"
    echo "--------------------------------------------------------------------------------"
    for exp in "${unfinished[@]}"; do
        echo "${exp}"
    done
    echo "--------------------------------------------------------------------------------"
else
    echo "All experiments are complete!"
fi

echo "================================================================================"

