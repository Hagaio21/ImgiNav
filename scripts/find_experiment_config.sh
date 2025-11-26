#!/bin/bash
# Find config file for an experiment name

EXP_NAME="${1}"
BASE_DIR="${2:-/work3/s233249/ImgiNav/ImgiNav}"

if [ -z "${EXP_NAME}" ]; then
    echo "Usage: $0 <experiment_name> [base_dir]" >&2
    exit 1
fi

# Search in common config locations
CONFIG_DIRS=(
    "experiments/diffusion/clip"
    "experiments/diffusion/clip/regular"
    "experiments/diffusion/clip/regular_rooms"
    "experiments/diffusion/clip/regular_scenes"
    "experiments/diffusion/clip/spatial"
    "experiments/diffusion/clip/spatial_rooms"
    "experiments/diffusion/clip/spatial_scenes"
)

for config_dir in "${CONFIG_DIRS[@]}"; do
    full_dir="${BASE_DIR}/${config_dir}"
    if [ -d "${full_dir}" ]; then
        # Find YAML file with matching experiment name
        config_file=$(grep -l "name: ${EXP_NAME}" "${full_dir}"/*.yaml 2>/dev/null | head -n 1)
        if [ -n "${config_file}" ] && [ -f "${config_file}" ]; then
            # Return relative path from BASE_DIR
            echo "${config_file#${BASE_DIR}/}"
            exit 0
        fi
    fi
done

# Not found
exit 1

