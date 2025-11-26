#!/bin/bash
# Experiment Registry Management
# Maintains a central registry of all experiments with their status and metadata

REGISTRY_FILE="${1:-/work3/s233249/ImgiNav/experiments/clip/experiment_registry.txt}"
BASE_DIR="${2:-/work3/s233249/ImgiNav/experiments/clip}"
REPO_DIR="${3:-/work3/s233249/ImgiNav/ImgiNav}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Registry format:
# exp_name|status|current_epoch|target_epochs|completion_pct|config_path|last_updated|queue|job_id|notes

echo "Updating experiment registry..."
echo "Registry file: ${REGISTRY_FILE}"
echo ""

# Initialize registry if it doesn't exist
if [ ! -f "${REGISTRY_FILE}" ]; then
    {
        echo "# Experiment Registry"
        echo "# Format: exp_name|status|current_epoch|target_epochs|completion_pct|config_path|last_updated|queue|job_id|notes"
        echo "# Updated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
        echo "#"
    } > "${REGISTRY_FILE}"
fi

# Read existing registry into associative array (if bash 4+)
declare -A existing_entries

# Read existing entries
while IFS='|' read -r exp_name status current target pct config updated queue job_id notes || [ -n "${exp_name}" ]; do
    [[ "${exp_name}" =~ ^#.*$ ]] && continue
    [[ -z "${exp_name}" ]] && continue
    existing_entries["${exp_name}"]="${status}|${current}|${target}|${pct}|${config}|${updated}|${queue}|${job_id}|${notes}"
done < "${REGISTRY_FILE}"

# Find all experiment directories
exp_dirs=($(find "${BASE_DIR}" -maxdepth 1 -type d -name "diff_clip_*" | sort))

if [ ${#exp_dirs[@]} -eq 0 ]; then
    echo "No experiment directories found."
    exit 0
fi

echo "Scanning ${#exp_dirs[@]} experiments..."

# Process each experiment
declare -a registry_entries

for exp_dir in "${exp_dirs[@]}"; do
    exp_name=$(basename "${exp_dir}")
    
    # Read statistics.txt if it exists
    stats_file="${exp_dir}/statistics.txt"
    metrics_csv="${exp_dir}/${exp_name}_metrics.csv"
    
    # Initialize values
    status="not_started"
    current_epoch=0
    target_epochs=1000
    completion_pct="0.00"
    config_path=""
    last_updated=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
    queue=""
    job_id=""
    notes=""
    
    # Read from statistics.txt if available
    if [ -f "${stats_file}" ]; then
        while IFS='=' read -r key value; do
            case "${key}" in
                status) status="${value}" ;;
                current_epoch) current_epoch="${value}" ;;
                target_epochs) target_epochs="${value}" ;;
                completion_percentage) completion_pct="${value}" ;;
                config_path) config_path="${value}" ;;
                last_updated) last_updated="${value}" ;;
            esac
        done < "${stats_file}"
    fi
    
    # If no config_path from stats, try to find it
    if [ -z "${config_path}" ]; then
        config_path=$(bash "${SCRIPT_DIR}/find_experiment_config.sh" "${exp_name}" "${REPO_DIR}" 2>/dev/null || echo "")
    fi
    
    # Check for running jobs (using bjobs)
    # Look for jobs with experiment name in job name
    job_info=$(bjobs -w -J "${exp_name}" 2>/dev/null | grep -v "^JOBID" | head -n 1)
    if [ -n "${job_info}" ]; then
        job_id=$(echo "${job_info}" | awk '{print $1}')
        queue=$(echo "${job_info}" | awk '{print $6}')
        if [ -z "${queue}" ]; then
            queue=$(echo "${job_info}" | awk '{print $5}')
        fi
        # Update status if job is running
        if echo "${job_info}" | grep -q "RUN"; then
            status="running"
        elif echo "${job_info}" | grep -q "PEND"; then
            status="pending"
        fi
    else
        # Check if there's a job ID in existing registry
        if [ -n "${existing_entries[${exp_name}]}" ]; then
            old_job_id=$(echo "${existing_entries[${exp_name}]}" | cut -d'|' -f8)
            if [ -n "${old_job_id}" ] && [ "${old_job_id}" != "-" ]; then
                # Check if job still exists
                if ! bjobs "${old_job_id}" >/dev/null 2>&1; then
                    # Job finished, preserve notes but clear job info
                    notes=$(echo "${existing_entries[${exp_name}]}" | cut -d'|' -f9)
                    job_id="-"
                    queue="-"
                else
                    # Job still exists, get current info
                    job_info=$(bjobs "${old_job_id}" 2>/dev/null | grep -v "^JOBID" | head -n 1)
                    if [ -n "${job_info}" ]; then
                        queue=$(echo "${job_info}" | awk '{print $6}')
                        if [ -z "${queue}" ]; then
                            queue=$(echo "${job_info}" | awk '{print $5}')
                        fi
                        if echo "${job_info}" | grep -q "RUN"; then
                            status="running"
                        elif echo "${job_info}" | grep -q "PEND"; then
                            status="pending"
                        fi
                        notes=$(echo "${existing_entries[${exp_name}]}" | cut -d'|' -f9)
                    fi
                fi
            else
                # No job ID, preserve notes if any
                notes=$(echo "${existing_entries[${exp_name}]}" | cut -d'|' -f9)
            fi
        fi
    fi
    
    # Format: exp_name|status|current_epoch|target_epochs|completion_pct|config_path|last_updated|queue|job_id|notes
    registry_entries+=("${exp_name}|${status}|${current_epoch}|${target_epochs}|${completion_pct}|${config_path}|${last_updated}|${queue}|${job_id}|${notes}")
done

# Sort by experiment name
IFS=$'\n' sorted_entries=($(printf '%s\n' "${registry_entries[@]}" | sort))
unset IFS

# Write registry
{
    echo "# Experiment Registry"
    echo "# Format: exp_name|status|current_epoch|target_epochs|completion_pct|config_path|last_updated|queue|job_id|notes"
    echo "# Updated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
    echo "#"
    for entry in "${sorted_entries[@]}"; do
        echo "${entry}"
    done
} > "${REGISTRY_FILE}"

echo ""
echo "Registry updated with ${#registry_entries[@]} experiments"
echo "Registry file: ${REGISTRY_FILE}"

# Show summary
echo ""
echo "Summary by status:"
grep -v "^#" "${REGISTRY_FILE}" | cut -d'|' -f2 | sort | uniq -c | while read count status; do
    printf "  %-15s: %3d\n" "${status}" "${count}"
done

