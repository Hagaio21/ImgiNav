#!/bin/bash
# Launch experiments from unfinished.txt, prioritizing almost-done experiments

# Default values
BASE_DIR="${1:-/work3/s233249/ImgiNav/experiments/clip}"
REPO_DIR="${2:-/work3/s233249/ImgiNav/ImgiNav}"
UNFINISHED_FILE="${BASE_DIR}/unfinished.txt"
MIN_COMPLETION="${3:-0}"  # Minimum completion % to launch (0 = launch all)
MAX_JOBS="${4:-10}"  # Maximum number of jobs to launch at once
DRY_RUN="${5:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${REPO_DIR}/training/hpc_scripts/run_train_diff_clip.sh"
GET_QUEUE_SCRIPT="${SCRIPT_DIR}/get_best_gpu_queue.sh"
REGISTRY_SCRIPT="${SCRIPT_DIR}/experiment_registry.sh"
REGISTRY_FILE="${BASE_DIR}/experiment_registry.txt"
registry_needs_update=false

echo "================================================================================"
echo "Launch Experiments from unfinished.txt"
echo "================================================================================"
echo "Unfinished file: ${UNFINISHED_FILE}"
echo "Repository directory: ${REPO_DIR}"
echo "Minimum completion %: ${MIN_COMPLETION}%"
echo "Maximum jobs: ${MAX_JOBS}"
if [ "${DRY_RUN}" = "true" ]; then
    echo "DRY RUN MODE - No jobs will be submitted"
fi
echo ""

# Get best available GPU queue
echo "Checking available GPU queues..."
CURRENT_USER="${USER:-$(whoami)}"
BEST_QUEUE=$(bash "${GET_QUEUE_SCRIPT}" "${CURRENT_USER}")
echo "Selected queue: ${BEST_QUEUE}"
echo ""

# Check if unfinished.txt exists
if [ ! -f "${UNFINISHED_FILE}" ]; then
    echo "Error: unfinished.txt not found at ${UNFINISHED_FILE}"
    echo "Run update_unfinished_list.sh first to generate it."
    exit 1
fi

# Read unfinished experiments (skip comments and empty lines)
declare -a candidates
while IFS='|' read -r exp_name current_epoch target_epochs completion_pct status config_path || [ -n "${exp_name}" ]; do
    # Skip comments and empty lines
    [[ "${exp_name}" =~ ^#.*$ ]] && continue
    [[ -z "${exp_name}" ]] && continue
    
    # Check minimum completion (compare as float using awk)
    if [ -n "${completion_pct}" ]; then
        # awk: if completion_pct >= MIN_COMPLETION, print 1, else print 0
        result=$(awk "BEGIN {if (${completion_pct} >= ${MIN_COMPLETION}) print 1; else print 0}")
        if [ "${result}" = "1" ]; then
            candidates+=("${exp_name}|${current_epoch}|${target_epochs}|${completion_pct}|${status}|${config_path}")
        fi
    fi
done < "${UNFINISHED_FILE}"

if [ ${#candidates[@]} -eq 0 ]; then
    echo "No experiments found matching criteria (min completion: ${MIN_COMPLETION}%)"
    exit 0
fi

echo "Found ${#candidates[@]} candidate experiments"
echo ""

# Limit to MAX_JOBS
if [ ${#candidates[@]} -gt ${MAX_JOBS} ]; then
    echo "Limiting to top ${MAX_JOBS} experiments (by completion %)..."
    candidates=("${candidates[@]:0:${MAX_JOBS}}")
fi

# Show what will be launched
echo "Experiments to launch:"
echo "--------------------------------------------------------------------------------"
for candidate in "${candidates[@]}"; do
    IFS='|' read -r exp_name current_epoch target_epochs completion_pct status config_path <<< "${candidate}"
    printf "  %-50s %4d/%4d (%5.1f%%) [%s]\n" "${exp_name}" "${current_epoch}" "${target_epochs}" "${completion_pct}" "${status}"
done
echo "--------------------------------------------------------------------------------"
echo ""

if [ "${DRY_RUN}" = "true" ]; then
    echo "[DRY RUN] Would launch ${#candidates[@]} experiments"
    exit 0
fi

# Ask for confirmation
read -p "Launch these ${#candidates[@]} experiments? (yes/no): " confirm
if [ "${confirm}" != "yes" ] && [ "${confirm}" != "y" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Launching experiments..."
echo ""

# Launch each experiment
launched=0
failed=0

for candidate in "${candidates[@]}"; do
    IFS='|' read -r exp_name current_epoch target_epochs completion_pct status config_path <<< "${candidate}"
    
    echo "Processing: ${exp_name} (${current_epoch}/${target_epochs}, ${completion_pct}%)"
    
    # Check if config path is valid
    if [ -z "${config_path}" ] || [ ! -f "${REPO_DIR}/${config_path}" ]; then
        echo "  ERROR: Config file not found: ${config_path}"
        ((failed++))
        continue
    fi
    
    # Sanitize job name
    job_name=$(echo "${exp_name}" | sed 's/[^a-zA-Z0-9_]/_/g' | sed 's/_\+/_/g')
    if [ ${#job_name} -gt 50 ]; then
        job_name="${job_name:0:50}"
    fi
    
    # Generate log suffix
    log_suffix=$(echo "${config_path}" | sed 's/[^a-zA-Z0-9]/_/g' | sed 's/_\+/_/g')
    
    echo "  Submitting job: ${job_name}"
    echo "  Config: ${config_path}"
    echo "  Queue: ${BEST_QUEUE}"
    
    # Submit job to best available queue
    job_output=$(bsub -J "${job_name}" \
        -o "${REPO_DIR}/training/hpc_scripts/logs/train_diff_clip_${log_suffix}.%J.out" \
        -e "${REPO_DIR}/training/hpc_scripts/logs/train_diff_clip_${log_suffix}.%J.err" \
        -n 4 \
        -R "rusage[mem=8000]" \
        -gpu "num=1" \
        -W 24:00 \
        -q "${BEST_QUEUE}" \
        bash "${TRAIN_SCRIPT}" "${config_path}" 2>&1)
    
    # Extract job ID from bsub output (format: "Job <job_id> is submitted to queue <queue>")
    job_id=$(echo "${job_output}" | grep -oP 'Job <\K[0-9]+' || echo "")
    
    if [ $? -eq 0 ] && [ -n "${job_id}" ]; then
        echo "  ✓ Submitted successfully (Job ID: ${job_id})"
        ((launched++))
        
        # Update registry with job info (will be fully updated at end)
        # Just mark that we need to update
        registry_needs_update=true
    else
        echo "  ✗ Failed to submit"
        ((failed++))
    fi
    
    # Re-check queue availability for next job (queue might have changed)
    if [ ${launched} -lt ${#candidates[@]} ]; then
        BEST_QUEUE=$(bash "${GET_QUEUE_SCRIPT}" "${CURRENT_USER}")
        echo "  Queue re-checked: ${BEST_QUEUE}"
    fi
    
    sleep 1
    echo ""
done

# Update registry after all launches
if [ "${DRY_RUN}" != "true" ] && [ "${registry_needs_update}" = "true" ]; then
    echo ""
    echo "Updating experiment registry..."
    bash "${REGISTRY_SCRIPT}" "${REGISTRY_FILE}" "${BASE_DIR}" "${REPO_DIR}" >/dev/null 2>&1
fi

echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo "Total candidates: ${#candidates[@]}"
echo "Successfully launched: ${launched}"
echo "Failed to launch: ${failed}"
echo "================================================================================"

