#!/bin/bash
# Get the best available GPU queue for job submission
# Parses bqueue output and selects the best non-interactive queue

USER_ARG="${1:-${USER:-$(whoami)}}"

if [ -z "${USER_ARG}" ]; then
    echo "Error: Could not determine user" >&2
    exit 1
fi

# Get queue information
QUEUE_INFO=$(bqueue -u "${USER_ARG}" 2>/dev/null)

if [ $? -ne 0 ] || [ -z "${QUEUE_INFO}" ]; then
    # Fallback to default queue if bqueue fails
    echo "gpuv100"
    exit 0
fi

# Parse queues and their availability
# bqueue output format varies, but typically shows:
# QUEUE_NAME  PRIO STATUS  MAX JL/U JL/P JL/H NJOBS PEND RUN SUSP
# We want to find queues with available slots (not full)

declare -a available_queues
declare -a queue_priorities

# Common GPU queue names (excluding interactive)
# Interactive queues typically have "interactive" or "int" in the name
INTERACTIVE_PATTERNS="interactive|int|debug"

# Parse bqueue output line by line
while IFS= read -r line; do
    # Skip header lines and empty lines
    [[ "${line}" =~ ^QUEUE ]] && continue
    [[ "${line}" =~ ^-+$ ]] && continue
    [[ -z "${line}" ]] && continue
    
    # Extract queue name (first column)
    queue_name=$(echo "${line}" | awk '{print $1}')
    
    # Skip if it's an interactive queue
    if echo "${queue_name}" | grep -qiE "${INTERACTIVE_PATTERNS}"; then
        continue
    fi
    
    # Skip if it doesn't look like a GPU queue
    if ! echo "${queue_name}" | grep -qiE "gpu|v100|a100|l40|h100"; then
        continue
    fi
    
    # Try to extract available slots
    # Format varies, but typically: MAX - RUN - PEND = available
    # We'll use a simpler heuristic: prefer queues with fewer pending jobs
    
    # Extract pending jobs (varies by bqueue version)
    # Try different column positions (bqueue format can vary)
    pending=999
    for col_offset in -3 -2 -1; do
        test_val=$(echo "${line}" | awk -v offset="${col_offset}" '{print $(NF+offset)}' 2>/dev/null)
        if [[ "${test_val}" =~ ^[0-9]+$ ]] && [ "${test_val}" -lt "${pending}" ]; then
            pending="${test_val}"
        fi
    done
    
    # If we still can't parse, try to find "PEND" column
    if [ "${pending}" -eq 999 ]; then
        # Look for PEND column header to find position
        if echo "${QUEUE_INFO}" | head -n 1 | grep -q "PEND"; then
            pend_col=$(echo "${QUEUE_INFO}" | head -n 1 | tr ' ' '\n' | grep -n "PEND" | cut -d: -f1)
            if [ -n "${pend_col}" ]; then
                pending=$(echo "${line}" | awk -v col="${pend_col}" '{print $col}' 2>/dev/null || echo "0")
            fi
        fi
    fi
    
    # If we can't parse, assume it's available (low priority)
    if ! [[ "${pending}" =~ ^[0-9]+$ ]]; then
        pending=0
    fi
    
    # Priority: lower pending = better
    # Also prefer certain queue types
    priority="${pending}"
    
    # Boost priority for preferred queue types
    if echo "${queue_name}" | grep -qi "gpuv100"; then
        priority=$((priority - 10))  # Prefer v100
    elif echo "${queue_name}" | grep -qi "gpua100"; then
        priority=$((priority - 5))  # Prefer a100
    fi
    
    available_queues+=("${queue_name}")
    queue_priorities+=("${priority}")
done <<< "${QUEUE_INFO}"

# If no queues found, use default
if [ ${#available_queues[@]} -eq 0 ]; then
    echo "gpuv100"
    exit 0
fi

# Find queue with lowest priority (best)
best_queue="${available_queues[0]}"
best_priority="${queue_priorities[0]}"

for i in "${!available_queues[@]}"; do
    if [ "${queue_priorities[$i]}" -lt "${best_priority}" ]; then
        best_queue="${available_queues[$i]}"
        best_priority="${queue_priorities[$i]}"
    fi
done

echo "${best_queue}"

