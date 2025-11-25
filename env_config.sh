#!/bin/bash
# Environment configuration script for ImgiNav HPC scripts
# Sources this file to get BASE_DIR variable set dynamically

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If IMGINAV_ROOT is set as environment variable, use it
# Otherwise, infer from script location (assumes script is in project root or subdirectory)
if [ -n "$IMGINAV_ROOT" ]; then
    BASE_DIR="$IMGINAV_ROOT"
else
    # Try to find project root by looking for common markers
    # Start from script directory and walk up
    CURRENT_DIR="$SCRIPT_DIR"
    while [ "$CURRENT_DIR" != "/" ]; do
        # Check for common project root markers
        if [ -f "$CURRENT_DIR/imginav_env.yml" ] || \
           [ -f "$CURRENT_DIR/requirements.txt" ] || \
           [ -d "$CURRENT_DIR/common" ] && [ -d "$CURRENT_DIR/models" ]; then
            BASE_DIR="$CURRENT_DIR"
            break
        fi
        CURRENT_DIR="$(dirname "$CURRENT_DIR")"
    done
    
    # Fallback: assume script is in project root or one level down
    if [ -z "$BASE_DIR" ]; then
        # If script is in a subdirectory, go up one level
        if [ -d "$(dirname "$SCRIPT_DIR")/common" ]; then
            BASE_DIR="$(dirname "$SCRIPT_DIR")"
        else
            BASE_DIR="$SCRIPT_DIR"
        fi
    fi
fi

# Resolve to absolute path
BASE_DIR="$(cd "$BASE_DIR" && pwd)"

# Export for use in other scripts
export BASE_DIR
export IMGINAV_ROOT="${IMGINAV_ROOT:-$BASE_DIR}"

