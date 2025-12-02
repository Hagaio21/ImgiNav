#!/bin/bash
#
# Launch Parallel Dataset Cleaning
#
# Creates shards of scene IDs and submits an LSF job array to check
# layout quality in parallel. Each shard outputs a rejections CSV,
# which are merged after all jobs complete.
#
# Usage:
#   bash launch_clean_dataset.sh --manifest /work3/s233249/ImgiNav/dataset_v2/manifests/manifest_seg.csv
#   bash launch_clean_dataset.sh --manifest manifests/manifest_seg.csv --num-shards 100
#   bash launch_clean_dataset.sh --manifest manifests/manifest_seg.csv --no-pov-check  # Disable POV
#   bash launch_clean_dataset.sh  # Without manifest (layout checking only, uses metadata)

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav"
DATASET_ROOT="${BASE_DIR}/dataset_v2"
SCRIPTS_DIR="${BASE_DIR}/ImgiNav/data_preparation_v2"
HPC_SCRIPTS_DIR="${SCRIPTS_DIR}/hpc_scripts"
LOG_DIR="${HPC_SCRIPTS_DIR}/logs"

SHARDS_DIR="${DATASET_ROOT}/shards"
OUTPUT_DIR="${DATASET_ROOT}/rejections"

# Defaults
NUM_SHARDS=100
DRY_RUN=0
MANIFEST_PATH="/work3/s233249/ImgiNav/dataset_v2/manifests/manifest_seg.csv"  # Default manifest
CHECK_POV_PALETTE=1  # POV checking is default when manifest provided
POV_COLOR_TOLERANCE=20
POV_MIN_MATCH_RATIO=0.7
PALETTE_COLOR_TOLERANCE=10
SKIP_ADD_SAMPLE_ID=0
ADD_SAMPLE_ID_JOB_ID=""  # Will be set if we submit add_sample_id job

# Layout quality thresholds
MIN_PIXELS=100
MAX_BLACK_FRACTION=0.95
MIN_CONTENT_FRACTION=0.05

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-shards)
            NUM_SHARDS="$2"
            shift 2
            ;;
        --manifest)
            MANIFEST_PATH="$2"
            CHECK_POV_PALETTE=1  # Manifest enables POV checking by default
            shift 2
            ;;
        --no-manifest)
            MANIFEST_PATH=""
            CHECK_POV_PALETTE=0
            shift
            ;;
        --pov-color-tolerance)
            POV_COLOR_TOLERANCE="$2"
            shift 2
            ;;
        --pov-min-match-ratio)
            POV_MIN_MATCH_RATIO="$2"
            shift 2
            ;;
        --palette-color-tolerance)
            PALETTE_COLOR_TOLERANCE="$2"
            shift 2
            ;;
        --min-pixels)
            MIN_PIXELS="$2"
            shift 2
            ;;
        --max-black-fraction)
            MAX_BLACK_FRACTION="$2"
            shift 2
            ;;
        --min-content-fraction)
            MIN_CONTENT_FRACTION="$2"
            shift 2
            ;;
        --skip-add-sample-id)
            SKIP_ADD_SAMPLE_ID=1
            shift
            ;;
        --no-pov-check)
            CHECK_POV_PALETTE=0
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --num-shards N              Number of parallel shards (default: 50)"
            echo ""
            echo "Layout Quality Thresholds:"
            echo "  --min-pixels N              Minimum pixels for required classes (default: 100)"
            echo "  --max-black-fraction F     Max fraction of black pixels (default: 0.95)"
            echo "  --min-content-fraction F   Min fraction of non-background content (default: 0.05)"
            echo ""
            echo "POV Uniformity Checking:"
            echo "  --manifest PATH             Manifest CSV file with sample_id (required, enables POV checking)"
            echo "  --no-pov-check              Disable POV uniformity checking"
            echo "  --skip-add-sample-id        Skip sample_id check (assumes already present)"
            echo "  --pov-color-tolerance N     Color quantization tolerance (default: 20)"
            echo "  --pov-min-match-ratio F     Max dominant color fraction - reject if >= this (default: 0.7)"
            echo ""
            echo "Other:"
            echo "  --dry-run                   Don't submit jobs, just create shards"
            echo "  --help                      Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Launching Parallel Dataset Cleaning"
echo "=========================================="
echo "Dataset Root: ${DATASET_ROOT}"
echo "Num Shards: ${NUM_SHARDS}"
echo ""
echo "Layout Quality Thresholds:"
echo "  Min Pixels: ${MIN_PIXELS}"
echo "  Max Black Fraction: ${MAX_BLACK_FRACTION}"
echo "  Min Content Fraction: ${MIN_CONTENT_FRACTION}"
echo ""
if [ "${CHECK_POV_PALETTE}" = "1" ]; then
            echo "POV Uniformity Checking: ENABLED (default)"
            if [ -n "${MANIFEST_PATH}" ]; then
                echo "  Manifest: ${MANIFEST_PATH}"
            fi
            echo "  Color Quantization Tolerance: ${POV_COLOR_TOLERANCE}"
            echo "  Max Dominant Color Fraction: ${POV_MIN_MATCH_RATIO}"
else
    echo "POV Palette Checking: DISABLED"
fi
echo ""
echo "Dry Run: ${DRY_RUN}"
echo ""

# =============================================================================
# CREATE DIRECTORIES
# =============================================================================
mkdir -p "${SHARDS_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# =============================================================================
# RESOLVE MANIFEST PATH AND ADD SAMPLE_ID
# =============================================================================
RESOLVED_MANIFEST_PATH=""
MANIFEST_WITH_IDS=""

if [ -n "${MANIFEST_PATH}" ]; then
    # Resolve manifest path
    if [ -f "${DATASET_ROOT}/${MANIFEST_PATH}" ]; then
        RESOLVED_MANIFEST_PATH="${DATASET_ROOT}/${MANIFEST_PATH}"
    elif [ -f "${MANIFEST_PATH}" ]; then
        RESOLVED_MANIFEST_PATH="${MANIFEST_PATH}"
    else
        echo "ERROR: Manifest file not found: ${MANIFEST_PATH}" >&2
        exit 1
    fi
    
    # Check if sample_id exists, if not submit job to add it (in place)
    if [ "${SKIP_ADD_SAMPLE_ID}" = "0" ]; then
        if ! head -1 "${RESOLVED_MANIFEST_PATH}" | grep -q "sample_id"; then
            echo "sample_id column not found, will submit job to add it (updating manifest in place)..."
            
            # Create temporary script to run add_sample_id (updates in place)
            TEMP_SCRIPT="${SHARDS_DIR}/temp_add_sample_id.sh"
            cat > "${TEMP_SCRIPT}" <<EOF
#!/bin/bash
#BSUB -J add_sample_id
#BSUB -n 1
#BSUB -R "rusage[mem=2000]"
#BSUB -W 00:30
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

cd "${BASE_DIR}"

if [ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "\$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || conda activate scenefactor || exit 1
fi

# Update manifest in place (output to same file)
python "${SCRIPTS_DIR}/add_sample_id.py" \
    --manifest "${RESOLVED_MANIFEST_PATH}" \
    --output "${RESOLVED_MANIFEST_PATH}"

exit \$?
EOF
            chmod +x "${TEMP_SCRIPT}"
            
            # Submit job to add sample_id
            echo "  Submitting job to add sample_id..."
            ADD_ID_OUTPUT=$(bsub < "${TEMP_SCRIPT}")
            
            ADD_ID_JOB_ID=$(echo "${ADD_ID_OUTPUT}" | grep -oP '(?<=Job <)\d+(?=>)' || echo "")
            
            if [ -z "${ADD_ID_JOB_ID}" ]; then
                echo "ERROR: Failed to submit add_sample_id job" >&2
                exit 1
            fi
            
            ADD_SAMPLE_ID_JOB_ID="${ADD_ID_JOB_ID}"
            echo "  Submitted: Job ${ADD_SAMPLE_ID_JOB_ID}"
            echo "  Cleaning array job will depend on this job completing"
            rm -f "${TEMP_SCRIPT}"
        else
            ADD_SAMPLE_ID_JOB_ID=""
        fi
    else
        ADD_SAMPLE_ID_JOB_ID=""
    fi
    
    # Use the same manifest path (will be updated in place by add_sample_id job)
    MANIFEST_WITH_IDS="${RESOLVED_MANIFEST_PATH}"
fi

# =============================================================================
# CREATE SHARDS (submit as job if add_sample_id was submitted, otherwise do it now)
# =============================================================================
if [ -n "${ADD_SAMPLE_ID_JOB_ID}" ] && [ -n "${MANIFEST_WITH_IDS}" ]; then
    # Need to create shards after add_sample_id finishes - submit as job
    echo "Will create shards after add_sample_id job completes..."
    
    # Create script to extract sample_id, scene_id, type and create CSV shards
    CREATE_SHARDS_SCRIPT="${SHARDS_DIR}/create_shards.sh"
    cat > "${CREATE_SHARDS_SCRIPT}" <<EOF
#!/bin/bash
#BSUB -J create_shards
#BSUB -n 1
#BSUB -R "rusage[mem=2000]"
#BSUB -W 00:10
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

cd "${BASE_DIR}"

# Set MKL_INTERFACE_LAYER before conda activation (required by conda env activation script)
export MKL_INTERFACE_LAYER=LP64

if [ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "\$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || conda activate scenefactor || exit 1
fi

# Use Python from conda environment to create CSV shards with sample_id, scene_id, type
python <<PYTHON_SCRIPT
import pandas as pd
import sys
from pathlib import Path

manifest_path = "${MANIFEST_WITH_IDS}"
shards_dir = Path("${SHARDS_DIR}")
num_shards = ${NUM_SHARDS}

# Load manifest
df = pd.read_csv(manifest_path, low_memory=False)
df.columns = df.columns.str.strip()

# Check required columns
required_cols = ['sample_id', 'scene_id']
if 'type' not in df.columns:
    print("WARNING: 'type' column not found in manifest, will use empty string", file=sys.stderr)
    df['type'] = ''

# Extract sample_id, scene_id, type
shard_df = df[['sample_id', 'scene_id', 'type']].copy()
shard_df = shard_df.dropna(subset=['sample_id', 'scene_id'])

# Get unique scene IDs for splitting
unique_scenes = shard_df['scene_id'].unique()
num_scenes = len(unique_scenes)
scenes_per_shard = max(1, num_scenes // num_shards)

# Group by scene_id and assign to shards
shard_df['shard_num'] = shard_df.groupby('scene_id').ngroup() // scenes_per_shard
shard_df['shard_num'] = shard_df['shard_num'].clip(upper=num_shards - 1)

# Write shards as CSV
shards_dir.mkdir(parents=True, exist_ok=True)
# Remove old shard files
import glob
for old_shard in glob.glob(str(shards_dir / "shard_*.csv")):
    Path(old_shard).unlink()
for old_shard in glob.glob(str(shards_dir / "shard_*.txt")):
    Path(old_shard).unlink()

for shard_num in range(num_shards):
    shard_data = shard_df[shard_df['shard_num'] == shard_num][['sample_id', 'scene_id', 'type']]
    if len(shard_data) > 0:
        shard_file = shards_dir / f"shard_{shard_num:03d}.csv"
        shard_data.to_csv(shard_file, index=False, header=True)
        print(f"Created {shard_file} with {len(shard_data)} samples")

# Also create scene_ids file for compatibility
scene_ids_file = shards_dir / "all_scene_ids.txt"
unique_scenes_sorted = sorted(unique_scenes)
with open(scene_ids_file, 'w') as f:
    for scene_id in unique_scenes_sorted:
        f.write(f"{scene_id}\n")
print(f"Created {scene_ids_file} with {len(unique_scenes_sorted)} unique scene IDs")

# Write number of shards
actual_shards = len([f for f in shards_dir.glob("shard_*.csv")])
(shards_dir / "num_shards.txt").write_text(str(actual_shards))
print(f"Created {actual_shards} shards")
PYTHON_SCRIPT

exit \$?
EOF
    chmod +x "${CREATE_SHARDS_SCRIPT}"
    
    # Submit shard creation job (depends on add_sample_id)
    echo "  Submitting job to create shards..."
    SHARDS_OUTPUT=$(bsub -J "create_shards" \
        -w "done(${ADD_SAMPLE_ID_JOB_ID})" \
        -o "${LOG_DIR}/create_shards.%J.out" \
        -e "${LOG_DIR}/create_shards.%J.err" \
        -n 1 \
        -R "rusage[mem=1000]" \
        -W 00:10 \
        -q hpc \
        < "${CREATE_SHARDS_SCRIPT}")
    
    CREATE_SHARDS_JOB_ID=$(echo "${SHARDS_OUTPUT}" | grep -oP '(?<=Job <)\d+(?=>)' || echo "")
    
    if [ -z "${CREATE_SHARDS_JOB_ID}" ]; then
        echo "ERROR: Failed to submit create_shards job" >&2
        exit 1
    fi
    
    echo "  Submitted: Job ${CREATE_SHARDS_JOB_ID}"
    CREATE_SHARDS_JOB_ID="${CREATE_SHARDS_JOB_ID}"
    
    # We'll read ACTUAL_SHARDS from file after job completes, but for now estimate
    ACTUAL_SHARDS="${NUM_SHARDS}"
else
    # No dependency needed - create shards now
    SCENE_IDS_FILE="${SHARDS_DIR}/all_scene_ids.txt"
    
    if [ -n "${MANIFEST_WITH_IDS}" ]; then
        echo "Creating shards with sample_id,scene_id,type from manifest..."
        
        # Activate conda environment and use its Python
        # Set MKL_INTERFACE_LAYER before conda activation (required by conda env)
        export MKL_INTERFACE_LAYER=LP64
        if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
            conda activate imginav || conda activate scenefactor || {
                echo "ERROR: Failed to activate conda environment" >&2
                exit 1
            }
        fi
        
        # Use Python from conda environment to extract sample_id, scene_id, and type columns and create shards
        python <<PYTHON_SCRIPT
import pandas as pd
import sys
from pathlib import Path

manifest_path = "${MANIFEST_WITH_IDS}"
shards_dir = Path("${SHARDS_DIR}")
num_shards = ${NUM_SHARDS}

# Load manifest
df = pd.read_csv(manifest_path, low_memory=False)
df.columns = df.columns.str.strip()

# Check required columns
required_cols = ['sample_id', 'scene_id']
if 'type' not in df.columns:
    print("WARNING: 'type' column not found in manifest, will use empty string", file=sys.stderr)
    df['type'] = ''

# Extract sample_id, scene_id, type
shard_df = df[['sample_id', 'scene_id', 'type']].copy()
shard_df = shard_df.dropna(subset=['sample_id', 'scene_id'])

# Get unique scene IDs for splitting
unique_scenes = shard_df['scene_id'].unique()
num_scenes = len(unique_scenes)
scenes_per_shard = max(1, num_scenes // num_shards)

# Group by scene_id and assign to shards
shard_df['shard_num'] = shard_df.groupby('scene_id').ngroup() // scenes_per_shard
shard_df['shard_num'] = shard_df['shard_num'].clip(upper=num_shards - 1)

# Write shards
shards_dir.mkdir(parents=True, exist_ok=True)
# Remove old shard files
import glob
for old_shard in glob.glob(str(shards_dir / "shard_*.csv")):
    Path(old_shard).unlink()
for old_shard in glob.glob(str(shards_dir / "shard_*.txt")):
    Path(old_shard).unlink()

for shard_num in range(num_shards):
    shard_data = shard_df[shard_df['shard_num'] == shard_num][['sample_id', 'scene_id', 'type']]
    if len(shard_data) > 0:
        shard_file = shards_dir / f"shard_{shard_num:03d}.csv"
        shard_data.to_csv(shard_file, index=False, header=True)
        print(f"Created {shard_file} with {len(shard_data)} samples")

# Also create scene_ids file for compatibility
scene_ids_file = shards_dir / "all_scene_ids.txt"
unique_scenes_sorted = sorted(unique_scenes)
with open(scene_ids_file, 'w') as f:
    for scene_id in unique_scenes_sorted:
        f.write(f"{scene_id}\n")
print(f"Created {scene_ids_file} with {len(unique_scenes_sorted)} unique scene IDs")
PYTHON_SCRIPT
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to create shards from manifest" >&2
            exit 1
        fi
        
        # Count shards created
        ACTUAL_SHARDS=$(ls -1 "${SHARDS_DIR}"/shard_*.csv 2>/dev/null | wc -l)
        if [ "${ACTUAL_SHARDS}" -eq 0 ]; then
            echo "ERROR: No shards were created" >&2
            exit 1
        fi
    else
        echo "Discovering scene IDs from metadata..."
        
        METADATA_DIR="${DATASET_ROOT}/metadata/scenes"
        
        if [ ! -d "${METADATA_DIR}" ]; then
            echo "ERROR: Metadata directory not found: ${METADATA_DIR}" >&2
            exit 1
        fi
        
        # Extract scene IDs from JSON filenames
        find "${METADATA_DIR}" -name "*.json" -printf "%f\n" | sed 's/\.json$//' | sort -u > "${SCENE_IDS_FILE}"
    fi
    
    TOTAL_SCENES=$(wc -l < "${SCENE_IDS_FILE}")
    echo "  Found ${TOTAL_SCENES} scenes"
    
    if [ "${TOTAL_SCENES}" -eq 0 ]; then
        echo "ERROR: No scene IDs found" >&2
        exit 1
    fi
    
    # Create shards
    echo ""
    echo "Creating ${NUM_SHARDS} shards..."
    
    # Remove old shard files (both .txt and .csv)
    rm -f "${SHARDS_DIR}"/shard_*.txt "${SHARDS_DIR}"/shard_*.csv
    
    # Count shards created by Python script
    ACTUAL_SHARDS=$(ls -1 "${SHARDS_DIR}"/shard_*.csv 2>/dev/null | wc -l)
    if [ "${ACTUAL_SHARDS}" -eq 0 ]; then
        echo "ERROR: No shards were created" >&2
        exit 1
    fi
    
    echo "  Created ${ACTUAL_SHARDS} shards"
    SCENES_PER_SHARD=$((TOTAL_SCENES / ACTUAL_SHARDS))
    echo "  ~${SCENES_PER_SHARD} scenes per shard"
    CREATE_SHARDS_JOB_ID=""
fi


# =============================================================================
# WRITE CONFIG FILE (for job array to read)
# =============================================================================
CONFIG_FILE="${SHARDS_DIR}/clean_config.sh"
cat > "${CONFIG_FILE}" <<EOF
# Auto-generated config for clean_dataset job array
DATASET_ROOT="${DATASET_ROOT}"
SCRIPTS_DIR="${SCRIPTS_DIR}"
SHARDS_DIR="${SHARDS_DIR}"
OUTPUT_DIR="${OUTPUT_DIR}"

# Quality thresholds (for layouts)
MIN_PIXELS="${MIN_PIXELS}"
MAX_BLACK_FRACTION="${MAX_BLACK_FRACTION}"
MIN_CONTENT_FRACTION="${MIN_CONTENT_FRACTION}"

# POV palette checking (manifest with sample_id)
MANIFEST_PATH="${MANIFEST_WITH_IDS}"
CHECK_POV_PALETTE="${CHECK_POV_PALETTE}"
POV_COLOR_TOLERANCE="${POV_COLOR_TOLERANCE}"
POV_MIN_MATCH_RATIO="${POV_MIN_MATCH_RATIO}"
PALETTE_COLOR_TOLERANCE="${PALETTE_COLOR_TOLERANCE}"
EOF

echo ""
echo "Config written to: ${CONFIG_FILE}"

# =============================================================================
# SUBMIT JOBS
# =============================================================================
if [ "${DRY_RUN}" = "1" ]; then
    echo ""
    echo "[DRY RUN] Would submit:"
    echo "  - Job array: clean_dataset[1-${ACTUAL_SHARDS}]"
    echo "  - Merge job: merge_rejections (after array completes)"
    echo ""
    echo "Shard files created in: ${SHARDS_DIR}"
    exit 0
fi

echo ""
echo "Submitting job array [1-${ACTUAL_SHARDS}]..."

# Submit the array job with dependency on create_shards if needed
if [ -n "${CREATE_SHARDS_JOB_ID}" ]; then
    echo "  Array job will start after shards are created (job ${CREATE_SHARDS_JOB_ID})"
    ARRAY_OUTPUT=$(bsub -J "clean_dataset[1-${ACTUAL_SHARDS}]" \
        -w "done(${CREATE_SHARDS_JOB_ID})" \
        -o "${LOG_DIR}/clean_dataset.%J.%I.out" \
        -e "${LOG_DIR}/clean_dataset.%J.%I.err" \
        -n 2 \
        -R "rusage[mem=4000]" \
        -W 01:00 \
        -q hpc \
        < "${HPC_SCRIPTS_DIR}/run_clean_dataset.sh")
else
    ARRAY_OUTPUT=$(bsub -J "clean_dataset[1-${ACTUAL_SHARDS}]" \
        -o "${LOG_DIR}/clean_dataset.%J.%I.out" \
        -e "${LOG_DIR}/clean_dataset.%J.%I.err" \
        -n 2 \
        -R "rusage[mem=4000]" \
        -W 01:00 \
        -q hpc \
        < "${HPC_SCRIPTS_DIR}/run_clean_dataset.sh")
fi

# Extract job ID
ARRAY_JOB_ID=$(echo "${ARRAY_OUTPUT}" | grep -oP '(?<=Job <)\d+(?=>)' || echo "")

if [ -z "${ARRAY_JOB_ID}" ]; then
    echo "ERROR: Failed to extract job ID from bsub output" >&2
    echo "Output was: ${ARRAY_OUTPUT}" >&2
    exit 1
fi

echo "  Submitted: Job ${ARRAY_JOB_ID}"
echo "  Logs: ${LOG_DIR}/clean_dataset.${ARRAY_JOB_ID}.*.out"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=========================================="
echo "Job Submitted"
echo "=========================================="
echo "Array job: ${ARRAY_JOB_ID} (${ACTUAL_SHARDS} tasks)"
echo ""
echo "Monitor with:"
echo "  bjobs -A ${ARRAY_JOB_ID}"
echo ""
echo "Output will be in:"
echo "  ${OUTPUT_DIR}/rejections_shard_*.csv"
echo ""
echo "After completion, merge with:"
echo "  head -1 ${OUTPUT_DIR}/rejections_shard_000.csv > ${OUTPUT_DIR}/rejections.csv"
echo "  tail -n +2 -q ${OUTPUT_DIR}/rejections_shard_*.csv >> ${OUTPUT_DIR}/rejections.csv"
echo "=========================================="