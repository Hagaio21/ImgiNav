#BSUB -J stage5_room_graphs[1-20]
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs/stage5_room_graphs.%J.%I.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs/stage5_room_graphs.%J.%I.err
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -W 05:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

TAXONOMY_FILE="/work3/s233249/ImgiNav/ImgiNav/config/taxonomy.json"
STAGE5_ROOM_SCRIPT="/work3/s233249/ImgiNav/ImgiNav/data_preperation/stage5_1_build_room_graphs.py"
LAYOUT_MANIFEST="/work3/s233249/ImgiNav/datasets/layouts.csv"
MANIFEST_DIR="/work3/s233249/ImgiNav/datasets/manifest_chunks"

NUM_JOBS=20
JOB_ID=${LSB_JOBINDEX}

echo "Running Stage 5.1 - Room-level graphs (Job ${JOB_ID}/${NUM_JOBS})"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || true
fi

# Create chunk directory
mkdir -p "${MANIFEST_DIR}"

CHUNK_FILE="${MANIFEST_DIR}/chunk_${JOB_ID}.csv"

# Filter out scene-level rows and count
TOTAL_LINES=$(awk -F',' 'NR>1 && $3!="scene"' "${LAYOUT_MANIFEST}" | wc -l)
LINES_PER_JOB=$(( (TOTAL_LINES + NUM_JOBS - 1) / NUM_JOBS ))

START_LINE=$(( (JOB_ID - 1) * LINES_PER_JOB + 1 ))
END_LINE=$(( JOB_ID * LINES_PER_JOB ))

echo "Processing room layouts ${START_LINE} to ${END_LINE} (total: ${TOTAL_LINES})"

# Extract header
head -n 1 "${LAYOUT_MANIFEST}" > "${CHUNK_FILE}"

# Extract only room rows, then get the specific range for this job
awk -F',' 'NR>1 && $3!="scene"' "${LAYOUT_MANIFEST}" | \
  sed -n "${START_LINE},${END_LINE}p" >> "${CHUNK_FILE}"

CHUNK_LINES=$(tail -n +2 "${CHUNK_FILE}" | wc -l)
echo "Chunk contains ${CHUNK_LINES} layouts"

if [ ${CHUNK_LINES} -eq 0 ]; then
  echo "No layouts in this chunk, skipping"
  exit 0
fi

# Run the script on this chunk
python "${STAGE5_ROOM_SCRIPT}" \
  --taxonomy "${TAXONOMY_FILE}" \
  --manifest "${CHUNK_FILE}"

echo "Job ${JOB_ID} completed successfully"