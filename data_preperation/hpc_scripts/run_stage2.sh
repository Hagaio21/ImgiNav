#!/bin/bash
#BSUB -J stage2_split2rooms[1-10]
#BSUB -o /zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/logs/stage2_split2rooms.%I.%J.out
#BSUB -e /zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/logs/stage2_split2rooms.%I.%J.err
#BSUB -n 10
#BSUB -R "rusage[mem=4000]"
#BSUB -W 03:00
#BSUB -q hpc

set -euo pipefail

SCENES_ROOT="/work3/s233249/ImgiNav/datasets/scenes"
TAXONOMY_FILE="/zhome/62/5/203350/ws/ImgiNav/config/taxonomy.json"
PYTHON_SCRIPT="/zhome/62/5/203350/ws/ImgiNav/data_preperation/stage2_split2rooms.py"
MANIFEST_DIR="/zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/manifests"

IDX=$((LSB_JOBINDEX - 1))
SHARD_MANIFEST="${MANIFEST_DIR}/stage2_manifest_shard$(printf "%03d" ${IDX}).csv"

echo "Running task ${LSB_JOBINDEX}/10 → shard ${IDX}"

# Create manifest for this shard if it doesn't exist
if [ ! -s "${SHARD_MANIFEST}" ]; then
  echo "Creating manifest for shard ${IDX}..."
  
  TMPDIR_LOCAL="${TMPDIR:-/tmp}"
  ALL_LIST="${TMPDIR_LOCAL}/all_parquet_files.$$"
  SHARD_PREFIX="${TMPDIR_LOCAL}/parquet_shard_"
  
  find "${SCENES_ROOT}" -type f -name "*_sem_pointcloud.parquet" -print | sort > "${ALL_LIST}"
  
  TOTAL_FILES=$(wc -l < "${ALL_LIST}")
  echo "Found ${TOTAL_FILES} total parquet files"
  
  split -d -n l/10 "${ALL_LIST}" "${SHARD_PREFIX}"
  
  SHARD_TXT="${SHARD_PREFIX}$(printf "%02d" ${IDX})"
  
  # Create manifest with correct column name
  echo "scene_id,scene_parquet,npz_path,scene_info_path" > "${SHARD_MANIFEST}"
  
  while IFS= read -r parquet_file; do
    scene_id=$(basename "$(dirname "${parquet_file}")")
    scene_dir=$(dirname "${parquet_file}")
    npz_file="${scene_dir}/${scene_id}_sem_pointcloud.npz"
    info_file="${scene_dir}/${scene_id}_scene_info.json"
    
    echo "${scene_id},${parquet_file},${npz_file},${info_file}" >> "${SHARD_MANIFEST}"
  done < "${SHARD_TXT}"
  
  rm -f "${ALL_LIST}" "${SHARD_PREFIX}"*
  echo "Created manifest: ${SHARD_MANIFEST}"
fi

echo "Using manifest: ${SHARD_MANIFEST}"
echo "Sample entries:"
head -3 "${SHARD_MANIFEST}"

# Conda activation
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate scenefactor || true
fi

# Run Stage 2 with manifest
python "${PYTHON_SCRIPT}" \
  --in_dir "${SCENES_ROOT}" \
  --glob "*_sem_pointcloud.parquet" \
  --taxonomy "${TAXONOMY_FILE}" \
  --inplace \
  --manifest "${SHARD_MANIFEST}" \
  --compute-frames

echo "Task ${LSB_JOBINDEX} completed successfully"