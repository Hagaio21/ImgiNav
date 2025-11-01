#!/bin/bash
#BSUB -J manifest_creator
#BSUB -o /zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/logs/manifest_creator.%J.out
#BSUB -e /zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/logs/manifest_creator.%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=2000]"
#BSUB -W 01:00
#BSUB -q hpc

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
SCENES_ROOT="/work3/s233249/ImgiNav/datasets/scenes"
MANIFEST_DIR="/zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts"
N_SHARDS=10

# =============================================================================

echo "Creating scene and room manifests with shards..."

# Conda activation
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate scenefactor || true
fi

# Create temporary files
TMPDIR_LOCAL="${TMPDIR:-/tmp}"
ALL_SCENES="${TMPDIR_LOCAL}/all_scene_entries.$$"
ALL_ROOMS="${TMPDIR_LOCAL}/all_room_entries.$$"
SCENE_SHARD_PREFIX="${TMPDIR_LOCAL}/scene_shard_"
ROOM_SHARD_PREFIX="${TMPDIR_LOCAL}/room_shard_"

echo "Discovering scene-level files..."

# =============================================================================
# COLLECT SCENE ENTRIES: scene_id | parquet_file_path | meta_file_path
# =============================================================================

> "${ALL_SCENES}"
scene_count=0

for scene_dir in "${SCENES_ROOT}"/*; do
  if [ ! -d "${scene_dir}" ]; then
    continue
  fi
  
  scene_id=$(basename "${scene_dir}")
  
  # Look for scene parquet file
  parquet_file="${scene_dir}/${scene_id}_sem_pointcloud.parquet"
  
  # Look for scene meta file
  meta_file=""
  if [ -f "${scene_dir}/${scene_id}_scene_info.json" ]; then
    meta_file="${scene_dir}/${scene_id}_scene_info.json"
  elif [ -f "${scene_dir}/${scene_id}_meta.json" ]; then
    meta_file="${scene_dir}/${scene_id}_meta.json"
  fi
  
  # Only include if parquet exists
  if [ -f "${parquet_file}" ]; then
    echo "${scene_id},${parquet_file},${meta_file}" >> "${ALL_SCENES}"
    scene_count=$((scene_count + 1))
  fi
done

echo "Found ${scene_count} scene entries"

# =======================================================================================
# COLLECT ROOM ENTRIES: scene_id | room_id | room_parquet_file_path | room_meta_file_path
# =======================================================================================

> "${ALL_ROOMS}"
room_count=0

for scene_dir in "${SCENES_ROOT}"/*; do
  if [ ! -d "${scene_dir}" ]; then
    continue
  fi
  
  scene_id=$(basename "${scene_dir}")
  rooms_dir="${scene_dir}/rooms"
  
  if [ ! -d "${rooms_dir}" ]; then
    continue
  fi
  
  # Look for room subdirectories
  for room_dir in "${rooms_dir}"/*; do
    if [ ! -d "${room_dir}" ]; then
      continue
    fi
    
    # Look for room parquet files: <scene_id>_<room_id>.parquet
    for room_parquet in "${room_dir}"/${scene_id}_*.parquet; do
      if [ ! -f "${room_parquet}" ]; then
        continue
      fi
      
      # Extract room_id from filename: <scene_id>_<room_id>.parquet
      filename=$(basename "${room_parquet}")
      # Remove scene_id prefix and .parquet suffix to get room_id
      room_id=$(echo "${filename}" | sed "s/^${scene_id}_//" | sed 's/\.parquet$//')
      
      # Look for corresponding meta file: <scene_id>_<room_id>_meta.json
      room_meta="${room_dir}/${scene_id}_${room_id}_meta.json"
      
      # Check if meta exists, use empty string if not
      if [ ! -f "${room_meta}" ]; then
        room_meta=""
      fi
      
      echo "${scene_id},${room_id},${room_parquet},${room_meta}" >> "${ALL_ROOMS}"
      room_count=$((room_count + 1))
    done
  done
done

echo "Found ${room_count} room entries"

if [ "${scene_count}" -eq 0 ]; then
  echo "ERROR: No scene files found" >&2
  exit 1
fi

if [ "${room_count}" -eq 0 ]; then
  echo "ERROR: No room files found" >&2
  exit 1
fi

# =============================================================================
# CREATE COMPLETE LISTS (before sharding)
# =============================================================================
echo "Creating complete manifest files..."

# Create complete scene list
SCENE_LIST="${MANIFEST_DIR}/scene_list.csv"
echo "scene_id,parquet_file_path,meta_file_path" > "${SCENE_LIST}"
cat "${ALL_SCENES}" >> "${SCENE_LIST}"
echo "Created complete scene list: ${SCENE_LIST} (${scene_count} scenes)"

# Create complete room list  
ROOM_LIST="${MANIFEST_DIR}/room_list.csv"
echo "scene_id,room_id,room_parquet_file_path,room_meta_file_path" > "${ROOM_LIST}"
cat "${ALL_ROOMS}" >> "${ROOM_LIST}"
echo "Created complete room list: ${ROOM_LIST} (${room_count} rooms)"

# =============================================================================
# CREATE SCENE MANIFESTS (sharded)
# =============================================================================
echo "Creating scene manifests for ${N_SHARDS} shards..."

# Split scene entries into shards
split -d -n l/${N_SHARDS} "${ALL_SCENES}" "${SCENE_SHARD_PREFIX}"

# Create scene manifest CSVs
for i in $(seq 0 $((N_SHARDS - 1))); do
  SHARD_NUM=$(printf "%03d" ${i})
  SCENE_SHARD="${SCENE_SHARD_PREFIX}$(printf "%02d" ${i})"
  SCENE_MANIFEST="${MANIFEST_DIR}/scene_manifest_shard${SHARD_NUM}.csv"
  
  echo "Creating scene manifest shard ${SHARD_NUM}..."
  echo "scene_id,parquet_file_path,meta_file_path" > "${SCENE_MANIFEST}"
  
  if [ -s "${SCENE_SHARD}" ]; then
    cat "${SCENE_SHARD}" >> "${SCENE_MANIFEST}"
    echo "  - Scene manifest ${SHARD_NUM}: $(wc -l < "${SCENE_SHARD}") scenes"
  else
    echo "  - Scene manifest ${SHARD_NUM}: 0 scenes (empty shard)"
  fi
done

# =============================================================================
# CREATE ROOM MANIFESTS (sharded)
# =============================================================================
echo "Creating room manifests for ${N_SHARDS} shards..."

# Split room entries into shards
split -d -n l/${N_SHARDS} "${ALL_ROOMS}" "${ROOM_SHARD_PREFIX}"

# Create room manifest CSVs
for i in $(seq 0 $((N_SHARDS - 1))); do
  SHARD_NUM=$(printf "%03d" ${i})
  ROOM_SHARD="${ROOM_SHARD_PREFIX}$(printf "%02d" ${i})"
  ROOM_MANIFEST="${MANIFEST_DIR}/room_manifest_shard${SHARD_NUM}.csv"
  
  echo "Creating room manifest shard ${SHARD_NUM}..."
  echo "scene_id,room_id,room_parquet_file_path,room_meta_file_path" > "${ROOM_MANIFEST}"
  
  if [ -s "${ROOM_SHARD}" ]; then
    cat "${ROOM_SHARD}" >> "${ROOM_MANIFEST}"
    echo "  - Room manifest ${SHARD_NUM}: $(wc -l < "${ROOM_SHARD}") rooms"
  else
    echo "  - Room manifest ${SHARD_NUM}: 0 rooms (empty shard)"
  fi
done

# =============================================================================
# CLEANUP AND SUMMARY
# =============================================================================

# Cleanup temp files
rm -f "${ALL_SCENES}" "${ALL_ROOMS}" "${SCENE_SHARD_PREFIX}"* "${ROOM_SHARD_PREFIX}"*

echo ""
echo "=== MANIFEST CREATION COMPLETE ==="
echo "Complete lists:"
echo "  - Scene list: ${MANIFEST_DIR}/scene_list.csv (${scene_count} scenes)"
echo "  - Room list: ${MANIFEST_DIR}/room_list.csv (${room_count} rooms)"
echo ""
echo "Sharded manifests:"
echo "  - Scene shards: ${MANIFEST_DIR}/scene_manifest_shard*.csv"
echo "  - Room shards: ${MANIFEST_DIR}/room_manifest_shard*.csv"
echo ""
echo "Scene manifest columns: scene_id,parquet_file_path,meta_file_path"
echo "Room manifest columns: scene_id,room_id,room_parquet_file_path,room_meta_file_path"
echo ""
echo "Total scenes distributed across ${N_SHARDS} shards: ${scene_count}"
echo "Total rooms distributed across ${N_SHARDS} shards: ${room_count}"