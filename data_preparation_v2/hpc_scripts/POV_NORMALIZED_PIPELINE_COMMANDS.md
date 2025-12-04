# POV-Normalized Pipeline Commands (Stage 4 v2 & Stage 5 v2)

This document provides the complete command sequence to run the POV-normalized layout creation, POV rendering, and graph generation pipeline.

## Prerequisites

- Dataset root: `/work3/s233249/ImgiNav/dataset_v2`
- Scripts directory: `/work3/s233249/ImgiNav/ImgiNav/data_preparation_v2`
- HPC scripts directory: `/work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts`

## Pipeline Steps

### Step 1: Stage 4 v2 - POV Rendering + Layout Rotation

This stage:
- Renders POV images with improved camera settings
- Rotates existing layouts to match POV viewing direction
- Generates POV info shards

**Command:**
```bash
cd /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts
./launch_stage4_v2.sh /work3/s233249/ImgiNav/dataset_v2 500
```

**Parameters:**
- First argument: Dataset root path
- Second argument: Number of shards (default: 500)
- Optional extra args can be added (e.g., `--only-graphs`, `--no-layouts`)

**Monitor progress:**
```bash
bjobs -w
```

**Check logs:**
```bash
tail -f /work3/s233249/ImgiNav/dataset_v2/logs/stage4v2/*.out
```

---

### Step 2: Merge POV Info Shards

After all Stage 4 v2 jobs complete, merge the shard files into a single `pov_info.json`.

**Command:**
```bash
cd /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts
bsub < run_merge_pov_info_shards.sh
```

**Optional: Clean shard files after merge:**
```bash
bsub -env "CLEAN=1" < run_merge_pov_info_shards.sh
```

**Check completion:**
```bash
ls -lh /work3/s233249/ImgiNav/dataset_v2/povs/pov_info.json
```

---

### Step 3: Stage 5 v2 - POV-Normalized Graph Generation

This stage generates POV-specific graphs with:
- Descriptive object naming (no numerical suffixes)
- POV-relative spatial relations ("ahead", "to your left")
- Second-person perspective descriptions

**Command:**
```bash
cd /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts
./launch_stage5_v2.sh /work3/s233249/ImgiNav/dataset_v2 100
```

**Parameters:**
- First argument: Dataset root path
- Second argument: Number of shards (default: 100, adjust based on number of scenes)

**Monitor progress:**
```bash
bjobs -w
```

**Check logs:**
```bash
tail -f /work3/s233249/ImgiNav/dataset_v2/logs/stage5v2/*.out
```

---

### Step 4: Collect POV-Normalized Manifest

After Stage 5 v2 completes, collect the manifest CSV files.

**Command:**
```bash
cd /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts
bsub < run_collect_manifest_pov_normalized.sh
```

**Generate only one variant:**
```bash
# Tex only
bsub -env "VARIANT=tex" < run_collect_manifest_pov_normalized.sh

# Seg only
bsub -env "VARIANT=seg" < run_collect_manifest_pov_normalized.sh
```

**Output files:**
- `/work3/s233249/ImgiNav/dataset_v2/manifests/manifest_tex_pov_normalized.csv`
- `/work3/s233249/ImgiNav/dataset_v2/manifests/manifest_seg_pov_normalized.csv`

---

### Step 5: Embed POVs and Graphs for Training

After collecting the manifest, create embeddings for POV images and graph texts. This script automatically detects POV-normalized manifests and creates POV-specific embeddings.

**Command:**
```bash
cd /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts
bsub < run_embed_pov_normalized.sh
```

**Generate embeddings for one variant:**
```bash
# Tex only
bsub -env "VARIANT=tex" < run_embed_pov_normalized.sh

# Seg only
bsub -env "VARIANT=seg" < run_embed_pov_normalized.sh
```

**Output:**
- POV embeddings: `/work3/s233249/ImgiNav/dataset_v2/povs/embeddings_{variant}/{scene_id}_{room_id}_{pov_id}_pov.pt`
- Graph embeddings: `/work3/s233249/ImgiNav/dataset_v2/graphs/embeddings/{scene_id}_{room_id}_{pov_id}_text.pt`
- Updated manifests with `pov_embedding_path` and `graph_embedding_path` columns

**Check completion:**
```bash
# Count POV embeddings
ls /work3/s233249/ImgiNav/dataset_v2/povs/embeddings_seg/*.pt | wc -l

# Count graph embeddings
ls /work3/s233249/ImgiNav/dataset_v2/graphs/embeddings/*.pt | wc -l
```

---

### Step 6: Train VAE CLIP v2

Train a VAE on POV-normalized segmented layouts with CLIP loss. This uses the POV-specific layouts and graphs for better alignment.

**Command:**
```bash
cd /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/vae
bsub < run_train_vae_clip_v2.sh
```

**Config file:**
- `/work3/s233249/ImgiNav/ImgiNav/experiments/autoencoders/v2/vae_clip_v2.yaml`

**Output:**
- Checkpoints: `/work3/s233249/ImgiNav/experiments/v2/autoencoders/vae_clip_v2/checkpoints/`
- Training logs: `/work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/train_vae_clip_v2.*.out`

**Monitor training:**
```bash
tail -f /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/train_vae_clip_v2.*.out
```

---

## Complete Command Sequence (Copy-Paste Ready)

```bash
# Set variables
DATASET_ROOT="/work3/s233249/ImgiNav/dataset_v2"
SCRIPTS_DIR="/work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts"

# Step 1: Launch Stage 4 v2
cd ${SCRIPTS_DIR}
./launch_stage4_v2.sh ${DATASET_ROOT} 500

# Wait for Stage 4 v2 to complete (monitor with: bjobs -w)

# Step 2: Merge POV info shards
bsub < ${SCRIPTS_DIR}/run_merge_pov_info_shards.sh

# Wait for merge to complete

# Step 3: Launch Stage 5 v2
./launch_stage5_v2.sh ${DATASET_ROOT} 100

# Wait for Stage 5 v2 to complete (monitor with: bjobs -w)

# Step 4: Collect manifest
bsub < ${SCRIPTS_DIR}/run_collect_manifest_pov_normalized.sh

# Wait for manifest collection to complete

# Step 5: Embed POVs and graphs
bsub < ${SCRIPTS_DIR}/run_embed_pov_normalized.sh

# Wait for embedding to complete

# Step 6: Train VAE CLIP v2
cd /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/vae
bsub < run_train_vae_clip_v2.sh
```

---

## Quick Status Checks

**Check Stage 4 v2 progress:**
```bash
# Count completed shards
ls /work3/s233249/ImgiNav/dataset_v2/povs/pov_info_shard_*.json | wc -l

# Check for errors
grep -l "ERROR\|FAILED" /work3/s233249/ImgiNav/dataset_v2/logs/stage4v2/*.err
```

**Check Stage 5 v2 progress:**
```bash
# Count POV graphs
find /work3/s233249/ImgiNav/dataset_v2/graphs/jsons -name "*door*_room_graph.json" -o -name "*window*_room_graph.json" | wc -l

# Check for errors
grep -l "ERROR\|FAILED" /work3/s233249/ImgiNav/dataset_v2/logs/stage5v2/*.err
```

---

## Troubleshooting

**If Stage 4 v2 jobs fail:**
- Check GPU availability: `bjobs -w` should show jobs in `RUN` state
- Check memory: Jobs use 2GB memory, ensure nodes have enough
- Check logs: `tail -f /work3/s233249/ImgiNav/dataset_v2/logs/stage4v2/*.err`

**If merge fails:**
- Ensure all Stage 4 v2 shards completed
- Check that shard files exist: `ls /work3/s233249/ImgiNav/dataset_v2/povs/pov_info_shard_*.json`

**If Stage 5 v2 fails:**
- Ensure Stage 4 v2 completed and merge was successful
- Check that POV info exists: `ls /work3/s233249/ImgiNav/dataset_v2/povs/pov_info.json`
- Verify scene metadata exists: `ls /work3/s233249/ImgiNav/dataset_v2/metadata/scenes/*.json | wc -l`

