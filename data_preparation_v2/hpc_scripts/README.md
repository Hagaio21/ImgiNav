# HPC Scripts for Data Preparation v2

This directory contains HPC job scripts for running the data preparation pipeline on cluster systems.

## Stage 1 & 2 Array Job

### Files
- `run_stage1_stage2_array.sh` - Main job script that processes scenes in parallel
- `launch_stage1_stage2_array.sh` - Launcher script to submit the array job

### Usage

1. **Update paths in `run_stage1_stage2_array.sh`**:
   - `SCENES_ROOT` - Directory containing original 3D-FRONT scene JSON files
   - `MODEL_DIR` - Directory containing 3D-FUTURE model files
   - `MODEL_INFO` - Path to model_info.json
   - `TAXONOMY_FILE` - Path to taxonomy.json
   - `TEXTURE_DIR` - Path to texture directory (optional)
   - `VALID_SCENES_FILE` - Path to valid_scenes.txt
   - `OUTPUT_GEOMETRY_DIR` - Output directory for Stage 1 (geometry)
   - `OUTPUT_METADATA_DIR` - Output directory for Stage 2 (metadata)
   - `STAGE1_SCRIPT` - Path to stage1_reconstruct_geometry.py
   - `STAGE2_SCRIPT` - Path to stage2_compile_metadata.py

2. **Submit the job**:
   ```bash
   cd data_preparation_v2/hpc_scripts
   bash launch_stage1_stage2_array.sh
   ```

### How it works

- The script reads `valid_scenes.txt` and splits it into 10 shards
- Each array job (1-10) processes one shard:
  1. Extracts its portion of scene IDs from `valid_scenes.txt`
  2. Copies the corresponding scene JSON files to a temporary directory
  3. Runs Stage 1 (geometry reconstruction) on those scenes
  4. Runs Stage 2 (metadata compilation) on those scenes
  5. Cleans up temporary files

### Job Configuration

- **Array size**: 10 parallel jobs (`[1-10]`)
- **Resources per job**:
  - CPUs: 8
  - Memory: 8000 MB
  - Time limit: 10 hours
  - Queue: hpc

### Monitoring

After submission, monitor jobs with:
```bash
bjobs <JOB_ID>           # Check status
bjobs -a <JOB_ID>        # Check all array tasks
bpeek <JOB_ID>[<TASK>]  # View output of specific task
```

Logs are written to: `data_preparation_v2/hpc_scripts/logs/`

### Notes

- Each job processes approximately 1/10 of the scenes from `valid_scenes.txt`
- Scenes are copied to temporary directories to avoid conflicts between parallel jobs
- Both Stage 1 and Stage 2 run sequentially within each job
- Temporary files are automatically cleaned up after processing

