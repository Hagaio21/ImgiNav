# GPU Queue Selection and Experiment Registry

This system automatically selects the best available GPU queue and maintains a comprehensive experiment registry.

## Features

1. **Smart Queue Selection**: Automatically finds the best available GPU queue
2. **Experiment Registry**: Central tracking of all experiments with status, job IDs, and queues
3. **Queue Monitoring**: Checks queue availability before each submission
4. **Interactive Queue Filtering**: Automatically excludes interactive/debug queues

## Queue Selection

### `get_best_gpu_queue.sh`

Automatically selects the best GPU queue by:
- Parsing `bqueue -u $USER` output
- Filtering out interactive queues (containing "interactive", "int", or "debug")
- Prioritizing queues with fewer pending jobs
- Preferring certain queue types (gpuv100 > gpua100 > others)

**Usage:**
```bash
# Get best queue for current user
bash scripts/get_best_gpu_queue.sh

# Get best queue for specific user
bash scripts/get_best_gpu_queue.sh username
```

**Output:**
Returns the queue name (e.g., `gpuv100`, `gpua100`, etc.)

## Experiment Registry

### `experiment_registry.sh`

Maintains a central registry file (`experiment_registry.txt`) with comprehensive experiment tracking.

**Registry Format:**
```
# Experiment Registry
# Format: exp_name|status|current_epoch|target_epochs|completion_pct|config_path|last_updated|queue|job_id|notes
# Updated: 2024-01-15 10:30:00 UTC
#
diff_clip_regular_rooms_large_down_bottleneck|running|849|1000|84.90|experiments/diffusion/clip/regular_rooms/large_down_bottleneck.yaml|2024-01-15 10:30:00 UTC|gpuv100|12345|
diff_clip_regular_both_medium_all|pending|288|300|96.00|experiments/diffusion/clip/regular/both_medium_all.yaml|2024-01-15 10:30:00 UTC|gpuv100|12346|
```

**Fields:**
- `exp_name`: Experiment name
- `status`: `not_started`, `unfinished`, `running`, `pending`, or `finished`
- `current_epoch`: Current training epoch
- `target_epochs`: Target number of epochs
- `completion_pct`: Completion percentage
- `config_path`: Path to experiment config file
- `last_updated`: Timestamp of last update
- `queue`: GPU queue where job is/was running
- `job_id`: LSF job ID (if running/pending)
- `notes`: Optional notes (for future use)

**Usage:**
```bash
# Update registry
bash scripts/experiment_registry.sh

# Custom registry file location
bash scripts/experiment_registry.sh /path/to/registry.txt /path/to/experiments /path/to/repo
```

**What it does:**
1. Scans all experiment directories
2. Reads `statistics.txt` from each experiment
3. Checks for running/pending jobs using `bjobs`
4. Updates registry with current status
5. Preserves job IDs and queue information

## Integration with Launch Scripts

The launch scripts now automatically:
1. Check available queues before each submission
2. Select the best queue
3. Submit to that queue
4. Update the registry with job ID and queue
5. Re-check queue availability for subsequent jobs

### Updated Launch Flow

```bash
# 1. Update unfinished list and registry
bash scripts/update_unfinished_list.sh

# 2. Launch with automatic queue selection
bash scripts/launch_from_unfinished.sh
```

The launch script will:
- Check `bqueue -u $USER` for available queues
- Filter out interactive queues
- Select queue with fewest pending jobs
- Submit job to that queue
- Record job ID and queue in registry
- Re-check queue before next submission

## Queue Selection Logic

The system prioritizes queues as follows:

1. **Fewer pending jobs** = higher priority
2. **Queue type preference**:
   - `gpuv100`: -10 priority (preferred)
   - `gpua100`: -5 priority (good)
   - Other GPU queues: 0 priority (acceptable)
3. **Excludes**:
   - Queues with "interactive" in name
   - Queues with "int" in name
   - Queues with "debug" in name
   - Non-GPU queues

## Examples

### Check available queues
```bash
$ bqueue -u $USER
QUEUE_NAME      PRIO STATUS          MAX JL/U JL/P JL/H NJOBS PEND RUN SUSP
gpuv100         50  Open:Active       -    -    -    -    45   12  33   0
gpua100         60  Open:Active       -    -    -    -    20    5  15   0
gpul40s         40  Open:Active       -    -    -    -    10    2   8   0
interactive     30  Open:Active       -    -    -    -     5    0   5   0

$ bash scripts/get_best_gpu_queue.sh
gpul40s
```

### View experiment registry
```bash
$ cat /work3/s233249/ImgiNav/experiments/clip/experiment_registry.txt
# Experiment Registry
# Format: exp_name|status|current_epoch|target_epochs|completion_pct|config_path|last_updated|queue|job_id|notes
# Updated: 2024-01-15 10:30:00 UTC
#
diff_clip_regular_both_medium_all|running|288|300|96.00|experiments/diffusion/clip/regular/both_medium_all.yaml|2024-01-15 10:30:00 UTC|gpuv100|12345|
diff_clip_regular_rooms_large_down_bottleneck|pending|849|1000|84.90|experiments/diffusion/clip/regular_rooms/large_down_bottleneck.yaml|2024-01-15 10:30:00 UTC|gpuv100|12346|
```

### Launch with automatic queue selection
```bash
$ bash scripts/launch_from_unfinished.sh
...
Checking available GPU queues...
Selected queue: gpul40s

Experiments to launch:
--------------------------------------------------------------------------------
  diff_clip_regular_both_medium_all                     288/ 300 ( 96.0%) [unfinished]
--------------------------------------------------------------------------------

Launch these 1 experiments? (yes/no): yes

Launching experiments...
Processing: diff_clip_regular_both_medium_all (288/300, 96.00%)
  Submitting job: diff_clip_regular_both_medium_all
  Config: experiments/diffusion/clip/regular/both_medium_all.yaml
  Queue: gpul40s
  ✓ Submitted successfully (Job ID: 12347)
```

## Benefits

1. **Automatic Queue Selection**: No need to manually check queue availability
2. **Optimal Resource Usage**: Submits to queues with fewer pending jobs
3. **Comprehensive Tracking**: Registry tracks all experiments and their jobs
4. **Dynamic Adaptation**: Re-checks queue availability between submissions
5. **Job Tracking**: Can see which experiments are running and on which queues

## Registry Maintenance

The registry is automatically updated when you:
- Run `update_unfinished_list.sh`
- Launch experiments via `launch_from_unfinished.sh`

You can also manually update it:
```bash
bash scripts/experiment_registry.sh
```

## Querying the Registry

### Find all running experiments
```bash
grep "|running|" experiment_registry.txt
```

### Find experiments on a specific queue
```bash
grep "|gpuv100|" experiment_registry.txt
```

### Find experiments by completion percentage
```bash
awk -F'|' '$5 >= 90' experiment_registry.txt
```

### Count experiments by status
```bash
cut -d'|' -f2 experiment_registry.txt | grep -v "^#" | sort | uniq -c
```

