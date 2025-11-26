# Smart Launch System for Unfinished Experiments

This system automatically manages and launches unfinished experiments, prioritizing those that are almost complete.

## Overview

The system consists of three main components:

1. **`update_unfinished_list.sh`** - Scans experiments and updates:
   - `unfinished.txt` - Central list of unfinished experiments (sorted by completion %)
   - `statistics.txt` - Per-experiment statistics file

2. **`launch_from_unfinished.sh`** - Launches experiments from `unfinished.txt`, prioritizing almost-done ones

3. **`auto_launch_smart.sh`** - Convenience wrapper that does both steps

## File Formats

### `unfinished.txt`
Located in the experiments base directory (e.g., `/work3/s233249/ImgiNav/experiments/clip/unfinished.txt`)

Format:
```
# Unfinished experiments list
# Format: exp_name|current_epoch|target_epochs|completion_pct|status|config_path
# Updated: 2024-01-15 10:30:00 UTC
#
diff_clip_regular_rooms_large_down_bottleneck|849|1000|84.90|unfinished|experiments/diffusion/clip/regular_rooms/large_down_bottleneck.yaml
diff_clip_regular_both_medium_all|288|300|96.00|unfinished|experiments/diffusion/clip/regular/both_medium_all.yaml
...
```

Experiments are sorted by completion percentage (highest first), so almost-done experiments appear at the top.

### `statistics.txt`
Located in each experiment directory (e.g., `/work3/s233249/ImgiNav/experiments/clip/diff_clip_regular_rooms_large_down_bottleneck/statistics.txt`)

Format:
```
experiment_name=diff_clip_regular_rooms_large_down_bottleneck
status=unfinished
current_epoch=849
target_epochs=1000
completion_percentage=84.90
config_path=experiments/diffusion/clip/regular_rooms/large_down_bottleneck.yaml
last_updated=2024-01-15 10:30:00 UTC
```

Status values:
- `not_started` - No metrics CSV exists
- `unfinished` - Has metrics CSV but current_epoch < target_epochs
- `finished` - current_epoch >= target_epochs

## Usage

### Quick Start (Recommended)

Launch experiments that are 90%+ complete (max 5 at a time):
```bash
bash scripts/auto_launch_smart.sh
```

### Step-by-Step

#### 1. Update the unfinished list
```bash
# Update unfinished.txt and all statistics.txt files
bash scripts/update_unfinished_list.sh

# Custom directories
bash scripts/update_unfinished_list.sh /path/to/experiments 1000 /path/to/repo
```

#### 2. Launch from unfinished.txt
```bash
# Launch experiments that are 90%+ complete (max 5 jobs)
bash scripts/launch_from_unfinished.sh

# Custom settings
bash scripts/launch_from_unfinished.sh <base_dir> <repo_dir> <min_completion> <max_jobs> <dry_run>

# Examples:
# Launch experiments 80%+ complete, max 10 jobs
bash scripts/launch_from_unfinished.sh /work3/s233249/ImgiNav/experiments/clip /work3/s233249/ImgiNav/ImgiNav 80 10 false

# Dry run (see what would be launched)
bash scripts/launch_from_unfinished.sh /work3/s233249/ImgiNav/experiments/clip /work3/s233249/ImgiNav/ImgiNav 90 5 true
```

### Advanced Usage

#### Launch only almost-done experiments (95%+)
```bash
bash scripts/launch_from_unfinished.sh /work3/s233249/ImgiNav/experiments/clip /work3/s233249/ImgiNav/ImgiNav 95 3 false
```

#### Launch all unfinished (no minimum completion)
```bash
bash scripts/launch_from_unfinished.sh /work3/s233249/ImgiNav/experiments/clip /work3/s233249/ImgiNav/ImgiNav 0 20 false
```

#### Update list and launch in one command
```bash
bash scripts/auto_launch_smart.sh /work3/s233249/ImgiNav/experiments/clip /work3/s233249/ImgiNav/ImgiNav 90 5
```

## Workflow

### Recommended Daily Workflow

1. **Morning**: Update the list and launch high-priority experiments
   ```bash
   bash scripts/auto_launch_smart.sh
   ```

2. **Throughout the day**: Re-run to launch more as experiments progress
   ```bash
   # Update list
   bash scripts/update_unfinished_list.sh
   
   # Launch more if any are now 90%+ complete
   bash scripts/launch_from_unfinished.sh
   ```

3. **Evening**: Check what's remaining
   ```bash
   cat /work3/s233249/ImgiNav/experiments/clip/unfinished.txt
   ```

### Automated Scheduling (Optional)

You can set up a cron job to automatically update the list:

```bash
# Add to crontab (crontab -e)
# Update unfinished list every 6 hours
0 */6 * * * /work3/s233249/ImgiNav/ImgiNav/scripts/update_unfinished_list.sh
```

## Examples

### Example 1: Update and view unfinished list
```bash
$ bash scripts/update_unfinished_list.sh
Updating unfinished experiments list...
Base directory: /work3/s233249/ImgiNav/experiments/clip
Output file: /work3/s233249/ImgiNav/experiments/clip/unfinished.txt

Scanning 15 experiments...

Updated 5 unfinished experiments
Output written to: /work3/s233249/ImgiNav/experiments/clip/unfinished.txt

Top 5 unfinished experiments (by completion %):
  diff_clip_regular_both_medium_all                     288/ 300 ( 96.0%) [unfinished]
  diff_clip_regular_rooms_large_down_bottleneck        849/1000 ( 84.9%) [unfinished]
  diff_clip_regular_both_small_all                     119/1000 ( 11.9%) [unfinished]
  diff_clip_regular_rooms_large_down_bottleneck_text_only   0/1000 (  0.0%) [not_started]
  diff_clip_regular_rooms_medium_down_bottleneck_text_only   0/1000 (  0.0%) [not_started]
```

### Example 2: Launch almost-done experiments
```bash
$ bash scripts/launch_from_unfinished.sh /work3/s233249/ImgiNav/experiments/clip /work3/s233249/ImgiNav/ImgiNav 90 5
...
Found 2 candidate experiments

Experiments to launch:
--------------------------------------------------------------------------------
  diff_clip_regular_both_medium_all                     288/ 300 ( 96.0%) [unfinished]
  diff_clip_regular_rooms_large_down_bottleneck        849/1000 ( 84.9%) [unfinished]
--------------------------------------------------------------------------------

Launch these 2 experiments? (yes/no): yes

Launching experiments...
Processing: diff_clip_regular_both_medium_all (288/300, 96.00%)
  Submitting job: diff_clip_regular_both_medium_all
  Config: experiments/diffusion/clip/regular/both_medium_all.yaml
  ✓ Submitted successfully
...
```

## Benefits

1. **Prioritization**: Almost-done experiments (90%+) are launched first
2. **Automation**: No need to manually check each experiment
3. **Tracking**: `statistics.txt` files provide per-experiment status
4. **Efficiency**: Limits concurrent jobs to avoid queue overload
5. **Flexibility**: Configurable minimum completion % and max jobs

## Notes

- The system automatically reads target epochs from config files
- Experiments are sorted by completion % (highest first)
- Jobs automatically resume from checkpoints
- `statistics.txt` is updated every time you run `update_unfinished_list.sh`
- The system handles experiments that haven't started yet (0% complete)

