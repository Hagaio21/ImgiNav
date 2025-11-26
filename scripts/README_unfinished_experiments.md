# Unfinished Experiments Management Scripts

This set of scripts helps you automatically identify and launch unfinished experiments.

## Why 288/300 instead of 288/1000?

Some experiments have different target epochs configured in their YAML files. For example:
- `diff_clip_regular_both_medium_all` has `epochs_target: 300` in its config
- Most other experiments have `epochs_target: 1000`

The scripts automatically read the target epochs from each experiment's config file, so you'll see the correct target for each experiment.

## Scripts Overview

### 1. `list_unfinished_experiments.sh`
Lists all unfinished experiments by checking metrics CSV files.

**Usage:**
```bash
# Use defaults (scans /work3/s233249/ImgiNav/experiments/clip, default target 1000)
bash scripts/list_unfinished_experiments.sh

# Custom base directory
bash scripts/list_unfinished_experiments.sh /path/to/experiments

# Custom base directory and default target epochs
bash scripts/list_unfinished_experiments.sh /path/to/experiments 500
```

**Output:**
- Shows progress as it scans each experiment
- Lists unfinished experiments in format: `<exp_name> - <current_epoch>/<target_epochs>`
- Includes experiments that haven't started (no metrics CSV)

### 2. `find_experiment_config.sh`
Helper script to find the config file for a given experiment name.

**Usage:**
```bash
bash scripts/find_experiment_config.sh <experiment_name> [base_dir]
```

### 3. `launch_unfinished_experiments.sh`
Automatically launches all unfinished experiments.

**Usage:**
```bash
# Launch all unfinished experiments
bash scripts/launch_unfinished_experiments.sh

# Custom directories
bash scripts/launch_unfinished_experiments.sh <base_dir> <default_target> <repo_dir> <dry_run>

# Dry run (see what would be launched without actually launching)
bash scripts/launch_unfinished_experiments.sh /work3/s233249/ImgiNav/experiments/clip 1000 /work3/s233249/ImgiNav/ImgiNav true
```

**What it does:**
1. Scans for unfinished experiments
2. Finds their config files
3. Submits them to the HPC queue using `bsub`
4. Uses the same job submission pattern as other launch scripts

### 4. `auto_launch_unfinished.sh` (Recommended)
Interactive wrapper that shows you what will be launched and asks for confirmation.

**Usage:**
```bash
# Interactive mode (recommended)
bash scripts/auto_launch_unfinished.sh

# Custom directories
bash scripts/auto_launch_unfinished.sh <base_dir> <repo_dir>
```

**What it does:**
1. Shows you the list of unfinished experiments
2. Asks for confirmation
3. Launches them all automatically

## Quick Start

To automatically launch all unfinished experiments:

```bash
bash scripts/auto_launch_unfinished.sh
```

This will:
1. Show you what experiments are unfinished
2. Ask for confirmation
3. Launch them all

## Examples

### List unfinished experiments
```bash
$ bash scripts/list_unfinished_experiments.sh
================================================================================
Unfinished Experiments Scanner
================================================================================
Base directory: /work3/s233249/ImgiNav/experiments/clip
Default target epochs: 1000

Found 15 experiment directories

  diff_clip_regular_both_medium_all: 288/300 (unfinished)
  diff_clip_regular_both_small_all: 119/1000 (unfinished)
  diff_clip_regular_rooms_large_down_bottleneck: 849/1000 (unfinished)
  diff_clip_regular_rooms_large_down_bottleneck_text_only: No metrics CSV (not started)
  ...

================================================================================
SUMMARY
================================================================================
Total unfinished experiments: 5

Unfinished experiments:
--------------------------------------------------------------------------------
  diff_clip_regular_both_medium_all - 288/300
  diff_clip_regular_both_small_all - 119/1000
  diff_clip_regular_rooms_large_down_bottleneck - 849/1000
  diff_clip_regular_rooms_large_down_bottleneck_text_only - 0/1000
  ...
================================================================================
```

### Launch unfinished experiments (dry run)
```bash
$ bash scripts/launch_unfinished_experiments.sh /work3/s233249/ImgiNav/experiments/clip 1000 /work3/s233249/ImgiNav/ImgiNav true
...
[DRY RUN] Would launch: diff_clip_regular_both_medium_all
[DRY RUN] Command: bsub ... bash run_train_diff_clip.sh experiments/diffusion/clip/regular/both_medium_all.yaml
...
```

### Launch unfinished experiments (actual)
```bash
$ bash scripts/auto_launch_unfinished.sh
...
Do you want to launch these experiments? (yes/no): yes

Step 2: Launching experiments...
...
✓ Submitted successfully
...
```

## Notes

- Experiments are submitted to the `gpuv100` queue with 24 hour walltime
- Each job uses 4 CPUs, 1 GPU, and 8000MB memory
- Jobs automatically resume from the latest checkpoint (using `--resume` flag)
- The scripts handle experiments that haven't started yet (no metrics CSV)
- Target epochs are read from each experiment's config file automatically

