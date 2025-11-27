# Checkpoint Migration Guide

This guide explains how to migrate old checkpoints to the new architecture.

## Overview

The migration script handles:
- Projection component key changes (if applicable)
- Preserving all extra state (optimizer, epoch, training history, etc.)
- Verifying migrated checkpoints can be loaded

## Quick Start

### 1. Inspect Checkpoint

First, inspect your checkpoint to see if migration is needed:

```bash
python scripts/inspect_checkpoint.py checkpoints/vae_clip_checkpoint_best.pt
```

### 2. Backup Checkpoints

Before migrating, backup your original checkpoints:

```bash
# Create backup directory
mkdir -p checkpoints/backups

# Backup checkpoints
cp checkpoints/vae_clip_checkpoint_best.pt checkpoints/backups/
cp checkpoints/diff_clip_regular_rooms_small_down_bottleneck_text_only_checkpoint_best.pt checkpoints/backups/
```

### 3. Dry Run Migration

See what would be migrated without actually migrating:

```bash
python scripts/migrate_checkpoint.py checkpoints/vae_clip_checkpoint_best.pt --dry-run
```

### 4. Migrate Checkpoint

Migrate a checkpoint:

```bash
# Migrate to new file (recommended)
python scripts/migrate_checkpoint.py checkpoints/vae_clip_checkpoint_best.pt --output checkpoints/vae_clip_checkpoint_best_migrated.pt

# Or let it create migrated_<name> automatically
python scripts/migrate_checkpoint.py checkpoints/vae_clip_checkpoint_best.pt
```

### 5. Verify Migrated Checkpoint

Verify that the migrated checkpoint can be loaded:

```bash
python scripts/verify_migrated_checkpoint.py checkpoints/vae_clip_checkpoint_best_migrated.pt
```

## Migration Details

### What Gets Migrated?

1. **Projection Component Keys** (if old format detected):
   - Old: `clip_projections.text_proj.0.weight`
   - New: `clip_projections.text_proj.proj.0.weight`

2. **Preserved State**:
   - All model weights (state_dict)
   - Model config
   - Optimizer state
   - Scheduler state
   - Training history
   - Epoch number
   - Best validation loss
   - Any other extra state

### Migration Process

1. Load old checkpoint
2. Inspect structure and identify migration needs
3. Migrate projection keys (if needed)
4. Preserve all extra state
5. Save migrated checkpoint
6. Verify migrated checkpoint can be loaded

## Examples

### Migrate Both Checkpoints

```bash
# VAE checkpoint
python scripts/migrate_checkpoint.py checkpoints/vae_clip_checkpoint_best.pt \
    --output checkpoints/vae_clip_checkpoint_best_migrated.pt

# Diffusion checkpoint
python scripts/migrate_checkpoint.py \
    checkpoints/diff_clip_regular_rooms_small_down_bottleneck_text_only_checkpoint_best.pt \
    --output checkpoints/diff_clip_regular_rooms_small_down_bottleneck_text_only_checkpoint_best_migrated.pt
```

### Verify Both

```bash
python scripts/verify_migrated_checkpoint.py checkpoints/vae_clip_checkpoint_best_migrated.pt
python scripts/verify_migrated_checkpoint.py checkpoints/diff_clip_regular_rooms_small_down_bottleneck_text_only_checkpoint_best_migrated.pt
```

## Troubleshooting

### Checkpoint Already in New Format

If the script says "No migration needed", your checkpoint is already compatible.

### Migration Fails

If migration fails:
1. Check that the checkpoint file exists and is not corrupted
2. Verify you have write permissions in the output directory
3. Check the error message for specific issues

### Verification Fails

If verification fails:
1. Check that the model classes are available
2. Verify the config structure is valid
3. Check for missing dependencies

## Rollback

If you need to rollback:
1. Use the backup checkpoints in `checkpoints/backups/`
2. Copy them back to the original location

```bash
cp checkpoints/backups/vae_clip_checkpoint_best.pt checkpoints/
```

## Related Scripts

- `scripts/inspect_checkpoint.py` - Inspect checkpoint structure
- `scripts/migrate_checkpoint.py` - Migrate checkpoint
- `scripts/verify_migrated_checkpoint.py` - Verify migrated checkpoint

