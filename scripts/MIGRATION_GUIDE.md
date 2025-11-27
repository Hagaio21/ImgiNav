# Checkpoint Migration Guide

This guide explains how to migrate old checkpoints to work with the refactored codebase.

## What Changed?

### 1. Projection Components
- **Old**: `clip_projections.text_proj.0.weight` (Sequential layers)
- **New**: `clip_projections.text_proj.proj.0.weight` (TextProjection component)

### 2. Encoder/Decoder Classes
- **Old**: `Encoder` with `variational=True` flag
- **New**: `VAEEncoder` subclass (no flag needed)
- **Old**: `Decoder` handles both deterministic and VAE modes
- **New**: `VAEDecoder` subclass for VAE-specific handling

### 3. Model Types
- **Old**: `Autoencoder` with `variational=True` encoder
- **New**: `VAE` subclass of `Autoencoder`

## Migration Scripts

### Single Checkpoint Migration

```bash
python scripts/migrate_projections.py <checkpoint_path> [--output <output_path>] [--dry-run]
```

**Examples:**
```bash
# Dry run (see what would change)
python scripts/migrate_projections.py checkpoints/vae_checkpoint.pt --dry-run

# Migrate to new file
python scripts/migrate_projections.py checkpoints/vae_checkpoint.pt --output checkpoints/vae_checkpoint_migrated.pt

# Migrate in-place (overwrite original)
python scripts/migrate_projections.py checkpoints/vae_checkpoint.pt
```

### Batch Migration

For migrating multiple checkpoints:

```bash
# Find and migrate all checkpoints in a directory
python scripts/migrate_all_checkpoints.py <directory> [--pattern "*.pt"] [--dry-run] [--backup]
```

**Examples:**
```bash
# Dry run on all checkpoints
python scripts/migrate_all_checkpoints.py checkpoints/ --dry-run

# Migrate all checkpoints with backup
python scripts/migrate_all_checkpoints.py checkpoints/ --backup

# Migrate only best checkpoints
python scripts/migrate_all_checkpoints.py checkpoints/ --pattern "*_best.pt"
```

### Verify Migrated Checkpoint

After migration, verify the checkpoint can be loaded:

```bash
python scripts/verify_migrated_checkpoint.py <checkpoint_path>
```

## What Gets Migrated?

### State Dict Keys
- `clip_projections.text_proj.*` → `clip_projections.text_proj.proj.*`
- `clip_projections.pov_proj.*` → `clip_projections.pov_proj.proj.*`
- `clip_projections.latent_proj.*` → `clip_projections.latent_proj.proj.*`

### Config Structure
- Adds `type: "TextProjection"` to text_projection config
- Adds `type: "ImageProjection"` to image_projection config
- Adds `type: "LatentProjection"` to latent_projection config
- Adds `type: "CLIPProjections"` to clip_projection config
- Migrates `encoder.variational=True` → `encoder.type="VAEEncoder"`
- Migrates `decoder` → `decoder.type="VAEDecoder"` (if used with VAE)
- Migrates `type: "Autoencoder"` → `type: "VAE"` (if encoder is variational)

## Migration Workflow

1. **Backup your checkpoints** (recommended):
   ```bash
   cp -r checkpoints/ checkpoints_backup/
   ```

2. **Test migration on a single checkpoint**:
   ```bash
   python scripts/migrate_projections.py checkpoints/test_checkpoint.pt --dry-run
   python scripts/migrate_projections.py checkpoints/test_checkpoint.pt --output checkpoints/test_checkpoint_migrated.pt
   python scripts/verify_migrated_checkpoint.py checkpoints/test_checkpoint_migrated.pt
   ```

3. **Migrate all checkpoints**:
   ```bash
   python scripts/migrate_all_checkpoints.py checkpoints/ --backup
   ```

4. **Verify a sample of migrated checkpoints**:
   ```bash
   python scripts/verify_migrated_checkpoint.py checkpoints/vae_checkpoint_migrated.pt
   python scripts/verify_migrated_checkpoint.py checkpoints/diffusion_checkpoint_migrated.pt
   ```

## Troubleshooting

### Checkpoint verification fails
- Check that the checkpoint has a valid `config` key
- Ensure all required dependencies are installed
- Check that the model type matches the config

### Migration script reports "No migration needed"
- The checkpoint may already be in the new format
- Or it may not have the components that need migration
- This is normal for some checkpoints

### State dict keys don't match
- The migration script uses `strict=False` when loading
- Some keys may be skipped if shapes don't match (this is expected)
- CLIP projection keys may be skipped if not needed

## Notes

- Migration is **idempotent**: running it multiple times is safe
- The migration script preserves all original checkpoint data
- Backups are recommended before batch migration
- The `--dry-run` flag shows what would change without modifying files

