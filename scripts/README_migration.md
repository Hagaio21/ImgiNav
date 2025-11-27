# Projection Component Migration Guide

## Overview

The projection components have been refactored to use a modular architecture with separate classes:
- `TextProjection` - for text/graph embeddings
- `ImageProjection` - for POV/image embeddings  
- `LatentProjection` - for VAE latent features
- `CLIPProjections` - composite class using the above

## Migration for Existing Checkpoints

### What Changed?

**Old Format:**
- `clip_projections.text_proj.0.weight` (Sequential layer)
- `clip_projections.pov_proj.0.weight` (Sequential layer)
- `clip_projections.latent_proj.0.weight` (Sequential layer)

**New Format:**
- `clip_projections.text_proj.proj.0.weight` (TextProjection component)
- `clip_projections.pov_proj.proj.0.weight` (ImageProjection component)
- `clip_projections.latent_proj.proj.0.weight` (LatentProjection component)

### Do I Need to Migrate?

**Short answer:** Migration is **recommended** but not strictly required.

- ✅ **With migration**: Proper structure, better compatibility, cleaner code
- ⚠️ **Without migration**: Code will still work (using `strict=False`), but structure is inconsistent

### How to Migrate

1. **Single checkpoint:**
   ```bash
   python scripts/migrate_projections.py path/to/checkpoint.pt --output path/to/migrated.pt
   ```

2. **Dry run (see what would change):**
   ```bash
   python scripts/migrate_projections.py path/to/checkpoint.pt --dry-run
   ```

3. **Batch migration (multiple checkpoints):**
   Create a text file `checkpoints.txt` with one path per line:
   ```
   outputs/experiment1/checkpoint_100.pt
   outputs/experiment2/checkpoint_200.pt
   outputs/experiment3/checkpoint_300.pt
   ```
   Then run:
   ```bash
   python scripts/migrate_projections.py dummy --batch checkpoints.txt
   ```

4. **In-place migration (overwrite original):**
   ```bash
   python scripts/migrate_projections.py path/to/checkpoint.pt
   ```

### What Gets Migrated?

1. **State Dict Keys:**
   - All `clip_projections.text_proj.*` → `clip_projections.text_proj.proj.*`
   - All `clip_projections.pov_proj.*` → `clip_projections.pov_proj.proj.*`
   - All `clip_projections.latent_proj.*` → `clip_projections.latent_proj.proj.*`

2. **Config Structure:**
   - Adds `type` fields to projection configs
   - Creates `text_projection` and `image_projection` configs if missing
   - Creates `latent_projection` config if missing

### Backward Compatibility

The code maintains backward compatibility:
- Old checkpoints can still be loaded (with warnings)
- Migration is optional but recommended
- New code works with both old and new formats

### Testing Migration

Test the migration logic:
```bash
python scripts/test_migration.py
```

This will verify that:
- State dict keys are correctly renamed
- Config structure is properly updated
- All expected migrations are performed

### Troubleshooting

**Issue:** Migration script says "already migrated" but model still fails to load
- **Solution:** Check if the checkpoint has mixed old/new format keys. Try loading with `strict=False` first.

**Issue:** Some keys are missing after migration
- **Solution:** The migration only affects projection keys. Other keys (encoder, decoder, etc.) remain unchanged.

**Issue:** Config migration fails
- **Solution:** The script will still migrate state dict even if config migration fails. You can manually update the config.

### Notes

- Migration is **idempotent**: Running it multiple times is safe
- Always **backup** your checkpoints before migration
- Migration creates a new checkpoint file (doesn't modify original unless using in-place mode)

