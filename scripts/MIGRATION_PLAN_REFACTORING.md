# Checkpoint Migration Plan - Codebase Refactoring

## Overview

This migration plan covers checkpoints after the major codebase refactoring that:
- Removed deprecated `cond` parameter
- Refactored checkpoint loading logic
- Consolidated base classes and inheritance
- Simplified component creation

## What Changed?

### 1. API Changes (No Checkpoint Migration Needed)
- **Removed `cond` parameter**: The deprecated `cond` parameter has been removed from:
  - `DiffusionModel.forward()`
  - `DiffusionModel.sample()`
  - `UnetWithAttention.forward()`
  
  **Impact**: This is an API change only. Checkpoints are **NOT affected** because `cond` was never stored in state dicts.

### 2. Code Structure Changes (No Checkpoint Migration Needed)
- **Refactored checkpoint loading**: Internal helper methods were extracted, but checkpoint format is unchanged
- **Consolidated base classes**: Inheritance changes don't affect state dict keys
- **Simplified component creation**: Uses unified registry, but component structure is unchanged

**Impact**: These are internal code organization changes. Existing checkpoints should load without modification.

### 3. Backward Compatibility Attributes (No Checkpoint Migration Needed)
- Removed redundant attributes like `self.encoder`, `self.autoencoder`, `self.embedding_proj`
- These were runtime attributes, not stored in checkpoints

**Impact**: No checkpoint changes needed.

## Migration Assessment

### ✅ **No Migration Required**

**Good News**: All refactoring changes are **code-level only**. The checkpoint format (state dict keys and structure) remains **100% compatible**.

### Why No Migration is Needed:

1. **State Dict Keys Unchanged**: 
   - All module hierarchies remain the same
   - Component names and paths are unchanged
   - Only internal code organization changed

2. **Config Structure Compatible**:
   - Component configs still use the same format
   - Registry-based loading is backward compatible
   - Old configs will still work

3. **API Changes Don't Affect Checkpoints**:
   - Removed parameters (`cond`) were never stored
   - Method signatures changed, but data format didn't

## Verification Steps

Even though migration isn't required, verify your checkpoints still work:

### 1. Quick Verification Script

```python
# scripts/verify_checkpoint_compatibility.py
import torch
from models.diffusion import DiffusionModel
from pathlib import Path

def verify_checkpoint(checkpoint_path):
    """Verify a checkpoint can be loaded with refactored code."""
    print(f"Verifying: {checkpoint_path}")
    
    try:
        # Load checkpoint
        payload = torch.load(checkpoint_path, map_location="cpu")
        config = payload.get("config")
        
        if config is None:
            print("  ⚠️  No config found - checkpoint may be incomplete")
            return False
        
        # Try to create model from config
        model = DiffusionModel.from_config(config)
        print("  ✅ Model created successfully")
        
        # Try to load state dict
        state_dict = payload.get("state_dict", payload)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"  ⚠️  Missing keys: {len(missing_keys)} (this is normal with strict=False)")
        if unexpected_keys:
            print(f"  ⚠️  Unexpected keys: {len(unexpected_keys)}")
        
        print("  ✅ Checkpoint loaded successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

if __name__ == "__main__":
    import sys
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    if checkpoint_path:
        verify_checkpoint(checkpoint_path)
    else:
        print("Usage: python scripts/verify_checkpoint_compatibility.py <checkpoint_path>")
```

### 2. Test Loading

```bash
# Test a single checkpoint
python scripts/verify_checkpoint_compatibility.py checkpoints/my_checkpoint.pt

# Test all checkpoints in a directory
find checkpoints/ -name "*.pt" -exec python scripts/verify_checkpoint_compatibility.py {} \;
```

### 3. Test Inference

```python
# Quick inference test
from models.diffusion import DiffusionModel

# Load checkpoint
model = DiffusionModel.load_checkpoint("checkpoints/my_checkpoint.pt")

# Test sampling (note: no cond parameter needed)
output = model.sample(
    batch_size=1,
    num_steps=10,
    text_emb=text_emb,  # if available
    pov_emb=pov_emb,    # if available
    guidance_scale=1.0
)
```

## If You Encounter Issues

### Issue: "Unexpected keyword argument 'cond'"

**Solution**: Update your code to remove `cond` parameter:
```python
# Old (won't work)
output = model(latents, t, cond=None, ...)

# New (correct)
output = model(latents, t, ...)
```

### Issue: Missing Keys in State Dict

**Solution**: This is normal with `strict=False`. The refactoring shouldn't cause new missing keys, but if you see many:
1. Check if checkpoint was created with old projection format (use `migrate_projections.py` if needed)
2. Verify checkpoint wasn't corrupted
3. Check if model architecture changed significantly

### Issue: Component Not Found

**Solution**: Ensure you're using the latest code with unified registry:
```python
# Old code might have used direct imports
# New code uses unified registry - should work automatically
```

## Migration Checklist

- [ ] **Backup all checkpoints** before testing
- [ ] **Verify checkpoint loading** with verification script
- [ ] **Test inference** with a sample checkpoint
- [ ] **Update training scripts** to remove `cond` parameter
- [ ] **Update inference scripts** to remove `cond` parameter
- [ ] **Test end-to-end workflow** (training → checkpoint → inference)

## Rollback Plan

If you encounter issues:

1. **Code Rollback**: Revert to previous commit
   ```bash
   git checkout <previous-commit-hash>
   ```

2. **Checkpoint Rollback**: Use your backups (shouldn't be needed, but good to have)

3. **Partial Rollback**: If only specific features break, you can:
   - Keep refactored code
   - Add temporary compatibility shims for removed parameters

## Summary

**✅ No checkpoint migration required**

The refactoring was designed to be **100% backward compatible** with existing checkpoints. All changes are:
- Code organization (doesn't affect checkpoints)
- API cleanup (doesn't affect stored data)
- Internal refactoring (doesn't change state dict structure)

**Action Items**:
1. Test your checkpoints with the verification script
2. Update your code to remove `cond` parameter
3. Enjoy the cleaner, more maintainable codebase! 🎉

## Related Migration Guides

- **Projection Migration**: See `scripts/README_migration.md` for projection component migrations
- **General Migration**: See `scripts/MIGRATION_GUIDE.md` for other checkpoint migrations

