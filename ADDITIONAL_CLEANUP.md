# Additional Cleanup Opportunities

Based on your usage (VAE, CLIP projection, spatial projection, diffusion model, UNet with attention), here are additional cleanups:

---

## Models - Safe to Remove

### 1. `MidBlock` class in `blocks.py` (34 lines) - ❌ DELETE
**Location:** Lines 221-254 in `models/components/blocks.py`
**Reason:** Never used anywhere in the codebase.

```python
# DELETE THIS:
class MidBlock(nn.Module):
    """Middle block with residual + attention."""
    # ... 34 lines
```

---

### 2. Non-VAE `Encoder` class - ⚠️ KEEP (needed as parent)
The base `Encoder` class is needed as parent for `VAEEncoder`. Cannot delete.

### 3. Non-VAE `Decoder` class - ⚠️ KEEP (needed as parent)
The base `Decoder` class is needed as parent for `VAEDecoder`. Cannot delete.

### 4. Non-VAE `Autoencoder` class - ⚠️ KEEP (needed as parent)
The base `Autoencoder` class is needed as parent for `VAE`. Cannot delete.

---

## Losses - Potentially Unused (Verify with your configs)

These losses are registered but may not be used in your training configs:

| Loss Class | Lines | Used In |
|------------|-------|---------|
| `ColorWeightedMSELoss` | ~165 | Only exported, never instantiated |
| `LatentStandardizationLoss` | ~80 | Only exported |
| `LatentStructuralLossAE` | ~165 | Only exported |
| `GradientDifferenceLoss` | ~70 | Only exported |

**Check your training configs** to see which losses you actually use. If you only use:
- `MSELoss` / `SNRWeightedMSELoss` for diffusion
- `KLDLoss` for VAE
- `CLIPLoss` for CLIP projection
- `CompositeLoss` to combine them

Then the others can be removed (~480 lines total).

---

## Data Preparation - Safe to Delete

### Already identified:
| File | Lines | Reason |
|------|-------|--------|
| `path_utils.py` | 347 | Never imported |
| `stage0_build_taxonomy.py` | 385 | Redundant (use common/taxonomy.py) |
| `config_loader.py` | 202 | Never used |

### Additional candidates:
| File | Lines | Check If Used |
|------|-------|---------------|
| `recolor_taxonomy.py` | 217 | Only for re-coloring existing taxonomy |
| `visualize_taxonomy_colors.py` | 254 | Only for visualization |
| `merge_pov_info_shards.py` | 109 | Only for HPC post-processing |
| `create_shards.py` | 99 | Only for HPC job splitting |

---

## Scripts - Verify If Used

These standalone scripts may or may not be needed:

| Script | Lines | Purpose |
|--------|-------|---------|
| `scripts/clean_layout.py` | 350 | Post-process layout images |
| `scripts/navigate.py` | 277 | Navigation visualization |
| `scripts/create_occupancy_grid.py` | 72 | Create occupancy grids |
| `scripts/diffusion_inference.py` | 157 | Standalone inference |

If you don't use these manually, they can be deleted (~856 lines).

---

## Summary: Maximum Cleanup Potential

| Category | Lines |
|----------|-------|
| `MidBlock` removal | 34 |
| Unused losses (if verified) | ~480 |
| Data prep scripts | ~934 |
| Standalone scripts (if unused) | ~856 |
| **Total potential** | **~2,300 lines** |

---

## Recommended Actions

### High Confidence (Safe to do now):
1. Delete `MidBlock` class from `blocks.py`
2. Delete `path_utils.py`, `stage0_build_taxonomy.py`, `config_loader.py`

### Medium Confidence (Check your configs first):
1. Review which loss classes you actually use
2. Delete unused loss classes

### Low Confidence (Verify manually):
1. Check if you use the standalone scripts
2. Check if you use the visualization/utility data prep scripts
