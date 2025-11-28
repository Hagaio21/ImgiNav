# VAE Training Scripts

This directory contains scripts for training VAE models on segmented and textured layouts at 256x256 resolution.

## Prerequisites

Before running the training scripts, ensure you have:

1. **Completed data preparation pipeline** (Stages 1-6)
2. **Generated manifest CSV files** using `collect_manifest.py`:
   - `dataset_v2/manifests/manifest_seg.csv`
   - `dataset_v2/manifests/manifest_tex.csv`
3. **Created embeddings** using `embed_for_training.py`:
   - POV embeddings in `dataset_v2/pov/embeddings_seg/` and `dataset_v2/pov/embeddings_tex/`
   - Graph text embeddings in `dataset_v2/graphs/embeddings/`

## Config Files

The training configs are located in `experiments/autoencoders/v2/`:

- **`vae_seg_256_clip.yaml`** - Segmented layouts VAE (256x256 input)
- **`vae_tex_256_clip.yaml`** - Textured layouts VAE (256x256 input)

Both configs use:
- Input size: 256×256
- Latent space: 16×16×4 (after 4 downsampling steps)
- CLIP loss for alignment with text/POV embeddings
- Precomputed sample weights for balanced training

## Training Scripts

### Individual Training

**Train Segmented VAE:**
```powershell
.\scripts\train_vae_seg.ps1
```

**Train Textured VAE:**
```powershell
.\scripts\train_vae_tex.ps1
```

### Train Both Sequentially

**Train both models:**
```powershell
.\scripts\train_vae_both.ps1
```

This will train the segmented VAE first, then the textured VAE.

## Output

Training outputs will be saved to:
- Segmented: `outputs/autoencoders/v2/vae_seg_256_clip/`
- Textured: `outputs/autoencoders/v2/vae_tex_256_clip/`

Each output directory contains:
- `checkpoints/` - Model checkpoints (latest, best, periodic)
- `samples/` - Sample reconstructions during training
- `{exp_name}_metrics.csv` - Training metrics
- `loss_curves.png` - Loss visualization

## Training Configuration

Both configs use:
- **Epochs**: 150
- **Batch size**: 32
- **Learning rate**: 0.0001
- **Optimizer**: AdamW
- **Early stopping**: Patience 10, min delta 0.00005
- **Mixed precision**: Enabled (AMP)

## Resuming Training

Training automatically resumes from the latest checkpoint if available. To force a fresh start, delete the checkpoint files in the output directory.

## Notes

- The 256×256 input size reduces memory usage compared to 512×512
- Latent resolution is 16×16 (256 / 2^4 = 16)
- Both models use the same architecture, only the input data differs

