# ControlNet Setup for New Layouts (UNet48)

This directory contains the ControlNet experiment setup for the 48-channel UNet with new layouts.

## Pipeline Overview

1. **Create Joint Manifest**: Collect layouts, POVs, graphs, and their embeddings
2. **Embed POVs and Graphs**: Create embeddings for POV images and graph texts
3. **Create Layout Embeddings**: Embed layout images using the autoencoder
4. **Create ControlNet Manifest**: Align all embeddings for training
5. **Train ControlNet**: Train the ControlNet model

## Step-by-Step Setup

### Option 1: Automated Pipeline (Recommended)

Launch the complete pipeline with automatic job chaining:

```bash
bsub < data_preparation/hpc_scripts/launch_controlnet_pipeline.sh
```

This will:
1. Submit the first job (create joint manifest)
2. Automatically submit the next job when each completes successfully
3. Chain through all steps automatically

**Pipeline flow:**
- Job 1 → Job 2 → Job 3 (automatic chaining)

Monitor progress:
```bash
bjobs  # Check job status
bpeek <job_id>  # View job output
```

### Option 2: Manual Step-by-Step

If you prefer to run steps manually:

#### Step 1: Create Joint Manifest

```bash
bsub < data_preparation/hpc_scripts/run_create_joint_manifest.sh
```

This creates:
- `/work3/s233249/ImgiNav/datasets/joint_manifest.csv`
- Copies graphs to `datasets/collected/graphs/`
- Copies POVs to `datasets/collected/povs/tex/` and `povs/seg/`

#### Step 2: Embed POVs and Graphs

```bash
bsub < data_preparation/hpc_scripts/run_embed_from_joint_manifest.sh
```

This creates:
- `/work3/s233249/ImgiNav/datasets/joint_manifest_with_embeddings.csv`
- POV embeddings (ResNet18) saved next to POV images
- Graph embeddings (SentenceTransformer) saved next to graph JSON files

#### Step 3: Create Layout Embeddings

If layout embeddings don't exist yet:

```bash
python data_preparation/create_embeddings.py \
    --type layout \
    --manifest /work3/s233249/ImgiNav/datasets/layouts_cleaned.csv \
    --output-manifest /work3/s233249/ImgiNav/datasets/layouts_cleaned_with_latents.csv \
    --autoencoder-config experiments/new_layouts/new_layouts_VAE_64x64_structural_256/config.yaml \
    --autoencoder-checkpoint /work3/s233249/ImgiNav/experiments/new_layouts/new_layouts_VAE_64x64_structural_256/new_layouts_VAE_64x64_structural_256_checkpoint_best.pt
```

#### Step 4: Create ControlNet Training Manifest

```bash
bsub < data_preparation/hpc_scripts/run_create_controlnet_manifest_new_layouts.sh
```

This creates:
- `/work3/s233249/ImgiNav/datasets/controlnet_training_manifest_new_layouts.csv`

**IMPORTANT**: Scenes are **NEVER skipped** - they are always included in training. By default, scenes use zero POV embeddings (`HANDLE_SCENES="zero"`). This is critical for the model to learn both room and scene layouts.

### Step 5: Train ControlNet

Update the config file with the diffusion checkpoint path, then:

```bash
python training/train_controlnet.py \
    --config experiments/controlnet/new_layouts/controlnet_unet48_d4_new_layouts.yaml
```

## Configuration

The ControlNet config (`controlnet_unet48_d4_new_layouts.yaml`) specifies:

- **UNet**: 48 base channels, depth 4 (matches diffusion model)
- **Adapter**: SimpleAdapter
  - Graph embedding dim: 384 (all-MiniLM-L6-v2)
  - POV embedding dim: 512 (ResNet18)
- **Fusion**: Additive fusion
- **Training**: 100 epochs, batch size 32, learning rate 0.0001

## Handling Scenes vs Rooms

**CRITICAL**: Scenes are **ALWAYS included** - they are never skipped.

- **Rooms**: Have POV embeddings (one training sample per POV)
- **Scenes**: No POV embeddings
  - Default: Use zero POV embedding (`HANDLE_SCENES="zero"`)
  - Alternative: Use empty string (`HANDLE_SCENES="empty"`)

**Dataset Loader**: The dataset loader (`models/datasets/datasets.py`) automatically handles the `"ZERO_EMBEDDING"` marker for scenes. When this marker is encountered, it creates a zero tensor with shape matching POV embeddings (512-dim for ResNet18). This ensures scenes are always included in training with consistent input shapes.

## File Structure

```
datasets/
├── joint_manifest.csv                          # Step 1 output
├── joint_manifest_with_embeddings.csv          # Step 2 output
├── layouts_cleaned_with_latents.csv            # Step 3 output
├── controlnet_training_manifest_new_layouts.csv # Step 4 output
└── collected/
    ├── graphs/                                  # Copied graph files
    └── povs/
        ├── tex/                                 # Textured POVs
        └── seg/                                 # Segmented POVs
```

## Troubleshooting

1. **Missing embeddings**: Check that all paths in manifests are absolute
2. **Scene handling**: Verify `HANDLE_SCENES` setting matches your needs
3. **Path resolution**: Ensure all file paths exist and are accessible

