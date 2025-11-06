# Analysis Module

This module contains scripts for analyzing datasets, model performance, and experimental results.

## Overview

The analysis module provides:
- Dataset analysis and statistics
- Latent space analysis
- Class distribution analysis
- Experiment result analysis
- Visualization generation

## Scripts

### `analyze_layout_dataset.py`
Analyzes the layout dataset and generates statistics.

**Features:**
- Dataset statistics (size, distribution)
- Image quality analysis
- Duplicate detection
- Empty sample detection
- Visualizations

**Output:**
- `dataset_stats.json` - Dataset statistics
- `aspect_ratio_hist.png` - Aspect ratio distribution
- `intensity_hist.png` - Intensity distribution
- `duplicates_heatmap.png` - Duplicate detection
- `empty_samples.png` - Empty sample examples
- `random_samples.png` - Random sample visualization

**Usage:**
```bash
python analysis/analyze_layout_dataset.py \
    --manifest /path/to/layouts.csv \
    --output_dir analysis/dataset_analysis
```

### `analyze_class_distribution.py`
Analyzes class distribution in the dataset.

**Features:**
- Room type distribution
- Scene distribution
- Class imbalance analysis
- Grouping and aggregation
- Statistical summaries

**Output:**
- `class_distribution_stats.json` - Distribution statistics
- `class_distribution_pie.png` - Pie chart
- `class_distribution_top20.png` - Top 20 classes
- `room_id_distribution.png` - Room ID distribution
- `scene_room_distribution.png` - Scene-room distribution

**Usage:**
```bash
python analysis/analyze_class_distribution.py \
    --manifest /path/to/layouts.csv \
    --output_dir analysis/class_distribution_results
```

### `analyze_latents.py`
Analyzes latent representations from autoencoders.

**Features:**
- Latent space visualization (UMAP, PCA)
- Interpolation analysis
- Random sample decoding
- Latent health metrics
- Clustering analysis

**Metrics:**
- Variance per feature
- Zero variance ratio
- Mean similarity
- PCA explained variance
- Silhouette score
- Davies-Bouldin score

**Output:**
- `metrics.json` - Latent health metrics
- `umap_*.png` - UMAP visualizations
- `interpolation.png` - Latent interpolation
- `random_decoded.png` - Random decoded samples

**Usage:**
```bash
python analysis/analyze_latents.py \
    --checkpoint /path/to/autoencoder.pt \
    --manifest /path/to/layouts.csv \
    --output_dir analysis/latent_analysis_results
```

### Phase Analysis Scripts

#### `phase1_1_analysis.py`
Analyzes Phase 1.1 (latent shape sweep) results.

**Features:**
- Compare all experiments
- Latent dimension analysis
- Reconstruction quality comparison
- Best configuration selection

**Usage:**
```bash
python analysis/phase1_1_analysis.py \
    --experiments_dir outputs/phase1_1_latent_shape_sweep \
    --output_dir analysis/phase1_1_results
```

#### `phase1_2_analysis.py`
Analyzes Phase 1.2 (VAE test) results.

**Features:**
- Deterministic vs VAE comparison
- KL divergence analysis
- Reconstruction quality comparison

**Usage:**
```bash
python analysis/phase1_2_analysis.py \
    --experiments_dir outputs/phase1_2_vae_test \
    --output_dir analysis/phase1_2_results
```

#### `phase1_3_analysis.py`
Analyzes Phase 1.3 (loss tuning) results.

**Features:**
- Loss weight comparison
- Multi-head performance analysis
- Best loss configuration selection

**Usage:**
```bash
python analysis/phase1_3_analysis.py \
    --experiments_dir outputs/phase1_3_loss_tuning \
    --output_dir analysis/phase1_3_results
```

### `plot_diffusion_ablation_losses.py`
Plots loss curves for diffusion ablation experiments.

**Features:**
- Loss curve visualization
- Multiple experiment comparison
- Training stability analysis

**Usage:**
```bash
python analysis/plot_diffusion_ablation_losses.py \
    --experiments_dir experiments/diffusion/ablation \
    --output_path analysis/diffusion_ablation_losses.png
```

## HPC Scripts

The `hpc_scripts/` directory contains shell scripts for running analysis on HPC clusters.

### Analysis Scripts
- `analyze_class_distribution.sh` - Run class distribution analysis
- `analyze_latents.sh` - Run latent analysis
- `analyze_layouts.sh` - Run layout dataset analysis

### Phase Analysis Scripts
- `launch_phase1_1_analysis.sh` - Launch Phase 1.1 analysis
- `launch_phase1_2_analysis.sh` - Launch Phase 1.2 analysis
- `launch_phase1_3_analysis.sh` - Launch Phase 1.3 analysis
- `run_phase1_X_analysis.sh` - Individual phase analysis scripts

### Plotting Scripts
- `plot_diffusion_ablation_losses.sh` - Plot diffusion ablation losses

## Output Structure

### Dataset Analysis
```
analysis/dataset_analysis/
├── dataset_stats.json
├── aspect_ratio_hist.png
├── intensity_hist.png
├── duplicates_heatmap.png
├── empty_samples.png
└── random_samples.png
```

### Class Distribution
```
analysis/class_distribution_results/
├── class_distribution_stats.json
├── class_distribution_pie.png
├── class_distribution_top20.png
├── room_id_distribution.png
└── scene_room_distribution.png
```

### Latent Analysis
```
analysis/latent_analysis_results/
├── {experiment_name}/
│   ├── metrics.json
│   ├── umap_continents.png
│   ├── umap_countries_room.png
│   ├── umap_countries_scene.png
│   ├── interpolation.png
│   └── random_decoded.png
└── summary_diffusion_fitness.png
```

## Key Metrics

### Dataset Metrics
- **Total Samples**: Number of samples in dataset
- **Empty Ratio**: Percentage of empty samples
- **Aspect Ratio**: Image aspect ratio distribution
- **Intensity**: Pixel intensity distribution

### Class Distribution Metrics
- **Class Counts**: Number of samples per class
- **Class Frequencies**: Relative frequencies
- **Imbalance Ratio**: Ratio of most to least frequent class
- **Grouped Statistics**: Statistics for grouped classes

### Latent Metrics
- **Variance per Feature**: Feature-wise variance
- **Zero Variance Ratio**: Percentage of dead features
- **Mean Similarity**: Average cosine similarity
- **PCA Explained Variance**: Dimensionality analysis
- **Clustering Scores**: Silhouette, Davies-Bouldin

## Visualization Types

1. **Distribution Plots**: Histograms, pie charts, bar charts
2. **UMAP Visualizations**: 2D/3D latent space projections
3. **Interpolation Plots**: Latent space interpolation paths
4. **Sample Grids**: Random sample visualizations
5. **Heatmaps**: Similarity matrices, duplicate detection

## Usage Examples

### Analyze Dataset
```bash
python analysis/analyze_layout_dataset.py \
    --manifest datasets/layouts.csv \
    --output_dir analysis/dataset_analysis
```

### Analyze Latents
```bash
python analysis/analyze_latents.py \
    --checkpoint experiments/phase1/phase1_6_AE_normalized/checkpoints/best.pt \
    --manifest datasets/layouts.csv \
    --output_dir analysis/latent_analysis_results/phase1_6
```

### Compare Experiments
```bash
python analysis/phase1_1_analysis.py \
    --experiments_dir outputs/phase1_1_latent_shape_sweep \
    --output_dir analysis/phase1_1_comparison
```

## Notes

- All analysis scripts support batch processing
- Visualizations use matplotlib with non-interactive backend (Agg) for HPC
- Results are saved as both JSON (metrics) and PNG (visualizations)
- Analysis can be run on checkpoints or during training

