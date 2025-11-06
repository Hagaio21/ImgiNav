# Visualization Module

This module contains visualization tools and utilities for exploring and interacting with ImgiNav data and models.

## Overview

The visualization module provides:
- Interactive web applications for model exploration
- 3D scene visualization utilities
- Latent space visualization
- Model output visualization

## Components

### Web Application (`app/`)

Interactive web application for exploring diffusion models and generated samples.

#### `diffusion_app.py`
Main application file for the diffusion visualization web app.

**Features:**
- Model loading and inference
- Sample generation
- Interactive parameter adjustment
- Real-time visualization
- Batch generation

**Usage:**
```bash
# Launch the app
cd visualization/app
python diffusion_app.py

# Or use provided scripts
bash launch.sh  # Linux/Mac
# or
.\launch.ps1     # Windows
```

**Access:**
- Default URL: `http://localhost:7860` (or as configured)

#### `embedding_utils.py`
Utilities for embedding visualization and manipulation.

**Features:**
- Embedding extraction
- Similarity computation
- Embedding interpolation

### Utilities

#### `lifting_utils.py`
Utilities for 3D scene lifting and visualization.

**Features:**
- 3D scene reconstruction
- Point cloud visualization
- Mesh processing

## HPC Scripts

### `hpc_scripts/run_zmap_extract.sh`
Script for extracting and processing zmap data on HPC clusters.

## App Structure

```
visualization/app/
├── diffusion_app.py      # Main application
├── embedding_utils.py     # Embedding utilities
├── launch.sh             # Launch script (Linux/Mac)
├── launch.ps1            # Launch script (Windows)
└── README.md             # App-specific documentation
```

## Usage

### Running the Web App

1. **Install Dependencies**:
   ```bash
   pip install gradio torch torchvision
   ```

2. **Launch the App**:
   ```bash
   cd visualization/app
   python diffusion_app.py
   ```

3. **Access the Interface**:
   - Open browser to the displayed URL (typically `http://localhost:7860`)
   - Use the interface to:
     - Load models
     - Generate samples
     - Adjust parameters
     - View results

### Using Visualization Utilities

```python
from visualization.lifting_utils import lift_scene_to_3d
from visualization.app.embedding_utils import extract_embeddings

# Extract embeddings
embeddings = extract_embeddings(model, dataloader)

# Lift to 3D
scene_3d = lift_scene_to_3d(layout_image, embeddings)
```

## Features

### Diffusion App Features
- **Model Selection**: Choose from available checkpoints
- **Parameter Control**: Adjust sampling parameters
- **Batch Generation**: Generate multiple samples
- **Real-time Preview**: See results as they're generated
- **Export Options**: Save generated images

### Embedding Visualization
- **UMAP Projections**: 2D/3D latent space visualization
- **Interpolation**: Visualize latent space paths
- **Similarity Search**: Find similar embeddings

### 3D Visualization
- **Scene Reconstruction**: Reconstruct 3D from 2D layouts
- **Point Cloud Display**: Visualize point clouds
- **Mesh Rendering**: Render 3D meshes

## Configuration

The app can be configured via command-line arguments or environment variables:

```bash
python diffusion_app.py \
    --model_path /path/to/model.pt \
    --port 7860 \
    --host 0.0.0.0
```

## Notes

- The web app uses Gradio for the interface
- Visualization utilities can be used independently
- 3D visualization requires additional dependencies (trimesh, open3d)
- The app supports both CPU and GPU inference

