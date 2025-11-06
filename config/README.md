# Config Module

This module contains configuration files used throughout the ImgiNav project.

## Overview

The config module provides:
- Taxonomy definitions
- Inference configurations
- Z-map configurations
- Reference configurations for experiments

## Files

### `taxonomy.json`
Taxonomy definition file containing all semantic categories, room types, and labels.

**Structure:**
- **Super Categories** (1000-1999): High-level object categories
  - Examples: Furniture, Structure, Lighting
- **Categories** (2000-2999): Specific object types
  - Examples: Chair, Table, Wall, Floor
- **Rooms** (3000-3999): Room types
  - Examples: Bedroom, Kitchen, Living Room
- **Labels** (4000-4999): Semantic labels for classification
  - Examples: Empty, Non-empty

**Key Mappings:**
- `id2name`: ID to name mapping
- `id2color`: ID to RGB color mapping (for segmentation)
- `id2super`: ID to super-category mapping
- `id2category`: ID to category mapping
- `id2room`: ID to room type mapping
- `id2label`: ID to label mapping

**Usage:**
```python
from common.taxonomy import Taxonomy

taxonomy = Taxonomy("config/taxonomy.json")
room_name = taxonomy.id_to_name(3001)  # Get room name
color = taxonomy.id_to_rgb(2053)  # Get RGB for wall
```

### `inference_config.yaml`
Configuration file for model inference.

**Sections:**
- Model configuration
- Sampling parameters
- Output settings
- Device settings

**Usage:**
```bash
python scripts/diffusion_inference.py --config config/inference_config.yaml
```

### `zmap.json`
Z-map configuration for depth/height mapping.

**Purpose:**
- Defines height mappings for layout visualization
- Used in 3D scene reconstruction
- Maps pixel values to 3D coordinates

**Usage:**
```python
import json
from pathlib import Path

zmap = json.loads(Path("config/zmap.json").read_text())
height = zmap[pixel_value]
```

## Configuration Structure

### Taxonomy Structure
```json
{
  "ranges": {
    "super": [1000, 1999],
    "category": [2000, 2999],
    "room": [3000, 3999],
    "label": [4000, 4999]
  },
  "id2super": {...},
  "id2category": {...},
  "id2room": {...},
  "id2label": {...},
  "id2color": {...}
}
```

### Inference Config Structure
```yaml
model:
  checkpoint: /path/to/model.pt
  config: /path/to/model_config.yaml

sampling:
  num_steps: 500
  guidance_scale: 1.0
  seed: 42

output:
  save_dir: outputs/inference
  format: png
```

## Notes

- Taxonomy file is used throughout the project for semantic understanding
- Inference config can be customized for different use cases
- Z-map is specific to 3D reconstruction tasks
- All configs use standard formats (JSON, YAML) for easy editing

## Related Modules

- **Common Module**: Uses taxonomy for semantic operations
- **Models Module**: Uses taxonomy for segmentation heads
- **Data Preparation**: Uses taxonomy for label processing
- **Scripts Module**: Uses inference config for model inference

