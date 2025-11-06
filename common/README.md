# Common Module

This module contains shared utility functions and classes used throughout the ImgiNav project.

## Overview

The `common` module provides:
- File I/O utilities for reading/writing manifests, JSON, and YAML files
- Taxonomy management for scene understanding and classification
- General utility functions for configuration loading, progress tracking, and directory management

## Files

### `file_io.py`
File I/O utilities for handling various file formats.

**Functions:**
- `read_manifest(path)` - Read CSV manifest files into list of dictionaries
- `create_manifest(rows, output, fieldnames)` - Write rows to CSV manifest
- `read_json(path)` - Read JSON files
- `write_json(path, data, indent)` - Write data to JSON files
- `read_yaml(path)` - Read YAML files
- `write_yaml(path, data)` - Write data to YAML files

**Usage:**
```python
from common.file_io import read_manifest, write_json

# Read a manifest
rows = read_manifest(Path("data/manifest.csv"))

# Write JSON
write_json(Path("output.json"), {"key": "value"})
```

### `taxonomy.py`
Taxonomy management for scene understanding, including object categories, room types, and semantic labels.

**Main Class: `Taxonomy`**

**Key Methods:**
- `id_to_name(val)` - Convert taxonomy ID to human-readable name
- `name_to_id(name)` - Convert name to taxonomy ID
- `rgb_to_id(rgb)` - Convert RGB color to taxonomy ID (for segmentation)
- `id_to_rgb(val)` - Convert taxonomy ID to RGB color
- `get_super_category(val)` - Get super-category for a given ID
- `get_room_type(val)` - Get room type for a given ID

**Usage:**
```python
from common.taxonomy import Taxonomy

taxonomy = Taxonomy("config/taxonomy.json")
room_name = taxonomy.id_to_name(1001)  # Get room name from ID
rgb = taxonomy.id_to_rgb(2053)  # Get RGB color for wall
```

**Taxonomy Structure:**
- **Super Categories** (1000-1999): High-level object categories (e.g., Furniture, Structure)
- **Categories** (2000-2999): Specific object types (e.g., Chair, Table, Wall)
- **Rooms** (3000-3999): Room types (e.g., Bedroom, Kitchen)
- **Labels** (4000-4999): Semantic labels for classification

### `utils.py`
General utility functions for configuration, progress tracking, and file system operations.

**Functions:**
- `safe_mkdir(path, parents, exist_ok)` - Safely create directories with error handling
- `write_json(data, path, indent)` - Write JSON with directory creation
- `create_progress_tracker(total, description)` - Create progress tracking function
- `load_config_with_profile(config_path, profile)` - Load configuration files (YAML/JSON) with profile support
- `set_seeds(seed)` - Set random seeds for reproducibility

**Usage:**
```python
from common.utils import safe_mkdir, load_config_with_profile

# Create directory
safe_mkdir(Path("output/experiments"))

# Load config
config = load_config_with_profile("config/experiment.yaml", profile="production")
```

## Dependencies

- `pathlib` - Path handling
- `json` - JSON parsing
- `yaml` (via `pyyaml`) - YAML parsing
- `csv` - CSV manifest handling

## Notes

- All file operations use UTF-8 encoding
- Path objects are preferred over strings for better cross-platform compatibility
- Taxonomy class precomputes lookup tables for performance
- Error handling is included for file operations

