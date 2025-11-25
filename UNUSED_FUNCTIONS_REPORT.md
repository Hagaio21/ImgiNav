# Unused Functions Report

This report documents all functions that are declared/imported but not being used in the codebase.

## Summary

- **Total unused functions found**: 12
- **Critical bug fixed**: 1 (missing `NumpySafeLoader` definition)

## Unused Functions by Module

### 1. `common/utils.py`

#### `extract_tensor_from_batch(batch, device=None, key="layout")`
- **Location**: `common/utils.py:80-106`
- **Status**: Defined but never imported or used anywhere
- **Description**: Extracts tensor from various batch types (dict, list/tuple, or tensor)

#### `is_augmented_path(path_str)`
- **Location**: `common/utils.py:109-124`
- **Status**: Defined but never imported or used anywhere
- **Description**: Checks if a path string indicates an augmented image

### 2. `common/file_io.py`

#### `read_yaml(path: Path) -> Any`
- **Location**: `common/file_io.py:47-53`
- **Status**: Defined but never imported or used anywhere (only mentioned in README)
- **Description**: Reads YAML file and returns parsed object

### 3. `common/weighting.py`

#### `weighted_sample(df, n, weight_column="sample_weight", random_state=42, replace=False)`
- **Location**: `common/weighting.py:167-194`
- **Status**: Defined but never imported or used anywhere
- **Description**: Performs weighted sampling from a DataFrame

### 4. `common/taxonomy.py` (Exported in `__init__.py` but never imported)

These functions are exported in `common/__init__.py` for external use, but are never actually imported from `common` anywhere in the codebase. They are used internally within `build_taxonomy()` but not externally.

#### `build_taxonomy_full(model_info_path, scenes_dir)`
- **Location**: `common/taxonomy.py:445-525`
- **Status**: Exported in `__init__.py` but never imported from `common`
- **Note**: Used internally by `build_taxonomy()`

#### `build_room_taxonomy(scenes_dir)`
- **Location**: `common/taxonomy.py:527-545`
- **Status**: Exported in `__init__.py` but never imported from `common`
- **Note**: Used internally by `build_taxonomy()`

#### `assign_colors(super2id, category2id, category2super)`
- **Location**: `common/taxonomy.py:548-627`
- **Status**: Exported in `__init__.py` but never imported from `common`
- **Note**: Used internally by `build_taxonomy()`

#### `assign_colors_golden_ratio(label2id) -> dict`
- **Location**: `common/taxonomy.py:630-643`
- **Status**: Exported in `__init__.py` but never imported from `common`
- **Note**: Used internally by `generate_palette_for_labels()`

#### `generate_palette_for_labels(json_path) -> bool`
- **Location**: `common/taxonomy.py:646-661`
- **Status**: Exported in `__init__.py` but never imported from `common`

#### `load_valid_colors(taxonomy_path, include_background=True)`
- **Location**: `common/taxonomy.py:417-432`
- **Status**: Exported in `__init__.py` but never imported from `common`

### 5. `common/utils.py` (Exported in `__init__.py` but never imported)

#### `ensure_columns_exist(df, required_columns, source="dataframe")`
- **Location**: `common/utils.py:63-66`
- **Status**: Exported in `__init__.py` but never imported from `common`

## Critical Bug Fixed

### Missing `NumpySafeLoader` Definition
- **Location**: `training/utils.py:63`
- **Issue**: `NumpySafeLoader` was used but never defined
- **Fix**: Added class definition `class NumpySafeLoader(yaml.SafeLoader):` before its usage
- **Status**: ✅ Fixed

## Recommendations

1. **Remove unused functions** if they're not needed for future use:
   - `extract_tensor_from_batch` (common/utils.py)
   - `is_augmented_path` (common/utils.py)
   - `read_yaml` (common/file_io.py) - unless YAML reading is planned
   - `weighted_sample` (common/weighting.py)

2. **Consider removing from `__init__.py` exports** if not intended for external use:
   - Functions in `common/taxonomy.py` that are only used internally
   - `ensure_columns_exist` from `common/utils.py`

3. **Keep exported functions** if they're part of a public API that may be used in the future:
   - Functions exported in `common/__init__.py` might be intended for external scripts or notebooks

## Notes

- Functions that are used internally within the same module (like `build_taxonomy_full` being used by `build_taxonomy`) are not considered unused, but their export in `__init__.py` suggests they were intended for external use.
- Some functions might be used in notebooks or scripts not tracked in this analysis.
- Functions with `main()` entry points are not considered unused as they're meant to be called directly.

