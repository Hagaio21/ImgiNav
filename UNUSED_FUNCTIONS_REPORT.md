# Unused Functions Report

This report documents all functions that were declared/imported but not being used in the codebase.

## Summary

- **Total unused functions found**: 12
- **Critical bug fixed**: 1 (missing `NumpySafeLoader` definition)
- **Status**: ✅ All unused functions have been removed

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

## Actions Taken

✅ **All unused functions have been removed:**

1. **Removed from `common/utils.py`:**
   - `extract_tensor_from_batch()` - ✅ Removed
   - `is_augmented_path()` - ✅ Removed
   - `ensure_columns_exist()` - ✅ Removed

2. **Removed from `common/file_io.py`:**
   - `read_yaml()` - ✅ Removed

3. **Removed from `common/weighting.py`:**
   - `weighted_sample()` - ✅ Removed

4. **Removed from `common/taxonomy.py`:**
   - `load_valid_colors()` - ✅ Removed
   - `generate_palette_for_labels()` - ✅ Removed
   - `assign_colors_golden_ratio()` - ✅ Removed (only used by removed function)

5. **Removed from `common/__init__.py` exports:**
   - `ensure_columns_exist` - ✅ Removed from exports
   - `load_valid_colors` - ✅ Removed from exports
   - `generate_palette_for_labels` - ✅ Removed from exports
   - `assign_colors_golden_ratio` - ✅ Removed from exports
   - `build_taxonomy_full` - ✅ Removed from exports (used internally only)
   - `build_room_taxonomy` - ✅ Removed from exports (used internally only)
   - `assign_colors` - ✅ Removed from exports (used internally only)

**Note:** Functions like `build_taxonomy_full`, `build_room_taxonomy`, and `assign_colors` are kept in the file as they are used internally by `build_taxonomy()`, but removed from public exports since they're not used externally.

## Notes

- Functions that are used internally within the same module (like `build_taxonomy_full` being used by `build_taxonomy`) are not considered unused, but their export in `__init__.py` suggests they were intended for external use.
- Some functions might be used in notebooks or scripts not tracked in this analysis.
- Functions with `main()` entry points are not considered unused as they're meant to be called directly.

