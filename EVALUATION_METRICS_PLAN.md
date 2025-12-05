# Evaluation Metrics Plan for ImgiNav

## Overview

This document outlines the comprehensive evaluation metrics for comparing 6 diffusion model experiments:
- **3 Conditioning Types**: POV-only, Graph-only, Both
- **3 Model Sizes**: Small, Medium, Large
- **Total**: 6 experiments

## 1. Object-Level Metrics

### 1.1 Object Presence Metrics

**Purpose**: Measure if all required objects are generated and if extra objects are created.

#### Metrics:
- **Precision**: `TP / (TP + FP)`
  - TP: Objects in both target and generated
  - FP: Objects in generated but not in target
  - Measures: Are detected objects correct?

- **Recall**: `TP / (TP + FN)`
  - FN: Objects in target but not in generated
  - Measures: Are all target objects detected?

- **F1-Score**: `2 * (Precision * Recall) / (Precision + Recall)`
  - Balanced measure of precision and recall

- **Missing Objects**: List of object classes present in target but not in generated
- **Extra Objects**: List of object classes present in generated but not in target
- **Object Count Accuracy**: `1 - |generated_count - target_count| / max(target_count, 1)`

#### Implementation:
```python
def compute_object_presence_metrics(target_objects, generated_objects):
    target_classes = set(obj["class"] for obj in target_objects)
    generated_classes = set(obj["class"] for obj in generated_objects)
    
    tp = len(target_classes & generated_classes)
    fp = len(generated_classes - target_classes)
    fn = len(target_classes - generated_classes)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missing": list(target_classes - generated_classes),
        "extra": list(generated_classes - target_classes)
    }
```

### 1.2 Per-Class Object Metrics

**Purpose**: Evaluate performance for each object class separately.

#### Metrics (per class):
- **Class Precision**: Precision for this specific class
- **Class Recall**: Recall for this specific class
- **Class F1**: F1-score for this specific class
- **Class Count Error**: `|generated_count - target_count|`
- **Class Presence**: Binary (1 if class present in both, 0 otherwise)

## 2. Spatial Metrics

### 2.1 Bounding Box Metrics (Existing)

**Purpose**: Measure spatial accuracy of object placement.

#### Metrics:
- **Mean Bbox IoU**: Average IoU across all matched object pairs
- **Per-Class Bbox IoU**: IoU for each object class
- **Bbox Coverage**: How much of target bbox is covered by generated bbox
- **Centroid Distance**: Euclidean distance between object centroids
- **Mean Centroid Distance**: Average across all matched objects

### 2.2 Pixel-Level Metrics

**Purpose**: Measure pixel-level accuracy of layout generation.

#### Metrics:
- **Pixel Accuracy**: `correct_pixels / total_pixels`
  - Measures overall pixel classification accuracy

- **Mean Pixel IoU (mIoU)**: Average IoU across all classes at pixel level
  - Formula: `mIoU = (1/N) * Σ(IoU_i)` where N is number of classes

- **Per-Class Pixel IoU**: IoU for each object class at pixel level
  - Formula: `IoU_i = intersection_i / union_i`

- **Background IoU**: How well background/walls are preserved
- **Floor IoU**: Accuracy of floor region

#### Implementation:
```python
def compute_pixel_metrics(target_img, generated_img, taxonomy):
    target_array = np.array(target_img)
    generated_array = np.array(generated_img)
    
    # Create class masks
    classes = get_all_classes(target_array, generated_array, taxonomy)
    
    pixel_accuracy = np.mean(target_array == generated_array)
    
    per_class_ious = {}
    for cls in classes:
        target_mask = create_class_mask(target_array, cls, taxonomy)
        generated_mask = create_class_mask(generated_array, cls, taxonomy)
        
        intersection = np.sum(target_mask & generated_mask)
        union = np.sum(target_mask | generated_mask)
        iou = intersection / union if union > 0 else 0.0
        per_class_ious[cls] = iou
    
    miou = np.mean(list(per_class_ious.values()))
    
    return {
        "pixel_accuracy": pixel_accuracy,
        "mean_pixel_iou": miou,
        "per_class_pixel_iou": per_class_ious
    }
```

## 3. Layout-Level Metrics

### 3.1 Structural Metrics

**Purpose**: Evaluate overall layout structure and coherence.

#### Metrics:
- **Room Coverage**: Fraction of room area covered by objects
  - Formula: `object_pixels / total_room_pixels`

- **Object Density**: Objects per unit area
  - Formula: `num_objects / room_area`

- **Overlap Ratio**: Fraction of objects that overlap
  - Measures: Are objects properly separated?

- **Wall Alignment**: How well objects align with walls
  - Measures distance from objects to nearest wall

- **Spatial Distribution**: Are objects evenly distributed?
  - Uses spatial statistics (e.g., nearest neighbor distance)

### 3.2 Semantic Consistency Metrics

**Purpose**: Evaluate semantic correctness of layouts.

#### Metrics:
- **Class Co-occurrence Score**: Do expected object pairs appear together?
  - Example: "sofa" and "coffee table" should appear together
  - Uses co-occurrence statistics from training data

- **Room Type Consistency**: Does layout match room type from text?
  - Compares generated objects to expected objects for room type

- **Object Size Consistency**: Are object sizes realistic?
  - Compares bbox sizes to expected sizes from training data

## 4. Aggregate Summary Metrics

### 4.1 Overall Scores

**Purpose**: Single-number metrics for easy comparison.

#### Metrics:
- **Overall IoU**: Weighted average across all classes
  - Formula: `Σ(class_iou * class_weight) / Σ(class_weight)`

- **Total Object Count Accuracy**: `1 - |total_generated - total_target| / total_target`

- **Object Completeness**: `matched_objects / total_target_objects`

- **Layout Plausibility Score**: Combined metric (weighted average of multiple metrics)
  - Formula: `w1 * object_f1 + w2 * spatial_iou + w3 * semantic_score`

### 4.2 Per-Experiment Aggregations

For each of the 6 experiments, compute:
- **Mean metrics** across all test samples
- **Std metrics** (standard deviation) for variance analysis
- **Per-class breakdowns** to identify difficult classes
- **Failure cases**: Samples with lowest scores

## 5. Evaluation Pipeline

### 5.1 Per-Sample Evaluation

For each test sample:
1. Extract objects from target layout
2. Extract objects from generated layout
3. Match objects (by class and spatial proximity)
4. Compute all metrics
5. Store per-sample results

### 5.2 Batch Evaluation

For each experiment:
1. Load all generated layouts
2. Load corresponding target layouts
3. Run per-sample evaluation
4. Aggregate results
5. Generate comparison tables

### 5.3 Cross-Experiment Comparison

Compare all 6 experiments:
1. Create comparison table (6 experiments × all metrics)
2. Statistical significance testing (t-tests, ANOVA)
3. Visualizations (bar charts, heatmaps)
4. Identify best performing configuration

## 6. Implementation Files

### 6.1 Core Evaluation Scripts

1. **`scripts/evaluate_layout.py`**: Single layout evaluation
   - Input: target layout, generated layout, taxonomy
   - Output: Dictionary of all metrics

2. **`scripts/evaluate_experiment.py`**: Batch evaluation for one experiment
   - Input: Experiment directory, test manifest, taxonomy
   - Output: CSV/JSON with per-sample and aggregate metrics

3. **`scripts/compare_all_experiments.py`**: Compare all 6 experiments
   - Input: 6 experiment directories
   - Output: Comparison tables, visualizations

### 6.2 Extended Metrics Script

**`scripts/evaluate_layout_extended.py`**: Extended version of `compare_layouts.py`
- Adds pixel-level metrics
- Adds precision/recall/F1
- Adds layout-level metrics
- Backward compatible with existing code

## 7. Output Format

### 7.1 Per-Sample Results (JSON)

```json
{
  "sample_id": "scene_room_pov",
  "object_level": {
    "precision": 0.85,
    "recall": 0.90,
    "f1": 0.87,
    "object_count_accuracy": 0.95,
    "missing_objects": ["coffee_table"],
    "extra_objects": []
  },
  "spatial": {
    "mean_bbox_iou": 0.72,
    "mean_centroid_distance": 15.3,
    "pixel_accuracy": 0.88,
    "mean_pixel_iou": 0.75,
    "per_class_pixel_iou": {
      "sofa": 0.82,
      "table": 0.68
    }
  },
  "layout": {
    "room_coverage": 0.45,
    "object_density": 0.12,
    "overlap_ratio": 0.05
  }
}
```

### 7.2 Aggregate Results (CSV)

| Experiment | Mean_Precision | Mean_Recall | Mean_F1 | Mean_Pixel_IoU | Mean_Bbox_IoU | ... |
|------------|---------------|-------------|---------|----------------|---------------|-----|
| pov_small   | 0.82          | 0.85       | 0.83    | 0.71           | 0.68          | ... |
| graph_small | 0.78          | 0.80       | 0.79    | 0.69           | 0.65          | ... |
| both_small  | 0.88          | 0.90       | 0.89    | 0.75           | 0.72          | ... |
| ...         | ...           | ...        | ...     | ...            | ...           | ... |

## 8. Key Metrics for Thesis

### 8.1 Primary Metrics (Most Important)

1. **Object Completeness (Recall)**: Can the model generate all required objects?
2. **Spatial Accuracy (Bbox IoU)**: Are objects in the right places?
3. **Count Accuracy**: Does it generate the right number of objects?
4. **Pixel Accuracy**: Overall quality of layout generation

### 8.2 Secondary Metrics (Supporting)

1. **Per-class performance**: Which object classes are hardest?
2. **Size effect**: How do small/medium/large models compare?
3. **Conditioning effect**: POV vs Graph vs Both
4. **Layout plausibility**: Semantic consistency

### 8.3 Statistical Analysis

- **Significance testing**: Are differences between experiments statistically significant?
- **Confidence intervals**: Report metrics with 95% CI
- **Effect sizes**: How large are the differences?

## 9. Evaluation Checklist

### Before Evaluation:
- [ ] All 6 experiments trained and checkpoints saved
- [ ] Test set defined and manifest created
- [ ] All layouts generated for test set
- [ ] Taxonomy file up to date

### During Evaluation:
- [ ] Run per-sample evaluation for all test samples
- [ ] Aggregate results per experiment
- [ ] Generate comparison tables
- [ ] Create visualizations
- [ ] Run statistical tests

### After Evaluation:
- [ ] Document results in thesis
- [ ] Identify failure cases for analysis
- [ ] Compare with baseline (if available)
- [ ] Prepare figures for thesis

## 10. Future Extensions

### 10.1 Navigation Metrics (Stretch Goal)

If navigation loop is implemented:
- **Path Validity**: Can A* find valid paths?
- **Path Success Rate**: % of successful navigation attempts
- **Path Stretch**: How much longer are paths vs optimal?

### 10.2 Iterative Refinement Metrics

If iterative refinement is implemented:
- **Refinement Gain**: Improvement in metrics per iteration
- **Convergence Rate**: How many iterations to converge?
- **Final vs Initial**: Comparison of final vs initial layout quality

