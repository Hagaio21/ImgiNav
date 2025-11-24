# Evaluation Metrics Guide

This document explains the evaluation metrics used during training and what values indicate good performance.

## Overview

The training script computes several evaluation metrics to assess model performance. Each metric is saved as a separate plot file: `{exp_name}_metric_{metric_name}.png`

## Core Metrics

### 1. CLIP Score (`clip_score` / `val_clip_score`)

**What it measures:** Semantic alignment between generated layouts and text/POV inputs using CLIP embeddings.

**Range:** Typically -1 to 1 (cosine similarity), but can vary
- **Higher is better**

**Good values:**
- **Excellent:** > 0.3
- **Good:** 0.1 - 0.3
- **Fair:** 0.0 - 0.1
- **Poor:** < 0.0

**Interpretation:**
- Measures how well the generated layouts match the semantic meaning of the input text/POV embeddings
- A higher score means the generated images are semantically similar to what the text/POV describes
- Note: This is a per-batch approximation and may vary significantly between batches

---

### 2. FID (Fréchet Inception Distance) (`fid` / `val_fid`)

**What it measures:** Distribution quality - how similar the distribution of generated images is to real images using Inception v3 features.

**Range:** 0 to ∞
- **Lower is better**

**Good values:**
- **Excellent:** < 10
- **Good:** 10 - 50
- **Fair:** 50 - 100
- **Poor:** > 100
- **Invalid:** > 1e6 (indicates numerical issues - should be filtered out)

**Interpretation:**
- Measures the quality and diversity of generated images
- Lower FID means generated images are more similar to real images in terms of visual features
- FID < 20 is typically considered very good for image generation tasks
- **Note:** This is a per-batch approximation. For accurate FID, features should be accumulated across all validation batches

**Common issues:**
- Very large values (> 1e6) indicate numerical problems (small batch size, singular covariance matrices)
- The code now includes safeguards to filter these out

---

### 3. mIoU (Mean Intersection over Union) (`miou` / `val_miou`)

**What it measures:** Spatial correctness - how well the model preserves object boundaries and spatial relationships by computing IoU for each category and averaging.

**Range:** 0 to 1
- **Higher is better**

**Good values:**
- **Excellent:** > 0.7
- **Good:** 0.5 - 0.7
- **Fair:** 0.3 - 0.5
- **Poor:** < 0.3

**Interpretation:**
- Critical for geometric/spatial correctness in layout generation
- Measures how accurately objects are placed and sized in the generated layouts
- Higher mIoU means better preservation of object boundaries and spatial relationships
- More accurate than pixel-wise MSE for geometric tasks

---

## Layout-Specific Metrics

### 4. Coverage Difference (`coverage_diff` / `val_coverage_diff`)

**What it measures:** Difference in total object coverage between generated and ground truth layouts.

**Range:** 0 to 1 (typically small values)
- **Lower is better**

**Good values:**
- **Excellent:** < 0.01
- **Good:** 0.01 - 0.05
- **Fair:** 0.05 - 0.1
- **Poor:** > 0.1

**Interpretation:**
- Measures whether the model generates the right amount of "stuff" (objects) in the layout
- A small difference means the generated layout has similar object density to the ground truth

---

### 5. Class IoU (`class_iou` / `val_class_iou`)

**What it measures:** Intersection over Union for object classes/categories between generated and ground truth.

**Range:** 0 to 1
- **Higher is better**

**Good values:**
- **Excellent:** > 0.8
- **Good:** 0.6 - 0.8
- **Fair:** 0.4 - 0.6
- **Poor:** < 0.4

**Interpretation:**
- Measures how well the model places objects of the correct categories
- Higher values mean better category-level spatial accuracy
- Different from mIoU: this focuses on category matching rather than pixel-level segmentation

---

### 6. Class KL Divergence (`class_kl_divergence` / `val_class_kl_divergence`)

**What it measures:** Distribution difference of object classes between generated and ground truth layouts.

**Range:** 0 to ∞
- **Lower is better**

**Good values:**
- **Excellent:** < 0.1
- **Good:** 0.1 - 0.3
- **Fair:** 0.3 - 0.5
- **Poor:** > 0.5

**Interpretation:**
- Measures whether the model generates the right distribution of object types
- Lower values mean the generated layouts have similar class distributions to ground truth
- A value of 0 means identical distributions

---

### 7. Class Total Variation (`class_total_variation` / `val_class_total_variation`)

**What it measures:** Spatial smoothness/variation of class distributions.

**Range:** 0 to ∞
- **Lower is better** (for most cases)

**Good values:**
- **Excellent:** < 0.01
- **Good:** 0.01 - 0.05
- **Fair:** 0.05 - 0.1
- **Poor:** > 0.1

**Interpretation:**
- Measures spatial smoothness of class distributions
- Lower values indicate smoother, more coherent layouts
- Very high values might indicate noisy or fragmented layouts

---

### 8. Class L1 Distance (`class_l1_distance` / `val_class_l1_distance`)

**What it measures:** L1 (Manhattan) distance between class distributions of generated and ground truth.

**Range:** 0 to ∞
- **Lower is better**

**Good values:**
- **Excellent:** < 0.01
- **Good:** 0.01 - 0.05
- **Fair:** 0.05 - 0.1
- **Poor:** > 0.1

**Interpretation:**
- Measures absolute difference in class distributions
- More sensitive to outliers than KL divergence
- Lower values mean better class distribution matching

---

## Training Metrics

### 9. MSE Pred Noise (`train_MSE_pred_noise` / `val_MSE_pred_noise`)

**What it measures:** Mean Squared Error between predicted and actual noise in the diffusion process.

**Range:** 0 to ∞
- **Lower is better**

**Good values:**
- **Excellent:** < 0.1
- **Good:** 0.1 - 0.3
- **Fair:** 0.3 - 0.5
- **Poor:** > 0.5

**Interpretation:**
- Primary training loss for diffusion models
- Measures how well the model predicts the noise to remove at each timestep
- Lower values indicate better noise prediction, which leads to better image generation

---

## Interpreting Multiple Metrics Together

When evaluating model performance, consider metrics together:

1. **For semantic quality:** Look at CLIP Score
2. **For visual quality:** Look at FID
3. **For spatial accuracy:** Look at mIoU and Class IoU
4. **For distribution matching:** Look at Class KL Divergence and Coverage Difference

**Example good model:**
- CLIP Score: > 0.2
- FID: < 30
- mIoU: > 0.6
- Class IoU: > 0.7
- Coverage Diff: < 0.02
- Class KL Divergence: < 0.2

**Example poor model:**
- CLIP Score: < 0.0
- FID: > 100
- mIoU: < 0.3
- Class IoU: < 0.4
- Coverage Diff: > 0.1
- Class KL Divergence: > 0.5

---

## Notes

1. **Per-batch approximations:** Some metrics (FID, CLIP Score) are computed per-batch and may vary. For accurate evaluation, accumulate features/metrics across all validation batches.

2. **Metric availability:** Not all metrics may be available for all experiments. Metrics are only computed if:
   - Required models are available (e.g., Inception for FID, CLIP for CLIP Score)
   - Required data is available (e.g., taxonomy for mIoU)
   - Evaluation is enabled in the config

3. **Invalid values:** The plotting code automatically filters out:
   - `NaN` values
   - `Inf` values
   - FID values > 1e6 (numerical errors)

4. **Plot files:** Each metric is saved as a separate plot: `{exp_name}_metric_{metric_name}.png`

---

## Troubleshooting

**FID is extremely large (> 1e6):**
- This indicates numerical issues, likely due to small batch size
- The code now includes safeguards, but if it persists, try:
  - Increasing batch size for evaluation
  - Checking if Inception model loaded correctly

**mIoU is not computed:**
- Check if taxonomy is available
- Check if LayoutSegmentor can segment the images
- Look for warnings in the logs

**CLIP Score is very low:**
- This might be normal if text/POV embeddings don't match well
- Check if CLIP model loaded correctly
- Consider using actual text prompts instead of embeddings

**Metrics are missing from plots:**
- Check if metrics have valid (non-NaN, non-Inf) values
- Check if metrics are being computed (look for warnings)
- Verify the metric names match what's in the CSV file

