#!/usr/bin/env python3
"""
Evaluation metrics module for generated layout images.

This module provides comprehensive evaluation metrics for comparing generated
layout images against ground truth layouts, including:
- Object extraction from color clusters
- Object counting per class
- Co-occurrence metrics (objects within radius r)
- Distribution comparison metrics
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Set
import numpy as np
from PIL import Image
from collections import defaultdict
from sklearn.cluster import DBSCAN
import json

from common.taxonomy import Taxonomy


class ObjectExtractor:
    """Extract objects from layout images using color clustering."""
    
    def __init__(self, taxonomy: Taxonomy, mode: str = "super", 
                 min_pixels_per_object: int = 10,
                 cluster_eps: float = 15.0,
                 min_samples_per_cluster: int = 5):
        """
        Initialize object extractor.
        
        Args:
            taxonomy: Taxonomy instance for color-to-category mapping
            mode: "super" or "category" - extract by super-category or category
            min_pixels_per_object: Minimum pixels for an object to be counted
            cluster_eps: DBSCAN eps parameter for clustering nearby pixels
            min_samples_per_cluster: DBSCAN min_samples parameter
        """
        self.taxonomy = taxonomy
        self.mode = mode
        self.min_pixels_per_object = min_pixels_per_object
        self.cluster_eps = cluster_eps
        self.min_samples_per_cluster = min_samples_per_cluster
        
        # Build color to category/super-category mapping
        self.color_to_label = self._build_color_mapping()
        
        # Background colors to skip
        self.background_colors = {
            (255, 255, 255), (240, 240, 240), 
            (200, 200, 200), (211, 211, 211),
            (250, 250, 250), (230, 230, 230)
        }
    
    def _build_color_mapping(self) -> Dict[Tuple[int, int, int], Dict]:
        """Build mapping from RGB colors to category/super-category labels."""
        color_to_label = {}
        id2color = self.taxonomy.data.get("id2color", {})
        
        if self.mode == "super":
            id2super = self.taxonomy.data.get("id2super", {})
            for super_id_str, super_name in id2super.items():
                color = id2color.get(super_id_str)
                if color:
                    rgb_tuple = tuple(color) if isinstance(color, list) else tuple(color)
                    color_to_label[rgb_tuple] = {
                        "label": super_name,
                        "label_id": int(super_id_str)
                    }
            # Also include wall (2053) which is commonly used
            wall_color = id2color.get("2053")
            if wall_color:
                rgb_tuple = tuple(wall_color) if isinstance(wall_color, list) else tuple(wall_color)
                wall_super_id = self.taxonomy.resolve_super(2053)
                if wall_super_id:
                    wall_super_name = self.taxonomy.id_to_name(wall_super_id)
                    color_to_label[rgb_tuple] = {
                        "label": wall_super_name,
                        "label_id": wall_super_id
                    }
        else:
            # Category mode
            id2category = self.taxonomy.data.get("id2category", {})
            for cat_id_str, cat_name in id2category.items():
                color = id2color.get(cat_id_str)
                if color:
                    rgb_tuple = tuple(color) if isinstance(color, list) else tuple(color)
                    color_to_label[rgb_tuple] = {
                        "label": cat_name,
                        "label_id": int(cat_id_str)
                    }
        
        return color_to_label
    
    def _find_color_clusters(self, points: List[Tuple[int, int]]) -> List[np.ndarray]:
        """Cluster nearby pixels of the same color into objects."""
        if len(points) == 0:
            return []
        
        pts = np.array(points, dtype=np.float32)
        if len(pts) < self.min_samples_per_cluster:
            return []
        
        if len(pts) == 1:
            return [pts]
        
        # Use DBSCAN to cluster nearby pixels
        clustering = DBSCAN(
            eps=self.cluster_eps, 
            min_samples=self.min_samples_per_cluster
        ).fit(pts)
        labels = clustering.labels_
        
        # Group points by cluster (-1 is noise, which we ignore)
        clusters = []
        for label in set(labels):
            if label == -1:
                continue
            cluster_mask = labels == label
            cluster_points = pts[cluster_mask]
            if len(cluster_points) >= self.min_pixels_per_object:
                clusters.append(cluster_points)
        
        return clusters
    
    def extract_objects(self, image: Union[Image.Image, np.ndarray, Path]) -> List[Dict]:
        """
        Extract objects from a layout image.
        
        Args:
            image: PIL Image, numpy array, or path to image file
        
        Returns:
            List of object dictionaries, each containing:
            - label: category/super-category name
            - label_id: category/super-category ID
            - center: (x, y) centroid of the object
            - pixel_count: number of pixels in the object
            - color: RGB color tuple
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        # Convert to numpy array
        if isinstance(image, Image.Image):
            img_array = np.array(image, dtype=np.uint8)
        else:
            img_array = image.copy()
            if img_array.dtype != np.uint8:
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
        
        h, w = img_array.shape[:2]
        
        # Group pixels by color
        color_regions = defaultdict(list)
        for y in range(h):
            for x in range(w):
                rgb = tuple(img_array[y, x])
                # Skip background colors
                if rgb in self.background_colors:
                    continue
                # Check if color is in taxonomy
                if rgb in self.color_to_label:
                    color_regions[rgb].append((x, y))
        
        # Extract objects from each color region
        objects = []
        for color, points in color_regions.items():
            label_info = self.color_to_label[color]
            # Cluster nearby pixels into separate objects
            clusters = self._find_color_clusters(points)
            
            for cluster in clusters:
                centroid = cluster.mean(axis=0)
                
                # Compute bounding box
                min_x, min_y = cluster.min(axis=0).astype(int)
                max_x, max_y = cluster.max(axis=0).astype(int)
                width = max_x - min_x + 1
                height = max_y - min_y + 1
                area = width * height
                aspect_ratio = width / height if height > 0 else 1.0
                
                objects.append({
                    "label": label_info["label"],
                    "label_id": label_info["label_id"],
                    "center": tuple(centroid.astype(int)),
                    "pixel_count": len(cluster),
                    "color": color,
                    "bbox": {
                        "min_x": int(min_x),
                        "min_y": int(min_y),
                        "max_x": int(max_x),
                        "max_y": int(max_y),
                        "width": int(width),
                        "height": int(height),
                        "area": int(area),
                        "aspect_ratio": float(aspect_ratio)
                    }
                })
        
        return objects


class LayoutEvaluator:
    """Evaluate generated layout images against ground truth."""
    
    def __init__(self, taxonomy: Taxonomy, mode: str = "super",
                 cooccurrence_radius: float = 0.15,
                 min_pixels_per_object: int = 10):
        """
        Initialize layout evaluator.
        
        Args:
            taxonomy: Taxonomy instance
            mode: "super" or "category" - evaluation mode
            cooccurrence_radius: Radius for co-occurrence (as fraction of image size, default 0.15 = 15%)
            min_pixels_per_object: Minimum pixels for an object to be counted
        """
        self.taxonomy = taxonomy
        self.mode = mode
        self.cooccurrence_radius = cooccurrence_radius
        self.extractor = ObjectExtractor(
            taxonomy=taxonomy,
            mode=mode,
            min_pixels_per_object=min_pixels_per_object
        )
    
    def count_objects(self, image: Union[Image.Image, np.ndarray, Path]) -> Dict[str, int]:
        """
        Count objects per class in an image.
        
        Args:
            image: Layout image to analyze
        
        Returns:
            Dictionary mapping class names to counts
        """
        objects = self.extractor.extract_objects(image)
        counts = defaultdict(int)
        for obj in objects:
            counts[obj["label"]] += 1
        return dict(counts)
    
    def compute_cooccurrence(self, image: Union[Image.Image, np.ndarray, Path],
                            class_a: str, class_b: str) -> Dict[str, float]:
        """
        Compute co-occurrence metrics between two classes.
        
        For each object of class_a, check if there's an object of class_b
        within radius r.
        
        Args:
            image: Layout image to analyze
            class_a: First class name (e.g., "Bed")
            class_b: Second class name (e.g., "Cabinet/Shelf/Desk")
        
        Returns:
            Dictionary with metrics:
            - cooccurrence_rate: Fraction of class_a objects that have class_b nearby
            - avg_distance: Average distance to nearest class_b object (for class_a objects)
            - min_distance: Minimum distance found
            - max_distance: Maximum distance found
        """
        objects = self.extractor.extract_objects(image)
        
        # Get image size for radius calculation
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        elif isinstance(image, Image.Image):
            img = image
        else:
            img = Image.fromarray(image)
        
        w, h = img.size
        radius_pixels = self.cooccurrence_radius * min(w, h)
        
        # Separate objects by class
        class_a_objects = [obj for obj in objects if obj["label"] == class_a]
        class_b_objects = [obj for obj in objects if obj["label"] == class_b]
        
        if len(class_a_objects) == 0:
            return {
                "cooccurrence_rate": 0.0,
                "avg_distance": float('inf'),
                "min_distance": float('inf'),
                "max_distance": float('inf'),
                "num_class_a": 0,
                "num_class_b": 0
            }
        
        if len(class_b_objects) == 0:
            return {
                "cooccurrence_rate": 0.0,
                "avg_distance": float('inf'),
                "min_distance": float('inf'),
                "max_distance": float('inf'),
                "num_class_a": len(class_a_objects),
                "num_class_b": 0
            }
        
        # For each class_a object, find nearest class_b object
        cooccurrences = 0
        distances = []
        
        for obj_a in class_a_objects:
            center_a = np.array(obj_a["center"])
            min_dist = float('inf')
            
            for obj_b in class_b_objects:
                center_b = np.array(obj_b["center"])
                dist = np.linalg.norm(center_a - center_b)
                min_dist = min(min_dist, dist)
            
            distances.append(min_dist)
            if min_dist <= radius_pixels:
                cooccurrences += 1
        
        cooccurrence_rate = cooccurrences / len(class_a_objects) if len(class_a_objects) > 0 else 0.0
        avg_distance = np.mean(distances) if distances else float('inf')
        min_distance = np.min(distances) if distances else float('inf')
        max_distance = np.max(distances) if distances else float('inf')
        
        return {
            "cooccurrence_rate": float(cooccurrence_rate),
            "avg_distance": float(avg_distance),
            "min_distance": float(min_distance),
            "max_distance": float(max_distance),
            "num_class_a": len(class_a_objects),
            "num_class_b": len(class_b_objects)
        }
    
    def compute_all_cooccurrences(self, image: Union[Image.Image, np.ndarray, Path],
                                  class_pairs: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Dict]:
        """
        Compute co-occurrence metrics for multiple class pairs.
        
        Args:
            image: Layout image to analyze
            class_pairs: List of (class_a, class_b) tuples. If None, computes for all pairs.
        
        Returns:
            Dictionary mapping "(class_a, class_b)" to co-occurrence metrics
        """
        objects = self.extractor.extract_objects(image)
        unique_classes = set(obj["label"] for obj in objects)
        
        if class_pairs is None:
            # Generate all pairs
            unique_classes_list = sorted(unique_classes)
            class_pairs = [
                (a, b) for a in unique_classes_list 
                for b in unique_classes_list 
                if a != b
            ]
        
        results = {}
        for class_a, class_b in class_pairs:
            key = f"{class_a}__{class_b}"
            results[key] = self.compute_cooccurrence(image, class_a, class_b)
        
        return results
    
    def compare_distributions(self, generated_counts: Dict[str, int],
                            ground_truth_counts: Dict[str, int]) -> Dict[str, float]:
        """
        Compare object count distributions between generated and ground truth.
        
        Args:
            generated_counts: Dictionary of class -> count for generated image
            ground_truth_counts: Dictionary of class -> count for ground truth
        
        Returns:
            Dictionary with comparison metrics:
            - kl_divergence: KL divergence (requires smoothing)
            - total_variation: Total variation distance
            - l1_distance: L1 distance between normalized distributions
            - l2_distance: L2 distance between normalized distributions
            - intersection_over_union: IoU of class sets
        """
        # Get all unique classes
        all_classes = set(generated_counts.keys()) | set(ground_truth_counts.keys())
        
        if len(all_classes) == 0:
            return {
                "kl_divergence": 0.0,
                "total_variation": 0.0,
                "l1_distance": 0.0,
                "l2_distance": 0.0,
                "intersection_over_union": 1.0
            }
        
        # Normalize to probabilities
        gen_total = sum(generated_counts.values())
        gt_total = sum(ground_truth_counts.values())
        
        gen_probs = np.array([generated_counts.get(c, 0) / gen_total if gen_total > 0 else 0 
                             for c in all_classes])
        gt_probs = np.array([ground_truth_counts.get(c, 0) / gt_total if gt_total > 0 else 0 
                            for c in all_classes])
        
        # Add small epsilon to avoid log(0) in KL divergence
        epsilon = 1e-10
        gen_probs_smooth = gen_probs + epsilon
        gt_probs_smooth = gt_probs + epsilon
        gen_probs_smooth = gen_probs_smooth / gen_probs_smooth.sum()
        gt_probs_smooth = gt_probs_smooth / gt_probs_smooth.sum()
        
        # KL divergence: D_KL(P||Q) = sum(P * log(P/Q))
        kl_div = np.sum(gen_probs_smooth * np.log(gen_probs_smooth / gt_probs_smooth))
        
        # Total variation distance: 0.5 * sum(|P - Q|)
        tv_dist = 0.5 * np.sum(np.abs(gen_probs - gt_probs))
        
        # L1 distance
        l1_dist = np.sum(np.abs(gen_probs - gt_probs))
        
        # L2 distance
        l2_dist = np.sqrt(np.sum((gen_probs - gt_probs) ** 2))
        
        # Intersection over Union of class sets
        gen_classes = set(generated_counts.keys())
        gt_classes = set(ground_truth_counts.keys())
        intersection = len(gen_classes & gt_classes)
        union = len(gen_classes | gt_classes)
        iou = intersection / union if union > 0 else 0.0
        
        return {
            "kl_divergence": float(kl_div),
            "total_variation": float(tv_dist),
            "l1_distance": float(l1_dist),
            "l2_distance": float(l2_dist),
            "intersection_over_union": float(iou)
        }
    
    def analyze_bbox_statistics(self, image: Union[Image.Image, np.ndarray, Path]) -> Dict:
        """
        Analyze bounding box statistics for objects in an image.
        
        Args:
            image: Layout image to analyze
        
        Returns:
            Dictionary with bbox statistics per class:
            - width, height, area, aspect_ratio statistics
            - Shape consistency metrics (coefficient of variation)
        """
        objects = self.extractor.extract_objects(image)
        
        # Group objects by class
        bbox_by_class = defaultdict(list)
        for obj in objects:
            if "bbox" in obj:
                bbox_by_class[obj["label"]].append(obj["bbox"])
        
        # Compute statistics per class
        bbox_stats = {}
        for class_name, bboxes in bbox_by_class.items():
            if len(bboxes) == 0:
                continue
            
            widths = [b["width"] for b in bboxes]
            heights = [b["height"] for b in bboxes]
            areas = [b["area"] for b in bboxes]
            aspect_ratios = [b["aspect_ratio"] for b in bboxes]
            
            # Compute statistics
            bbox_stats[class_name] = {
                "count": len(bboxes),
                "width": {
                    "mean": float(np.mean(widths)),
                    "std": float(np.std(widths)),
                    "min": int(np.min(widths)),
                    "max": int(np.max(widths)),
                    "cv": float(np.std(widths) / np.mean(widths)) if np.mean(widths) > 0 else 0.0  # Coefficient of variation
                },
                "height": {
                    "mean": float(np.mean(heights)),
                    "std": float(np.std(heights)),
                    "min": int(np.min(heights)),
                    "max": int(np.max(heights)),
                    "cv": float(np.std(heights) / np.mean(heights)) if np.mean(heights) > 0 else 0.0
                },
                "area": {
                    "mean": float(np.mean(areas)),
                    "std": float(np.std(areas)),
                    "min": int(np.min(areas)),
                    "max": int(np.max(areas)),
                    "cv": float(np.std(areas) / np.mean(areas)) if np.mean(areas) > 0 else 0.0
                },
                "aspect_ratio": {
                    "mean": float(np.mean(aspect_ratios)),
                    "std": float(np.std(aspect_ratios)),
                    "min": float(np.min(aspect_ratios)),
                    "max": float(np.max(aspect_ratios)),
                    "cv": float(np.std(aspect_ratios) / np.mean(aspect_ratios)) if np.mean(aspect_ratios) > 0 else 0.0
                }
            }
        
        return bbox_stats
    
    def analyze_bbox_density(self, image: Union[Image.Image, np.ndarray, Path]) -> Dict:
        """
        Analyze bounding box density (objects per unit area).
        
        Args:
            image: Layout image to analyze
        
        Returns:
            Dictionary with density metrics:
            - overall_density: objects per pixel
            - density_by_class: density per class
            - spatial_distribution: distribution of object centers
        """
        objects = self.extractor.extract_objects(image)
        
        # Get image size
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        elif isinstance(image, Image.Image):
            img = image
        else:
            img = Image.fromarray(image)
        
        w, h = img.size
        total_area = w * h
        
        # Overall density
        overall_density = len(objects) / total_area if total_area > 0 else 0.0
        
        # Density by class
        density_by_class = {}
        for obj in objects:
            class_name = obj["label"]
            if class_name not in density_by_class:
                density_by_class[class_name] = 0
            density_by_class[class_name] += 1
        
        # Normalize by area
        for class_name in density_by_class:
            density_by_class[class_name] = density_by_class[class_name] / total_area if total_area > 0 else 0.0
        
        # Spatial distribution (center positions normalized to [0, 1])
        centers_x = []
        centers_y = []
        for obj in objects:
            cx, cy = obj["center"]
            centers_x.append(cx / w if w > 0 else 0.0)
            centers_y.append(cy / h if h > 0 else 0.0)
        
        spatial_distribution = {
            "centers_x": {
                "mean": float(np.mean(centers_x)) if centers_x else 0.0,
                "std": float(np.std(centers_x)) if centers_x else 0.0,
                "min": float(np.min(centers_x)) if centers_x else 0.0,
                "max": float(np.max(centers_x)) if centers_x else 0.0
            },
            "centers_y": {
                "mean": float(np.mean(centers_y)) if centers_y else 0.0,
                "std": float(np.std(centers_y)) if centers_y else 0.0,
                "min": float(np.min(centers_y)) if centers_y else 0.0,
                "max": float(np.max(centers_y)) if centers_y else 0.0
            }
        }
        
        return {
            "overall_density": float(overall_density),
            "density_by_class": {k: float(v) for k, v in density_by_class.items()},
            "spatial_distribution": spatial_distribution,
            "total_objects": len(objects),
            "image_area": total_area
        }
    
    def analyze_shape_similarity(self, image: Union[Image.Image, np.ndarray, Path],
                                class_name: str) -> Dict:
        """
        Analyze shape similarity for objects of the same class.
        
        Measures how similar bbox shapes are within the same class.
        Lower variation = more consistent shapes.
        
        Args:
            image: Layout image to analyze
            class_name: Class to analyze
        
        Returns:
            Dictionary with shape similarity metrics:
            - shape_consistency: Coefficient of variation for aspect ratios (lower = more consistent)
            - size_consistency: Coefficient of variation for areas
            - shape_diversity: Standard deviation of aspect ratios
        """
        objects = self.extractor.extract_objects(image)
        
        # Filter objects by class
        class_objects = [obj for obj in objects if obj["label"] == class_name]
        
        if len(class_objects) < 2:
            return {
                "shape_consistency": 0.0,
                "size_consistency": 0.0,
                "shape_diversity": 0.0,
                "num_objects": len(class_objects)
            }
        
        # Extract bbox properties
        aspect_ratios = []
        areas = []
        widths = []
        heights = []
        
        for obj in class_objects:
            if "bbox" in obj:
                bbox = obj["bbox"]
                aspect_ratios.append(bbox["aspect_ratio"])
                areas.append(bbox["area"])
                widths.append(bbox["width"])
                heights.append(bbox["height"])
        
        if len(aspect_ratios) == 0:
            return {
                "shape_consistency": 0.0,
                "size_consistency": 0.0,
                "shape_diversity": 0.0,
                "num_objects": len(class_objects)
            }
        
        # Compute consistency metrics (coefficient of variation)
        # Lower CV = more consistent
        ar_mean = np.mean(aspect_ratios)
        ar_std = np.std(aspect_ratios)
        shape_consistency = 1.0 - (ar_std / ar_mean) if ar_mean > 0 else 0.0  # Normalized to [0, 1]
        shape_consistency = max(0.0, min(1.0, shape_consistency))  # Clamp to [0, 1]
        
        area_mean = np.mean(areas)
        area_std = np.std(areas)
        size_consistency = 1.0 - (area_std / area_mean) if area_mean > 0 else 0.0
        size_consistency = max(0.0, min(1.0, size_consistency))
        
        # Shape diversity (standard deviation of aspect ratios)
        shape_diversity = float(ar_std)
        
        return {
            "shape_consistency": float(shape_consistency),
            "size_consistency": float(size_consistency),
            "shape_diversity": float(shape_diversity),
            "num_objects": len(class_objects),
            "aspect_ratio_mean": float(ar_mean),
            "aspect_ratio_std": float(ar_std),
            "area_mean": float(area_mean),
            "area_std": float(area_std)
        }
    
    def compare_bbox_statistics(self, generated_image: Union[Image.Image, np.ndarray, Path],
                                ground_truth_image: Union[Image.Image, np.ndarray, Path]) -> Dict:
        """
        Compare bbox statistics between generated and ground truth images.
        
        Args:
            generated_image: Generated layout image
            ground_truth_image: Ground truth layout image
        
        Returns:
            Dictionary with comparison metrics per class
        """
        gen_bbox = self.analyze_bbox_statistics(generated_image)
        gt_bbox = self.analyze_bbox_statistics(ground_truth_image)
        
        all_classes = set(gen_bbox.keys()) | set(gt_bbox.keys())
        
        comparison = {}
        for class_name in all_classes:
            gen_stats = gen_bbox.get(class_name, {})
            gt_stats = gt_bbox.get(class_name, {})
            
            if not gen_stats or not gt_stats:
                continue
            
            comparison[class_name] = {
                "generated": gen_stats,
                "ground_truth": gt_stats,
                "differences": {
                    "width_mean_diff": gen_stats.get("width", {}).get("mean", 0) - gt_stats.get("width", {}).get("mean", 0),
                    "height_mean_diff": gen_stats.get("height", {}).get("mean", 0) - gt_stats.get("height", {}).get("mean", 0),
                    "area_mean_diff": gen_stats.get("area", {}).get("mean", 0) - gt_stats.get("area", {}).get("mean", 0),
                    "aspect_ratio_mean_diff": gen_stats.get("aspect_ratio", {}).get("mean", 0) - gt_stats.get("aspect_ratio", {}).get("mean", 0),
                    "shape_consistency_diff": gen_stats.get("aspect_ratio", {}).get("cv", 0) - gt_stats.get("aspect_ratio", {}).get("cv", 0)
                }
            }
        
        return comparison
    
    def evaluate_single(self, generated_image: Union[Image.Image, np.ndarray, Path],
                       ground_truth_image: Union[Image.Image, np.ndarray, Path],
                       class_pairs: Optional[List[Tuple[str, str]]] = None,
                       include_bbox_analysis: bool = True) -> Dict:
        """
        Evaluate a single generated image against ground truth.
        
        Args:
            generated_image: Generated layout image
            ground_truth_image: Ground truth layout image
            class_pairs: Optional list of class pairs for co-occurrence analysis
            include_bbox_analysis: If True, include bbox and shape analysis
        
        Returns:
            Dictionary with all evaluation metrics
        """
        # Count objects
        gen_counts = self.count_objects(generated_image)
        gt_counts = self.count_objects(ground_truth_image)
        
        # Compare distributions
        dist_metrics = self.compare_distributions(gen_counts, gt_counts)
        
        # Compute co-occurrences for generated image
        gen_cooccurrences = self.compute_all_cooccurrences(generated_image, class_pairs)
        
        # Compute co-occurrences for ground truth
        gt_cooccurrences = self.compute_all_cooccurrences(ground_truth_image, class_pairs)
        
        # Compare co-occurrence rates
        cooccurrence_comparison = {}
        for key in set(gen_cooccurrences.keys()) | set(gt_cooccurrences.keys()):
            gen_cooc = gen_cooccurrences.get(key, {})
            gt_cooc = gt_cooccurrences.get(key, {})
            
            gen_rate = gen_cooc.get("cooccurrence_rate", 0.0)
            gt_rate = gt_cooc.get("cooccurrence_rate", 0.0)
            
            cooccurrence_comparison[key] = {
                "generated_rate": gen_rate,
                "ground_truth_rate": gt_rate,
                "difference": gen_rate - gt_rate,
                "absolute_difference": abs(gen_rate - gt_rate)
            }
        
        result = {
            "object_counts": {
                "generated": gen_counts,
                "ground_truth": gt_counts
            },
            "distribution_metrics": dist_metrics,
            "cooccurrence_metrics": {
                "generated": gen_cooccurrences,
                "ground_truth": gt_cooccurrences,
                "comparison": cooccurrence_comparison
            }
        }
        
        # Add bbox analysis if requested
        if include_bbox_analysis:
            gen_density = self.analyze_bbox_density(generated_image)
            gt_density = self.analyze_bbox_density(ground_truth_image)
            bbox_comparison = self.compare_bbox_statistics(generated_image, ground_truth_image)
            
            result["bbox_analysis"] = {
                "generated_density": gen_density,
                "ground_truth_density": gt_density,
                "bbox_statistics_comparison": bbox_comparison,
                "density_difference": {
                    "overall": gen_density["overall_density"] - gt_density["overall_density"],
                    "by_class": {
                        k: gen_density["density_by_class"].get(k, 0.0) - gt_density["density_by_class"].get(k, 0.0)
                        for k in set(gen_density["density_by_class"].keys()) | set(gt_density["density_by_class"].keys())
                    }
                }
            }
        
        return result
    
    def evaluate_batch(self, generated_images: List[Union[Image.Image, np.ndarray, Path]],
                      ground_truth_images: List[Union[Image.Image, np.ndarray, Path]],
                      class_pairs: Optional[List[Tuple[str, str]]] = None) -> Dict:
        """
        Evaluate a batch of generated images against ground truth.
        
        Args:
            generated_images: List of generated layout images
            ground_truth_images: List of ground truth layout images
            class_pairs: Optional list of class pairs for co-occurrence analysis
        
        Returns:
            Dictionary with aggregated metrics across the batch
        """
        if len(generated_images) != len(ground_truth_images):
            raise ValueError("Number of generated and ground truth images must match")
        
        # Evaluate each pair
        individual_results = []
        for gen_img, gt_img in zip(generated_images, ground_truth_images):
            result = self.evaluate_single(gen_img, gt_img, class_pairs)
            individual_results.append(result)
        
        # Aggregate metrics
        all_gen_counts = defaultdict(int)
        all_gt_counts = defaultdict(int)
        
        for result in individual_results:
            for class_name, count in result["object_counts"]["generated"].items():
                all_gen_counts[class_name] += count
            for class_name, count in result["object_counts"]["ground_truth"].items():
                all_gt_counts[class_name] += count
        
        # Average distribution metrics
        avg_dist_metrics = {}
        for key in individual_results[0]["distribution_metrics"].keys():
            values = [r["distribution_metrics"][key] for r in individual_results]
            avg_dist_metrics[key] = float(np.mean(values))
            avg_dist_metrics[f"{key}_std"] = float(np.std(values))
        
        # Average co-occurrence comparison
        all_cooc_keys = set()
        for result in individual_results:
            all_cooc_keys.update(result["cooccurrence_metrics"]["comparison"].keys())
        
        avg_cooc_comparison = {}
        for key in all_cooc_keys:
            gen_rates = []
            gt_rates = []
            diffs = []
            
            for result in individual_results:
                comp = result["cooccurrence_metrics"]["comparison"].get(key, {})
                gen_rates.append(comp.get("generated_rate", 0.0))
                gt_rates.append(comp.get("ground_truth_rate", 0.0))
                diffs.append(comp.get("difference", 0.0))
            
            avg_cooc_comparison[key] = {
                "avg_generated_rate": float(np.mean(gen_rates)),
                "avg_ground_truth_rate": float(np.mean(gt_rates)),
                "avg_difference": float(np.mean(diffs)),
                "std_difference": float(np.std(diffs))
            }
        
        return {
            "num_samples": len(generated_images),
            "aggregate_object_counts": {
                "generated": dict(all_gen_counts),
                "ground_truth": dict(all_gt_counts)
            },
            "average_distribution_metrics": avg_dist_metrics,
            "average_cooccurrence_comparison": avg_cooc_comparison,
            "individual_results": individual_results
        }


def evaluate_layouts(generated_paths: List[Union[str, Path]],
                     ground_truth_paths: List[Union[str, Path]],
                     taxonomy_path: Union[str, Path] = "config/taxonomy.json",
                     mode: str = "super",
                     cooccurrence_radius: float = 0.15,
                     class_pairs: Optional[List[Tuple[str, str]]] = None,
                     output_path: Optional[Union[str, Path]] = None) -> Dict:
    """
    Convenience function to evaluate generated layouts against ground truth.
    
    Args:
        generated_paths: List of paths to generated layout images
        ground_truth_paths: List of paths to ground truth layout images
        taxonomy_path: Path to taxonomy.json file
        mode: "super" or "category" - evaluation mode
        cooccurrence_radius: Radius for co-occurrence (as fraction of image size)
        class_pairs: Optional list of class pairs for co-occurrence analysis
        output_path: Optional path to save results as JSON
    
    Returns:
        Dictionary with evaluation results
    """
    taxonomy = Taxonomy(taxonomy_path)
    evaluator = LayoutEvaluator(
        taxonomy=taxonomy,
        mode=mode,
        cooccurrence_radius=cooccurrence_radius
    )
    
    results = evaluator.evaluate_batch(
        generated_paths,
        ground_truth_paths,
        class_pairs=class_pairs
    )
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate generated layout images")
    parser.add_argument("--generated", required=True, nargs="+", 
                       help="Paths to generated layout images")
    parser.add_argument("--ground_truth", required=True, nargs="+",
                       help="Paths to ground truth layout images")
    parser.add_argument("--taxonomy", default="config/taxonomy.json",
                       help="Path to taxonomy.json file")
    parser.add_argument("--mode", default="super", choices=["super", "category"],
                       help="Evaluation mode: super or category")
    parser.add_argument("--cooccurrence_radius", type=float, default=0.15,
                       help="Radius for co-occurrence (as fraction of image size)")
    parser.add_argument("--output", help="Path to save results JSON")
    
    args = parser.parse_args()
    
    results = evaluate_layouts(
        generated_paths=args.generated,
        ground_truth_paths=args.ground_truth,
        taxonomy_path=args.taxonomy,
        mode=args.mode,
        cooccurrence_radius=args.cooccurrence_radius,
        output_path=args.output
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    print(f"Number of samples: {results['num_samples']}")
    print(f"\nAverage Distribution Metrics:")
    for key, value in results["average_distribution_metrics"].items():
        if not key.endswith("_std"):
            print(f"  {key}: {value:.4f}")
    
    print(f"\nAggregate Object Counts:")
    print("  Generated:")
    for class_name, count in sorted(results["aggregate_object_counts"]["generated"].items()):
        print(f"    {class_name}: {count}")
    print("  Ground Truth:")
    for class_name, count in sorted(results["aggregate_object_counts"]["ground_truth"].items()):
        print(f"    {class_name}: {count}")

