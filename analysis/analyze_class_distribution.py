#!/usr/bin/env python3
"""
Analyze class distribution in the dataset.
Classes are defined by:
1. sample_type: "scene" vs "room" (or "pov")
2. room_id: specific room types (3001-3033) or "0000" for scenes

This script:
- Computes statistics per class (scene vs room_id)
- Shows distribution within rooms
- Identifies rare classes
- Suggests a rare class threshold for grouping
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
import sys
from collections import defaultdict
import yaml

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load room_id to class mapping
def load_room_id_mapping():
    """Load room_id to class name mapping from YAML."""
    yaml_path = Path(__file__).parent.parent / "models" / "losses" / "room_id_to_class.yaml"
    if not yaml_path.exists():
        print(f"Warning: {yaml_path} not found, using room_id as class name")
        return {}
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    room_id_to_name = {}
    for room_name, room_data in data.items():
        if isinstance(room_data, dict) and "room_id" in room_data:
            room_id = str(room_data["room_id"])
            room_id_to_name[room_id] = room_name
    
    return room_id_to_name


def analyze_class_distribution(manifest_path, output_dir, rare_threshold_percentile=10):
    """
    Analyze class distribution in the dataset.
    
    Args:
        manifest_path: Path to CSV manifest
        output_dir: Output directory for results
        rare_threshold_percentile: Percentile below which classes are considered rare (default: 10th percentile)
    """
    from common.utils import safe_mkdir, write_json
    
    output_dir = Path(output_dir)
    safe_mkdir(output_dir)
    
    # Load manifest
    print(f"Loading manifest from {manifest_path}...")
    df = pd.read_csv(manifest_path)
    print(f"Loaded {len(df)} samples")
    
    # Load room_id to name mapping
    room_id_to_name = load_room_id_mapping()
    
    # Normalize room_id to string for consistent handling
    df["room_id_str"] = df["room_id"].astype(str)
    
    # Get sample_type if available, otherwise infer from room_id
    if "sample_type" in df.columns:
        sample_type_col = "sample_type"
    elif "type" in df.columns:
        sample_type_col = "type"
    else:
        # Infer from room_id: "0000" or 0 means scene
        df["sample_type"] = df["room_id_str"].apply(
            lambda x: "scene" if (x == "0000" or x == "0" or pd.isna(x)) else "room"
        )
        sample_type_col = "sample_type"
    
    # Create class identifier: scene vs room_id
    def get_class_id(row):
        if row[sample_type_col] == "scene" or row["room_id_str"] in ["0000", "0", "nan"]:
            return "scene"
        else:
            return row["room_id_str"]
    
    df["class_id"] = df.apply(get_class_id, axis=1)
    
    # ============================================================
    # 1. Overall class distribution
    # ============================================================
    print("\n" + "="*60)
    print("1. Overall Class Distribution")
    print("="*60)
    
    class_counts = df["class_id"].value_counts().sort_values(ascending=False)
    total_samples = len(df)
    
    class_stats = []
    for class_id, count in class_counts.items():
        percentage = (count / total_samples) * 100
        class_name = room_id_to_name.get(class_id, class_id)
        class_stats.append({
            "class_id": class_id,
            "class_name": class_name,
            "count": int(count),
            "percentage": float(percentage)
        })
        print(f"  {class_name:20s} (id: {class_id:>6s}): {count:6d} samples ({percentage:5.2f}%)")
    
    # ============================================================
    # 2. Scene vs Room distribution
    # ============================================================
    print("\n" + "="*60)
    print("2. Scene vs Room Distribution")
    print("="*60)
    
    scene_count = (df[sample_type_col] == "scene").sum()
    room_count = (df[sample_type_col] == "room").sum()
    pov_count = (df[sample_type_col] == "pov").sum() if "pov" in df[sample_type_col].values else 0
    
    print(f"  Scenes: {scene_count:6d} ({scene_count/total_samples*100:5.2f}%)")
    print(f"  Rooms:  {room_count:6d} ({room_count/total_samples*100:5.2f}%)")
    if pov_count > 0:
        print(f"  POVs:   {pov_count:6d} ({pov_count/total_samples*100:5.2f}%)")
    
    # ============================================================
    # 3. Room ID distribution (excluding scenes)
    # ============================================================
    print("\n" + "="*60)
    print("3. Room ID Distribution (excluding scenes)")
    print("="*60)
    
    room_df = df[df[sample_type_col] != "scene"].copy()
    if len(room_df) > 0:
        room_id_counts = room_df["room_id_str"].value_counts().sort_values(ascending=False)
        total_rooms = len(room_df)
        
        room_stats = []
        for room_id, count in room_id_counts.items():
            percentage = (count / total_rooms) * 100
            room_name = room_id_to_name.get(room_id, room_id)
            room_stats.append({
                "room_id": room_id,
                "room_name": room_name,
                "count": int(count),
                "percentage": float(percentage)
            })
            print(f"  {room_name:20s} (id: {room_id:>6s}): {count:6d} samples ({percentage:5.2f}%)")
    else:
        room_stats = []
        print("  No room samples found")
    
    # ============================================================
    # 4. Distribution within scenes (rooms per scene)
    # ============================================================
    print("\n" + "="*60)
    print("4. Distribution of Room Types within Scenes")
    print("="*60)
    
    if "scene_id" in df.columns:
        scene_room_dist = defaultdict(lambda: defaultdict(int))
        
        for _, row in df.iterrows():
            scene_id = row.get("scene_id", "unknown")
            if pd.isna(scene_id):
                continue
            room_id = row["room_id_str"]
            if room_id not in ["0000", "0"]:
                room_name = room_id_to_name.get(room_id, room_id)
                scene_room_dist[scene_id][room_name] += 1
        
        # Statistics per scene
        rooms_per_scene = [len(rooms) for rooms in scene_room_dist.values()]
        if rooms_per_scene:
            print(f"  Average unique room types per scene: {np.mean(rooms_per_scene):.2f}")
            print(f"  Min room types per scene: {np.min(rooms_per_scene)}")
            print(f"  Max room types per scene: {np.max(rooms_per_scene)}")
            print(f"  Scenes with only 1 room type: {sum(1 for x in rooms_per_scene if x == 1)}")
            print(f"  Scenes with 2-5 room types: {sum(1 for x in rooms_per_scene if 2 <= x <= 5)}")
            print(f"  Scenes with 6+ room types: {sum(1 for x in rooms_per_scene if x >= 6)}")
    else:
        print("  No scene_id column found, skipping scene-level analysis")
        scene_room_dist = {}
    
    # ============================================================
    # 5. Identify rare classes
    # ============================================================
    print("\n" + "="*60)
    print("5. Rare Class Identification")
    print("="*60)
    
    # Calculate threshold based on percentile
    counts_array = np.array([stat["count"] for stat in class_stats])
    threshold_count = np.percentile(counts_array, rare_threshold_percentile)
    
    print(f"  Rare class threshold (count < {threshold_count:.0f}, {rare_threshold_percentile}th percentile)")
    print(f"  Total classes: {len(class_stats)}")
    
    rare_classes = []
    common_classes = []
    
    for stat in class_stats:
        if stat["count"] < threshold_count:
            rare_classes.append(stat)
        else:
            common_classes.append(stat)
    
    print(f"  Rare classes: {len(rare_classes)}")
    print(f"  Common classes: {len(common_classes)}")
    
    if rare_classes:
        print("\n  Rare classes list:")
        for stat in sorted(rare_classes, key=lambda x: x["count"]):
            print(f"    {stat['class_name']:20s} (id: {stat['class_id']:>6s}): {stat['count']:6d} samples")
    
    # ============================================================
    # 6. Suggested weighting strategy
    # ============================================================
    print("\n" + "="*60)
    print("6. Suggested Weighting Strategy")
    print("="*60)
    
    # Group rare classes into a single "rare" category
    rare_class_ids = [stat["class_id"] for stat in rare_classes]
    rare_total_count = sum(stat["count"] for stat in rare_classes)
    
    # Create mapping: class_id -> grouped_class_id
    class_grouping = {}
    for stat in class_stats:
        if stat["class_id"] in rare_class_ids:
            class_grouping[stat["class_id"]] = "rare"
        else:
            class_grouping[stat["class_id"]] = stat["class_id"]
    
    # Compute weights for grouped classes
    grouped_counts = defaultdict(int)
    for class_id, grouped_id in class_grouping.items():
        count = next(stat["count"] for stat in class_stats if stat["class_id"] == class_id)
        grouped_counts[grouped_id] += count
    
    max_count = max(grouped_counts.values())
    grouped_weights = {class_id: max_count / count for class_id, count in grouped_counts.items()}
    
    print(f"  Grouped classes into {len(grouped_counts)} categories:")
    for grouped_id, count in sorted(grouped_counts.items(), key=lambda x: -x[1]):
        weight = grouped_weights[grouped_id]
        if grouped_id == "rare":
            print(f"    {grouped_id:20s}: {count:6d} samples, weight={weight:.3f} (includes {len(rare_classes)} rare classes)")
        else:
            class_name = room_id_to_name.get(grouped_id, grouped_id)
            print(f"    {class_name:20s}: {count:6d} samples, weight={weight:.3f}")
    
    # ============================================================
    # Save results
    # ============================================================
    results = {
        "total_samples": int(total_samples),
        "scene_count": int(scene_count),
        "room_count": int(room_count),
        "pov_count": int(pov_count),
        "total_classes": len(class_stats),
        "rare_threshold_count": float(threshold_count),
        "rare_threshold_percentile": rare_threshold_percentile,
        "rare_classes_count": len(rare_classes),
        "common_classes_count": len(common_classes),
        "class_statistics": class_stats,
        "room_statistics": room_stats,
        "rare_classes": [{"class_id": s["class_id"], "class_name": s["class_name"], "count": s["count"]} for s in rare_classes],
        "class_grouping": class_grouping,
        "grouped_weights": {k: float(v) for k, v in grouped_weights.items()},
        "grouped_counts": {k: int(v) for k, v in grouped_counts.items()}
    }
    
    write_json(results, output_dir / "class_distribution_stats.json")
    
    # ============================================================
    # Create visualizations
    # ============================================================
    print("\n" + "="*60)
    print("7. Creating Visualizations")
    print("="*60)
    
    # Plot 1: Class distribution (top 20)
    plt.figure(figsize=(14, 8))
    top_classes = class_stats[:20]
    class_names = [s["class_name"] for s in top_classes]
    counts = [s["count"] for s in top_classes]
    
    plt.barh(range(len(class_names)), counts)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Sample Count")
    plt.title(f"Top 20 Class Distribution (Total: {total_samples} samples)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution_top20.png", dpi=200)
    plt.close()
    
    # Plot 2: All classes (sorted by count)
    plt.figure(figsize=(14, max(8, len(class_stats) * 0.3)))
    all_class_names = [s["class_name"] for s in class_stats]
    all_counts = [s["count"] for s in class_stats]
    
    colors = ['red' if stat["class_id"] in rare_class_ids else 'steelblue' for stat in class_stats]
    
    plt.barh(range(len(all_class_names)), all_counts, color=colors)
    plt.yticks(range(len(all_class_names)), all_class_names)
    plt.xlabel("Sample Count")
    plt.title(f"All Class Distribution (Red = Rare, Blue = Common)")
    plt.axvline(x=threshold_count, color='orange', linestyle='--', label=f'Rare threshold ({threshold_count:.0f})')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution_all.png", dpi=200)
    plt.close()
    
    # Plot 3: Scene vs Room pie chart
    plt.figure(figsize=(8, 8))
    labels = []
    sizes = []
    if scene_count > 0:
        labels.append("Scene")
        sizes.append(scene_count)
    if room_count > 0:
        labels.append("Room")
        sizes.append(room_count)
    if pov_count > 0:
        labels.append("POV")
        sizes.append(pov_count)
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title("Scene vs Room vs POV Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "scene_room_distribution.png", dpi=200)
    plt.close()
    
    # Plot 4: Room ID distribution (if rooms exist)
    if len(room_stats) > 0:
        plt.figure(figsize=(14, max(8, len(room_stats) * 0.3)))
        room_names = [s["room_name"] for s in room_stats]
        room_counts = [s["count"] for s in room_stats]
        
        rare_room_ids = [s["room_id"] for s in rare_classes if s["class_id"] != "scene"]
        room_colors = ['red' if s["room_id"] in rare_room_ids else 'steelblue' for s in room_stats]
        
        plt.barh(range(len(room_names)), room_counts, color=room_colors)
        plt.yticks(range(len(room_names)), room_names)
        plt.xlabel("Sample Count")
        plt.title("Room ID Distribution (Red = Rare, Blue = Common)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_dir / "room_id_distribution.png", dpi=200)
        plt.close()
    
    # Plot 5: Pie chart of all classes
    plt.figure(figsize=(12, 12))
    
    # Prepare data for pie chart
    # If too many classes, show top N and group the rest
    max_classes_in_pie = 15
    if len(class_stats) > max_classes_in_pie:
        # Show top N-1 classes individually, group the rest
        top_classes = class_stats[:max_classes_in_pie - 1]
        other_classes = class_stats[max_classes_in_pie - 1:]
        other_total = sum(s["count"] for s in other_classes)
        
        pie_labels = [s["class_name"] for s in top_classes] + [f"Others ({len(other_classes)} classes)"]
        pie_counts = [s["count"] for s in top_classes] + [other_total]
        pie_colors = ['red' if stat["class_id"] in rare_class_ids else 'steelblue' for stat in top_classes] + ['lightgray']
    else:
        # Show all classes
        pie_labels = [s["class_name"] for s in class_stats]
        pie_counts = [s["count"] for s in class_stats]
        pie_colors = ['red' if stat["class_id"] in rare_class_ids else 'steelblue' for stat in class_stats]
    
    # Create pie chart with seaborn styling
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", len(pie_labels)) if len(pie_labels) <= 10 else sns.color_palette("Set3", len(pie_labels))
    
    # Use custom colors for rare/common distinction if not too many classes
    if len(class_stats) <= max_classes_in_pie:
        colors = pie_colors
    
    wedges, texts, autotexts = plt.pie(
        pie_counts,
        labels=pie_labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 9}
    )
    
    # Improve text readability
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(8)
    
    plt.title(f"Class Distribution Pie Chart\n(Total: {total_samples} samples, {len(class_stats)} classes)", 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution_pie.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot 6: Pie chart of grouped classes (if grouping was done)
    if class_grouping and len(grouped_counts) > 0:
        plt.figure(figsize=(10, 10))
        
        grouped_labels = []
        grouped_sizes = []
        grouped_colors_list = []
        
        # Sort by count for better visualization
        sorted_groups = sorted(grouped_counts.items(), key=lambda x: -x[1])
        
        for grouped_id, count in sorted_groups:
            if grouped_id == "rare":
                grouped_labels.append(f"Rare ({len(rare_classes)} classes)")
                grouped_colors_list.append('coral')
            else:
                class_name = room_id_to_name.get(grouped_id, grouped_id)
                grouped_labels.append(class_name)
                grouped_colors_list.append('steelblue')
            grouped_sizes.append(count)
        
        sns.set_style("whitegrid")
        colors = sns.color_palette("husl", len(grouped_labels))
        
        wedges, texts, autotexts = plt.pie(
            grouped_sizes,
            labels=grouped_labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=grouped_colors_list,
            textprops={'fontsize': 10}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.title(f"Grouped Class Distribution (Rare Classes Combined)\n(Total: {total_samples} samples)", 
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_dir / "class_distribution_grouped_pie.png", dpi=200, bbox_inches='tight')
        plt.close()
    
    print(f"\nResults saved to {output_dir}")
    print(f"  - class_distribution_stats.json")
    print(f"  - class_distribution_top20.png")
    print(f"  - class_distribution_all.png")
    print(f"  - scene_room_distribution.png")
    print(f"  - class_distribution_pie.png")
    if len(room_stats) > 0:
        print(f"  - room_id_distribution.png")
    if class_grouping and len(grouped_counts) > 0:
        print(f"  - class_distribution_grouped_pie.png")
    
    return results


def load_room_id_to_name():
    """Load room_id to name mapping."""
    yaml_path = Path(__file__).parent.parent / "models" / "losses" / "room_id_to_class.yaml"
    if not yaml_path.exists():
        return {}
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    room_id_to_name = {}
    for room_name, room_data in data.items():
        if isinstance(room_data, dict) and "room_id" in room_data:
            room_id = str(room_data["room_id"])
            room_id_to_name[room_id] = room_name
    
    return room_id_to_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze class distribution in dataset")
    parser.add_argument("--manifest", required=True, help="Path to dataset CSV manifest")
    parser.add_argument("--output_dir", required=True, help="Output directory for analysis results")
    parser.add_argument("--rare_threshold_percentile", type=float, default=10.0,
                        help="Percentile below which classes are considered rare (default: 10.0)")
    args = parser.parse_args()
    
    analyze_class_distribution(
        args.manifest,
        args.output_dir,
        rare_threshold_percentile=args.rare_threshold_percentile
    )

