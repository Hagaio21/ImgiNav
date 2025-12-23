import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

def analyze_dataset(manifest_path):
    """
    Analyze the dataset manifest and calculate statistics for the appendix tables.
    
    Args:
        manifest_path (str): Path to the CSV manifest file
    """
    
    # Read the CSV file
    print(f"Reading manifest from: {manifest_path}")
    df = pd.read_csv(manifest_path)
    
    print(f"Total rows in manifest: {len(df)}")
    print(f"Columns: {df.columns.tolist()}\n")
    
    # ========== TABLE 1: Dataset Scale ==========
    print("="*70)
    print("TABLE 1: DATASET SCALE")
    print("="*70)
    
    valid_scenes = df['scene_id'].nunique()
    print(f"Valid scenes: {valid_scenes:,}")
    
    total_rooms = df['room_id'].nunique()
    print(f"Total rooms: {total_rooms:,}")
    
    non_empty_rooms = df[df['is_empty'] == False]['room_id'].nunique()
    empty_rooms = df[df['is_empty'] == True]['room_id'].nunique()
    print(f"Non-empty rooms: {non_empty_rooms:,}")
    print(f"Empty rooms: {empty_rooms:,}")
    print(f"Total (check): {non_empty_rooms + empty_rooms}")
    
    total_pov_samples = len(df)
    print(f"Total POV samples: {total_pov_samples:,}")
    
    # POVs per room
    povs_per_room = df.groupby('room_id').size()
    min_povs = povs_per_room.min()
    max_povs = povs_per_room.max()
    mean_povs = povs_per_room.mean()
    print(f"POVs per room: {min_povs}--{max_povs} (mean: {mean_povs:.2f})")
    
    unique_layouts = df['layout_path'].nunique()
    print(f"Unique layouts: {unique_layouts:,}")
    
    # ========== TABLE 2: Room Type Distribution ==========
    print("\n" + "="*70)
    print("TABLE 2: ROOM TYPE DISTRIBUTION (TOP 10)")
    print("="*70)
    
    # Get unique rooms with their room types
    room_type_df = df.drop_duplicates(subset=['room_id'])[['room_type', 'room_id']]
    room_type_counts = room_type_df['room_type'].value_counts()
    
    total_rooms_check = room_type_counts.sum()
    print(f"Total rooms (check): {total_rooms_check}")
    
    top_10 = room_type_counts.head(10)
    print("\nTop 10 Room Types:")
    for room_type, count in top_10.items():
        percentage = (count / total_rooms) * 100
        print(f"  {room_type}: {count:,} ({percentage:.1f}%)")
    
    # ========== TABLE 3: Class Imbalance Statistics ==========
    print("\n" + "="*70)
    print("TABLE 3: CLASS IMBALANCE STATISTICS")
    print("="*70)
    
    empty_room_rate = (empty_rooms / total_rooms) * 100
    furnished_room_rate = (non_empty_rooms / total_rooms) * 100
    print(f"Empty room rate: {empty_room_rate:.1f}%")
    print(f"Rooms with furniture: {furnished_room_rate:.1f}%")
    
    # Furniture class distribution (among non-empty rooms)
    if 'furniture_count' in df.columns:
        # Count rooms by furniture count
        furniture_dist = df[df['is_empty'] == False].groupby('room_id')['furniture_count'].first()
        furniture_counts = furniture_dist.value_counts().sort_index()
        
        print("\nFurniture count distribution:")
        for count, freq in furniture_counts.items():
            pct = (freq / non_empty_rooms) * 100
            print(f"  {int(count)} furniture items: {freq:,} rooms ({pct:.1f}%)")
    
    # Most and least common furniture counts
    if not furniture_dist.empty:
        most_common_furniture = furniture_dist.value_counts().idxmax()
        most_common_count = furniture_dist.value_counts().max()
        most_common_pct = (most_common_count / non_empty_rooms) * 100
        
        least_common_furniture = furniture_dist.value_counts().idxmin()
        least_common_count = furniture_dist.value_counts().min()
        least_common_pct = (least_common_count / non_empty_rooms) * 100
        
        print(f"\nMost common: {int(most_common_furniture)} furniture items ({most_common_pct:.1f}%)")
        print(f"Least common: {int(least_common_furniture)} furniture items ({least_common_pct:.1f}%)")
        
        max_ratio = most_common_count / least_common_count if least_common_count > 0 else 0
        print(f"Class imbalance ratio: {max_ratio:.1f}:1")
    
    # ========== TABLE 4: Layout Diversity ==========
    print("\n" + "="*70)
    print("TABLE 4: LAYOUT DIVERSITY AND EMBEDDING COLLISION ANALYSIS")
    print("="*70)
    
    unique_layouts_non_empty = df[df['is_empty'] == False]['layout_path'].nunique()
    print(f"Unique non-empty layouts: {unique_layouts_non_empty:,}")
    
    # Rotation augmentations per room
    rotation_per_room = df.groupby('room_id')['rotation_angle_deg'].nunique()
    print(f"Rotation augmentations per room: {rotation_per_room.min():.0f}--{rotation_per_room.max():.0f}")
    
    # Unique graph texts
    unique_graph_texts = df['graph_text_path'].nunique()
    print(f"Unique graph texts: {unique_graph_texts:,}")
    
    # Layouts per unique graph text
    graphs_to_layouts = df.groupby('graph_text_path')['layout_path'].nunique()
    print(f"Layouts per unique graph text: {graphs_to_layouts.min():.0f}--{graphs_to_layouts.max():.0f}")
    print(f"Mean layouts per graph: {graphs_to_layouts.mean():.1f}")
    
    # POV types distribution
    print("\n" + "="*70)
    print("ADDITIONAL STATISTICS")
    print("="*70)
    
    pov_type_counts = df['pov_type'].value_counts()
    print("\nPOV Type Distribution:")
    for pov_type, count in pov_type_counts.items():
        pct = (count / total_pov_samples) * 100
        print(f"  {pov_type}: {count:,} ({pct:.1f}%)")
    
    if 'rejected' in df.columns:
        rejected_count = df['rejected'].sum()
        rejected_pct = (rejected_count / len(df)) * 100
        print(f"\nRejected samples: {rejected_count:,} ({rejected_pct:.1f}%)")
    
    # Door and window statistics
    avg_doors = df.groupby('room_id')['door_count'].first().mean()
    avg_windows = df.groupby('room_id')['window_count'].first().mean()
    print(f"\nAverage doors per room: {avg_doors:.2f}")
    print(f"Average windows per room: {avg_windows:.2f}")
    
    # ========== Generate LaTeX Table Snippets ==========
    print("\n" + "="*70)
    print("LATEX TABLE SNIPPETS")
    print("="*70)
    
    print("\n% TABLE 2: Room Type Distribution (copy-paste into LaTeX)")
    print("\\begin{tabular}{lr}")
    print("\\toprule")
    print("\\textbf{Room Type} & \\textbf{Percentage} \\\\")
    print("\\midrule")
    for room_type, count in top_10.items():
        percentage = (count / total_rooms) * 100
        print(f"{room_type} & {percentage:.1f}\\% \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    
    print("\n% TABLE 3: Class Imbalance (copy-paste into LaTeX)")
    if not furniture_dist.empty:
        print(f"Empty room rate & {empty_room_rate:.0f}\\% \\\\")
        print(f"Rooms with furniture & {furnished_room_rate:.0f}\\% \\\\")
        print(f"Most common class & {int(most_common_furniture)} items ({most_common_pct:.0f}\\%) \\\\")
        print(f"Least common class & {int(least_common_furniture)} items ({least_common_pct:.0f}\\%) \\\\")
        print(f"Class imbalance ratio & {max_ratio:.1f}:1 \\\\")
    
    # Create a summary dictionary
    summary = {
        'valid_scenes': valid_scenes,
        'total_rooms': total_rooms,
        'non_empty_rooms': non_empty_rooms,
        'empty_rooms': empty_rooms,
        'total_pov_samples': total_pov_samples,
        'povs_per_room_min': min_povs,
        'povs_per_room_max': max_povs,
        'povs_per_room_mean': mean_povs,
        'unique_layouts': unique_layouts,
        'unique_layouts_non_empty': unique_layouts_non_empty,
        'top_10_room_types': top_10.to_dict(),
        'empty_room_rate': empty_room_rate,
        'furnished_room_rate': furnished_room_rate,
        'unique_graph_texts': unique_graph_texts,
    }
    
    return summary, df

if __name__ == "__main__":
    # Update this path to your actual manifest file
    manifest_path = r"C:\Users\Hagai.LAPTOP-QAG9263N\Desktop\Thesis\manifest_seg_pov_normalized_with_latents_cleaned.csv"
    
    try:
        summary, df = analyze_dataset(manifest_path)
        print("\n" + "="*70)
        print("Analysis complete!")
        print("="*70)
    except FileNotFoundError:
        print(f"Error: File not found at {manifest_path}")
        print("Please check the path and try again.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()