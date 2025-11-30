#!/usr/bin/env python3
"""
Clean Dataset - Semantic Quality Assessment

Checks for semantic content in segmentation images to determine quality.
Instead of deleting files, outputs a rejections CSV that can be joined with manifest.

Rejection Criteria for Layouts and POVs:
1. Must have Floor (≥ min_pixels)
2. Must have Wall (≥ min_pixels)
3. Must have Door OR Window (≥ min_pixels)
4. Must not be mostly black (render failure)
5. Must not be mostly white (overexposed)
6. Content area (floor+wall+furniture) must be reasonable (not just background)

Additional POV-specific checks:
7. Floor shouldn't dominate (camera pointing down)
8. Wall shouldn't dominate (staring at wall)

Rejection Propagation:
- If scene layout is rejected -> all rooms and POVs in that scene are rejected
- If room layout is rejected -> all POVs for that room are rejected

Segmentation Colors (configurable via --color-map):
- Floor: (180, 180, 180)
- Wall: (60, 60, 60)  
- Door: (255, 100, 100)
- Window: (100, 200, 255)
- Background/Ceiling: (0, 0, 0) - black

Usage:
    python clean_dataset.py \\
        --dataset-root dataset_v2 \\
        --output rejections.csv

    python clean_dataset.py \\
        --dataset-root dataset_v2 \\
        --shard-file shards/shard_000.txt \\
        --output rejections_shard_000.csv
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple
from enum import Enum, auto

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Default Segmentation Color Map
# =============================================================================

DEFAULT_SEG_COLORS = {
    # Structure
    "Floor": (180, 180, 180),
    "Wall": (60, 60, 60),
    # Openings
    "Door": (255, 100, 100),
    "Window": (100, 200, 255),
    # Background (white = outside/empty space/ceiling in top-down views)
    "Background": (255, 255, 255),
}

# Color matching tolerance (for compression artifacts)
COLOR_TOLERANCE = 10


# =============================================================================
# Rejection Reasons
# =============================================================================

class RejectionReason(Enum):
    """Reasons for rejection."""
    # Semantic content issues
    NO_FLOOR = auto()
    NO_WALL = auto()
    NO_DOOR_OR_WINDOW = auto()
    
    # Render quality issues
    MOSTLY_BLACK = auto()       # Failed render (all black)
    TOO_LITTLE_CONTENT = auto() # Too much background, not enough room content
    
    # POV-specific issues
    FLOOR_DOMINATES = auto()    # Camera pointing down
    WALL_DOMINATES = auto()     # Staring at wall
    
    # Propagation
    SCENE_LAYOUT_REJECTED = auto()
    ROOM_LAYOUT_REJECTED = auto()


@dataclass
class SemanticReport:
    """Report for semantic content check."""
    path: Path
    total_pixels: int = 0
    
    # Semantic class presence
    has_floor: bool = False
    has_wall: bool = False
    has_door: bool = False
    has_window: bool = False
    
    # Pixel counts
    floor_pixels: int = 0
    wall_pixels: int = 0
    door_pixels: int = 0
    window_pixels: int = 0
    background_pixels: int = 0  # White = outside/ceiling
    other_pixels: int = 0  # Furniture and other objects
    
    # Quality metrics
    black_fraction: float = 0.0  # For detecting render failures
    white_fraction: float = 0.0  # For reference (white is valid background)
    content_fraction: float = 0.0  # Non-background content
    
    # Fractions for dominance checks
    floor_fraction: float = 0.0
    wall_fraction: float = 0.0


@dataclass
class RejectionRecord:
    """A single rejection decision."""
    image_type: str  # 'scene_layout', 'room_layout', 'pov'
    scene_id: str
    room_id: str
    file_path: str
    rejected: bool
    rejection_reasons: List[str]
    propagated_from: str  # 'self', 'scene', 'room'
    
    # Semantic details
    has_floor: bool
    has_wall: bool
    has_door: bool
    has_window: bool
    
    # Pixel counts
    floor_pixels: int
    wall_pixels: int
    door_pixels: int
    window_pixels: int
    background_pixels: int
    content_fraction: float


# =============================================================================
# Semantic Checking
# =============================================================================

def check_semantic_content(
    image_path: Path,
    seg_colors: Dict[str, Tuple[int, int, int]],
    min_pixels: int = 100,
    max_black_fraction: float = 0.95,
    max_white_fraction: float = 0.95,
    min_content_fraction: float = 0.05,
    max_floor_fraction: float = 0.85,  # For POV dominance check
    max_wall_fraction: float = 0.90,   # For POV dominance check
) -> SemanticReport:
    """
    Check for semantic content in a segmentation image.
    """
    report = SemanticReport(path=image_path)
    
    try:
        img = Image.open(image_path).convert("RGB")
        pixels = np.array(img)
    except Exception as e:
        logger.warning(f"Could not load {image_path}: {e}")
        return report
    
    height, width = pixels.shape[:2]
    report.total_pixels = height * width
    
    # Check for pure black (render failures) - not white since white is valid background
    # Pure black = (0,0,0) pixels
    black_mask = np.all(pixels < 5, axis=2)  # Very dark pixels
    report.black_fraction = black_mask.mean()
    
    # White fraction (for reference, but white is valid background)
    white_mask = np.all(pixels > 250, axis=2)  # Very bright pixels
    report.white_fraction = white_mask.mean()
    
    # Count pixels for each semantic class
    accounted_pixels = np.zeros((height, width), dtype=bool)
    
    for class_name, color in seg_colors.items():
        color_arr = np.array(color)
        diff = np.abs(pixels.astype(np.int16) - color_arr)
        mask = np.all(diff <= COLOR_TOLERANCE, axis=2)
        pixel_count = mask.sum()
        
        # Mark these pixels as accounted for
        accounted_pixels |= mask
        
        if class_name == "Floor":
            report.floor_pixels = pixel_count
            report.has_floor = pixel_count >= min_pixels
            report.floor_fraction = pixel_count / report.total_pixels
        elif class_name == "Wall":
            report.wall_pixels = pixel_count
            report.has_wall = pixel_count >= min_pixels
            report.wall_fraction = pixel_count / report.total_pixels
        elif class_name == "Door":
            report.door_pixels = pixel_count
            report.has_door = pixel_count >= min_pixels
        elif class_name == "Window":
            report.window_pixels = pixel_count
            report.has_window = pixel_count >= min_pixels
        elif class_name == "Background":
            report.background_pixels = pixel_count
    
    # Everything else is "other" (furniture, objects, etc.)
    report.other_pixels = (~accounted_pixels).sum()
    
    # Content = everything that's not background
    content_pixels = report.total_pixels - report.background_pixels
    report.content_fraction = content_pixels / report.total_pixels if report.total_pixels > 0 else 0
    
    return report


def evaluate_layout(
    report: SemanticReport,
    min_pixels: int = 100,
    max_black_fraction: float = 0.95,
    min_content_fraction: float = 0.05,
) -> List[RejectionReason]:
    """Evaluate a layout image and return rejection reasons."""
    reasons = []
    
    # Check render quality - black means failed render
    if report.black_fraction > max_black_fraction:
        reasons.append(RejectionReason.MOSTLY_BLACK)
    
    # Check content amount (too much background = empty/invalid)
    if report.content_fraction < min_content_fraction:
        reasons.append(RejectionReason.TOO_LITTLE_CONTENT)
    
    # Check semantic requirements
    if not report.has_floor:
        reasons.append(RejectionReason.NO_FLOOR)
    if not report.has_wall:
        reasons.append(RejectionReason.NO_WALL)
    if not report.has_door and not report.has_window:
        reasons.append(RejectionReason.NO_DOOR_OR_WINDOW)
    
    return reasons


def evaluate_pov(
    report: SemanticReport,
    min_pixels: int = 100,
    max_black_fraction: float = 0.95,
    min_content_fraction: float = 0.05,
    max_floor_fraction: float = 0.85,
    max_wall_fraction: float = 0.90,
) -> List[RejectionReason]:
    """Evaluate a POV image and return rejection reasons."""
    reasons = []
    
    # Check render quality - black means failed render
    if report.black_fraction > max_black_fraction:
        reasons.append(RejectionReason.MOSTLY_BLACK)
    
    # Check content amount
    if report.content_fraction < min_content_fraction:
        reasons.append(RejectionReason.TOO_LITTLE_CONTENT)
    
    # Check semantic requirements
    if not report.has_floor:
        reasons.append(RejectionReason.NO_FLOOR)
    if not report.has_wall:
        reasons.append(RejectionReason.NO_WALL)
    if not report.has_door and not report.has_window:
        reasons.append(RejectionReason.NO_DOOR_OR_WINDOW)
    
    # POV-specific: check for bad viewpoints
    if report.floor_fraction > max_floor_fraction:
        reasons.append(RejectionReason.FLOOR_DOMINATES)
    if report.wall_fraction > max_wall_fraction:
        reasons.append(RejectionReason.WALL_DOMINATES)
    
    return reasons


# =============================================================================
# File Discovery
# =============================================================================

def extract_scene_id(filename: str) -> Optional[str]:
    """Extract scene ID from filename."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 1 and len(parts[0]) >= 8:
        return parts[0]
    return None


def extract_room_id(filename: str, scene_id: str) -> Optional[str]:
    """Extract room ID from filename."""
    stem = Path(filename).stem
    
    if stem.startswith(scene_id + "_"):
        remainder = stem[len(scene_id) + 1:]
    else:
        return None
    
    room_types = [
        "Bedroom", "LivingRoom", "DiningRoom", "Kitchen", "Bathroom",
        "Balcony", "Storage", "Corridor", "OtherRoom", "Library",
        "MasterBedroom", "SecondBedroom", "KidsRoom", "Study", "Entrance"
    ]
    
    for rt in room_types:
        if remainder.startswith(rt):
            after_rt = remainder[len(rt):]
            if after_rt.startswith("_"):
                next_part = after_rt[1:].split("_")[0]
                if next_part.isdigit():
                    return f"{rt}_{next_part}"
                else:
                    return rt
            else:
                return rt
    
    return None


def find_scene_layouts(dataset_root: Path, scene_ids: Optional[Set[str]] = None) -> Dict[str, Path]:
    """Find all scene layout segmentation files."""
    layouts_seg_dir = dataset_root / "layouts" / "seg"
    if not layouts_seg_dir.exists():
        return {}
    
    scene_layouts = {}
    
    for path in layouts_seg_dir.glob("*_seg_layout.png"):
        scene_id = extract_scene_id(path.name)
        if scene_id is None:
            continue
        if scene_ids is not None and scene_id not in scene_ids:
            continue
        
        room_id = extract_room_id(path.name, scene_id)
        if room_id is None:
            scene_layouts[scene_id] = path
    
    return scene_layouts


def find_room_layouts(dataset_root: Path, scene_ids: Optional[Set[str]] = None) -> Dict[Tuple[str, str], Path]:
    """Find all room layout segmentation files."""
    layouts_seg_dir = dataset_root / "layouts" / "seg"
    if not layouts_seg_dir.exists():
        return {}
    
    room_layouts = {}
    
    for path in layouts_seg_dir.glob("*_seg_layout.png"):
        scene_id = extract_scene_id(path.name)
        if scene_id is None:
            continue
        if scene_ids is not None and scene_id not in scene_ids:
            continue
        
        room_id = extract_room_id(path.name, scene_id)
        if room_id is not None:
            room_layouts[(scene_id, room_id)] = path
    
    return room_layouts


def find_povs(dataset_root: Path, scene_ids: Optional[Set[str]] = None) -> Dict[Tuple[str, str, str], Path]:
    """Find all POV segmentation files."""
    povs_seg_dir = dataset_root / "povs" / "seg"
    if not povs_seg_dir.exists():
        return {}
    
    povs = {}
    
    for path in povs_seg_dir.glob("*_seg_pov.png"):
        scene_id = extract_scene_id(path.name)
        if scene_id is None:
            continue
        if scene_ids is not None and scene_id not in scene_ids:
            continue
        
        stem = path.stem
        prefix = stem.replace("_seg_pov", "")
        
        if prefix.startswith(scene_id + "_"):
            remainder = prefix[len(scene_id) + 1:]
        else:
            continue
        
        parts = remainder.rsplit("_", 1)
        if len(parts) == 2:
            room_id = parts[0]
            pov_id = parts[1]
        else:
            continue
        
        povs[(scene_id, room_id, pov_id)] = path
    
    return povs


# =============================================================================
# Main Processing
# =============================================================================

def process_dataset(
    dataset_root: Path,
    seg_colors: Dict[str, Tuple[int, int, int]],
    scene_ids: Optional[Set[str]] = None,
    min_pixels: int = 100,
    max_black_fraction: float = 0.95,
    min_content_fraction: float = 0.05,
    max_floor_fraction: float = 0.85,
    max_wall_fraction: float = 0.90,
) -> List[RejectionRecord]:
    """Process dataset and generate rejection records."""
    records = []
    
    rejected_scenes: Set[str] = set()
    rejected_rooms: Set[Tuple[str, str]] = set()
    
    # ==========================================================================
    # Phase 1: Check Scene Layouts
    # ==========================================================================
    logger.info("Phase 1: Checking scene layouts...")
    scene_layouts = find_scene_layouts(dataset_root, scene_ids)
    logger.info(f"  Found {len(scene_layouts)} scene layouts")
    
    for scene_id, seg_path in tqdm(scene_layouts.items(), desc="Scene layouts"):
        report = check_semantic_content(seg_path, seg_colors, min_pixels)
        reasons = evaluate_layout(report, min_pixels, max_black_fraction, 
                                   min_content_fraction)
        
        rejected = len(reasons) > 0
        if rejected:
            rejected_scenes.add(scene_id)
        
        # Record for seg file
        records.append(RejectionRecord(
            image_type="scene_layout",
            scene_id=scene_id,
            room_id="",
            file_path=str(seg_path.relative_to(dataset_root)),
            rejected=rejected,
            rejection_reasons=[r.name for r in reasons],
            propagated_from="self",
            has_floor=report.has_floor,
            has_wall=report.has_wall,
            has_door=report.has_door,
            has_window=report.has_window,
            floor_pixels=report.floor_pixels,
            wall_pixels=report.wall_pixels,
            door_pixels=report.door_pixels,
            window_pixels=report.window_pixels,
            background_pixels=report.background_pixels,
            content_fraction=report.content_fraction,
        ))
        
        # Record for tex file
        tex_path = seg_path.parent.parent / "tex" / seg_path.name.replace("_seg_", "_tex_")
        if tex_path.exists():
            records.append(RejectionRecord(
                image_type="scene_layout",
                scene_id=scene_id,
                room_id="",
                file_path=str(tex_path.relative_to(dataset_root)),
                rejected=rejected,
                rejection_reasons=[r.name for r in reasons],
                propagated_from="self",
                has_floor=report.has_floor,
                has_wall=report.has_wall,
                has_door=report.has_door,
                has_window=report.has_window,
                floor_pixels=report.floor_pixels,
                wall_pixels=report.wall_pixels,
                door_pixels=report.door_pixels,
                window_pixels=report.window_pixels,
                background_pixels=report.background_pixels,
                content_fraction=report.content_fraction,
            ))
    
    logger.info(f"  Rejected scenes: {len(rejected_scenes)}")
    
    # ==========================================================================
    # Phase 2: Check Room Layouts
    # ==========================================================================
    logger.info("\nPhase 2: Checking room layouts...")
    room_layouts = find_room_layouts(dataset_root, scene_ids)
    logger.info(f"  Found {len(room_layouts)} room layouts")
    
    for (scene_id, room_id), seg_path in tqdm(room_layouts.items(), desc="Room layouts"):
        if scene_id in rejected_scenes:
            rejected = True
            propagated_from = "scene"
            reasons = [RejectionReason.SCENE_LAYOUT_REJECTED]
            report = check_semantic_content(seg_path, seg_colors, min_pixels)
        else:
            report = check_semantic_content(seg_path, seg_colors, min_pixels)
            reasons = evaluate_layout(report, min_pixels, max_black_fraction,
                                       min_content_fraction)
            rejected = len(reasons) > 0
            propagated_from = "self"
            
            if rejected:
                rejected_rooms.add((scene_id, room_id))
        
        records.append(RejectionRecord(
            image_type="room_layout",
            scene_id=scene_id,
            room_id=room_id,
            file_path=str(seg_path.relative_to(dataset_root)),
            rejected=rejected,
            rejection_reasons=[r.name for r in reasons],
            propagated_from=propagated_from,
            has_floor=report.has_floor,
            has_wall=report.has_wall,
            has_door=report.has_door,
            has_window=report.has_window,
            floor_pixels=report.floor_pixels,
            wall_pixels=report.wall_pixels,
            door_pixels=report.door_pixels,
            window_pixels=report.window_pixels,
            background_pixels=report.background_pixels,
            content_fraction=report.content_fraction,
        ))
        
        tex_path = seg_path.parent.parent / "tex" / seg_path.name.replace("_seg_", "_tex_")
        if tex_path.exists():
            records.append(RejectionRecord(
                image_type="room_layout",
                scene_id=scene_id,
                room_id=room_id,
                file_path=str(tex_path.relative_to(dataset_root)),
                rejected=rejected,
                rejection_reasons=[r.name for r in reasons],
                propagated_from=propagated_from,
                has_floor=report.has_floor,
                has_wall=report.has_wall,
                has_door=report.has_door,
                has_window=report.has_window,
                floor_pixels=report.floor_pixels,
                wall_pixels=report.wall_pixels,
                door_pixels=report.door_pixels,
                window_pixels=report.window_pixels,
                background_pixels=report.background_pixels,
                content_fraction=report.content_fraction,
            ))
    
    logger.info(f"  Rejected rooms (own issues): {len(rejected_rooms)}")
    
    # ==========================================================================
    # Phase 3: Check POVs
    # ==========================================================================
    logger.info("\nPhase 3: Checking POVs...")
    povs = find_povs(dataset_root, scene_ids)
    logger.info(f"  Found {len(povs)} POVs")
    
    pov_rejected_self = 0
    pov_rejected_propagated = 0
    
    for (scene_id, room_id, pov_id), seg_path in tqdm(povs.items(), desc="POVs"):
        if scene_id in rejected_scenes:
            rejected = True
            propagated_from = "scene"
            reasons = [RejectionReason.SCENE_LAYOUT_REJECTED]
            pov_rejected_propagated += 1
            report = check_semantic_content(seg_path, seg_colors, min_pixels)
        elif (scene_id, room_id) in rejected_rooms:
            rejected = True
            propagated_from = "room"
            reasons = [RejectionReason.ROOM_LAYOUT_REJECTED]
            pov_rejected_propagated += 1
            report = check_semantic_content(seg_path, seg_colors, min_pixels)
        else:
            report = check_semantic_content(seg_path, seg_colors, min_pixels)
            reasons = evaluate_pov(report, min_pixels, max_black_fraction,
                                    min_content_fraction,
                                    max_floor_fraction, max_wall_fraction)
            rejected = len(reasons) > 0
            propagated_from = "self"
            if rejected:
                pov_rejected_self += 1
        
        records.append(RejectionRecord(
            image_type="pov",
            scene_id=scene_id,
            room_id=room_id,
            file_path=str(seg_path.relative_to(dataset_root)),
            rejected=rejected,
            rejection_reasons=[r.name for r in reasons],
            propagated_from=propagated_from,
            has_floor=report.has_floor,
            has_wall=report.has_wall,
            has_door=report.has_door,
            has_window=report.has_window,
            floor_pixels=report.floor_pixels,
            wall_pixels=report.wall_pixels,
            door_pixels=report.door_pixels,
            window_pixels=report.window_pixels,
            background_pixels=report.background_pixels,
            content_fraction=report.content_fraction,
        ))
        
        tex_path = seg_path.parent.parent / "tex" / seg_path.name.replace("_seg_", "_tex_")
        if tex_path.exists():
            records.append(RejectionRecord(
                image_type="pov",
                scene_id=scene_id,
                room_id=room_id,
                file_path=str(tex_path.relative_to(dataset_root)),
                rejected=rejected,
                rejection_reasons=[r.name for r in reasons],
                propagated_from=propagated_from,
                has_floor=report.has_floor,
                has_wall=report.has_wall,
                has_door=report.has_door,
                has_window=report.has_window,
                floor_pixels=report.floor_pixels,
                wall_pixels=report.wall_pixels,
                door_pixels=report.door_pixels,
                window_pixels=report.window_pixels,
                background_pixels=report.background_pixels,
                content_fraction=report.content_fraction,
            ))
    
    logger.info(f"  POVs rejected (own issues): {pov_rejected_self}")
    logger.info(f"  POVs rejected (propagated): {pov_rejected_propagated}")
    
    return records


# =============================================================================
# Output
# =============================================================================

def write_rejections(records: List[RejectionRecord], output_path: Path):
    """Write rejection records to CSV."""
    if not records:
        logger.warning("No records to write")
        return
    
    logger.info(f"\nWriting {len(records)} records to {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    columns = [
        "image_type", "scene_id", "room_id", "file_path", "rejected",
        "rejection_reasons", "propagated_from",
        "has_floor", "has_wall", "has_door", "has_window",
        "floor_pixels", "wall_pixels", "door_pixels", "window_pixels",
        "background_pixels", "content_fraction",
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for record in records:
            writer.writerow({
                "image_type": record.image_type,
                "scene_id": record.scene_id,
                "room_id": record.room_id,
                "file_path": record.file_path,
                "rejected": record.rejected,
                "rejection_reasons": "|".join(record.rejection_reasons),
                "propagated_from": record.propagated_from,
                "has_floor": record.has_floor,
                "has_wall": record.has_wall,
                "has_door": record.has_door,
                "has_window": record.has_window,
                "floor_pixels": record.floor_pixels,
                "wall_pixels": record.wall_pixels,
                "door_pixels": record.door_pixels,
                "window_pixels": record.window_pixels,
                "background_pixels": record.background_pixels,
                "content_fraction": round(record.content_fraction, 4),
            })
    
    size_kb = output_path.stat().st_size / 1024
    logger.info(f"  Written: {size_kb:.2f} KB")
    
    # Summary
    total = len(records)
    rejected = sum(1 for r in records if r.rejected)
    logger.info(f"\nSummary:")
    logger.info(f"  Total records: {total}")
    logger.info(f"  Rejected: {rejected} ({100*rejected/total:.1f}%)")
    logger.info(f"  Accepted: {total - rejected} ({100*(total-rejected)/total:.1f}%)")
    
    # Breakdown by reason
    reason_counts: Dict[str, int] = {}
    for r in records:
        if r.rejected:
            for reason in r.rejection_reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    if reason_counts:
        logger.info(f"\nRejection reasons:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {reason}: {count}")


def load_scene_list(shard_file: Path) -> Set[str]:
    """Load scene IDs from a shard file."""
    scene_ids = set()
    if not shard_file.exists():
        logger.warning(f"Shard file not found: {shard_file}")
        return scene_ids
    
    with open(shard_file, "r", encoding="utf-8") as f:
        for line in f:
            scene_id = line.strip()
            if scene_id and not scene_id.startswith("#"):
                scene_ids.add(scene_id)
    
    return scene_ids


def load_color_map(color_map_path: Path) -> Dict[str, Tuple[int, int, int]]:
    """Load segmentation color map from JSON file."""
    with open(color_map_path, "r") as f:
        data = json.load(f)
    
    # Convert lists to tuples
    return {k: tuple(v) for k, v in data.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Check dataset quality using semantic content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shard-file", type=Path)
    parser.add_argument("--color-map", type=Path,
                        help="JSON file with segmentation colors (optional)")
    
    # Thresholds
    parser.add_argument("--min-pixels", type=int, default=100,
                        help="Min pixels for class presence (default: 100)")
    parser.add_argument("--max-black-fraction", type=float, default=0.95,
                        help="Max fraction of black pixels - render failure (default: 0.95)")
    parser.add_argument("--min-content-fraction", type=float, default=0.05,
                        help="Min fraction of non-background content (default: 0.05)")
    parser.add_argument("--max-floor-fraction", type=float, default=0.85,
                        help="Max floor fraction for POVs (default: 0.85)")
    parser.add_argument("--max-wall-fraction", type=float, default=0.90,
                        help="Max wall fraction for POVs (default: 0.90)")
    
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.dataset_root.exists():
        logger.error(f"Dataset root not found: {args.dataset_root}")
        return 1
    
    # Load color map
    if args.color_map:
        seg_colors = load_color_map(args.color_map)
        logger.info(f"Loaded color map from {args.color_map}")
    else:
        seg_colors = DEFAULT_SEG_COLORS
        logger.info("Using default color map")
    
    # Load scene IDs
    scene_ids = None
    if args.shard_file:
        logger.info(f"Loading scene IDs from: {args.shard_file}")
        scene_ids = load_scene_list(args.shard_file)
        logger.info(f"Loaded {len(scene_ids)} scene IDs")
        if len(scene_ids) == 0:
            logger.warning("No scene IDs found")
            return 0
    
    # Process
    records = process_dataset(
        args.dataset_root,
        seg_colors,
        scene_ids=scene_ids,
        min_pixels=args.min_pixels,
        max_black_fraction=args.max_black_fraction,
        min_content_fraction=args.min_content_fraction,
        max_floor_fraction=args.max_floor_fraction,
        max_wall_fraction=args.max_wall_fraction,
    )
    
    write_rejections(records, args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())