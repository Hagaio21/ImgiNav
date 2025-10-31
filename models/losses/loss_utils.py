"""
Utilities for loss computation.
Hardcoded mappings for segmentation (RGB -> class) and classification (room_id -> class).
"""

import torch
import numpy as np


# Hardcoded RGB -> class_index mapping
# Classes: All super-categories (1001-1009) except Structure (1008) + wall (2053) + background
# Excluded: Structure (1008), Unknown (0)
# Background: (240, 240, 240) from stage3
# Colors from taxonomy.json id2color
RGB_TO_CLASS = {
    (228, 26, 28): 0,      # Bed (1001)
    (55, 126, 184): 1,      # Cabinet/Shelf/Desk (1002)
    (77, 175, 74): 2,       # Chair (1003)
    (152, 78, 163): 3,      # Lighting (1004)
    (127, 127, 127): 4,     # Others (1005)
    (255, 127, 0): 5,       # Pier/Stool (1006)
    (255, 255, 51): 6,      # Sofa (1007)
    # Structure (1008) - EXCLUDED (color: [0, 0, 0])
    (166, 86, 40): 7,       # Table (1009)
    # Wall (2053)
    (200, 200, 200): 8,     # Wall (2053)
    # Background (240, 240, 240) from stage3
    (240, 240, 240): 9,     # Background
}

NUM_CLASSES = len(RGB_TO_CLASS)


# Hardcoded room_id -> class_index mapping for classification
# Room IDs from taxonomy (3000-3999 range) mapped to sequential class indices
# Room IDs can be strings ("0000") or integers (0, 3001, etc.)
# Based on common room types - adjust as needed
ROOM_ID_TO_CLASS = {
    "0000": 0,      # Scene
    0: 0,           # Scene (also handle as int)
    3001: 1,        # Aisle
    3002: 2,        # Auditorium
    3003: 3,        # Balcony
    3004: 4,        # BathRoom
    3005: 5,        # Bathroom
    3006: 6,        # BedRoom
    3007: 7,        # Bedroom
    3008: 8,        # CloakRoom
    3009: 9,        # Corridor
    3010: 10,        # Courtyard
    3011: 11,        # DiningRoom
    3012: 12,        # ElderlyRoom
    3013: 13,        # EquipmentRoom
    3014: 14,        # Garage
    3015: 15,        # Hallway
    3016: 16,        # KidsRoom
    3017: 17,        # Kitchen
    3018: 18,        # LaundryRoom
    3019: 19,        # Library
    3020: 20,        # LivingDiningRoom
    3021: 21,        # LivingRoom
    3022: 22,        # Lounge
    3023: 23,        # MasterBathroom
    3024: 24,        # MasterBedroom
    3025: 25,        # NannyRoom
    3026: 26,        # OtherRoom
    3027: 27,        # OtherSpace
    3028: 28,        # SecondBathroom
    3029: 29,        # SecondBedroom
    3030: 30,        # Stairwell
    3031: 31,        # StorageRoom
    3032: 32,        # Terrace
    3033: 33,        # non
}

NUM_ROOM_CLASSES = len(set(ROOM_ID_TO_CLASS.values()))


def create_seg_mask(image: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:

    # Normalize to [0, 255] uint8 if needed
    if image.dtype == torch.float32 or image.dtype == torch.float64:
        image = (image * 255).clamp(0, 255).to(torch.uint8)
    else:
        image = image.to(torch.uint8)
    
    # Handle batch dimension
    if image.dim() == 4:  # (B, 3, H, W)
        B, C, H, W = image.shape
        seg = torch.full((B, H, W), ignore_index, dtype=torch.long)
        for b in range(B):
            seg[b] = _segment_single(image[b], ignore_index=ignore_index)
        return seg
    elif image.dim() == 3:
        if image.shape[0] == 3:  # (3, H, W)
            return _segment_single(image, ignore_index=ignore_index)
        else:  # (H, W, 3)
            return _segment_single(image.permute(2, 0, 1), ignore_index=ignore_index)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def _segment_single(image: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    """Segment a single (3, H, W) image tensor."""
    _, H, W = image.shape
    seg = torch.full((H, W), ignore_index, dtype=torch.long)
    
    # Convert to numpy for efficient processing
    img_np = image.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    pixels = img_np.reshape(-1, 3)
    
    # Map each pixel
    for idx, rgb in enumerate(pixels):
        rgb_tuple = tuple(rgb.astype(int))
        if rgb_tuple in RGB_TO_CLASS:
            seg.view(-1)[idx] = RGB_TO_CLASS[rgb_tuple]
    
    return seg


def room_id_to_class_index(room_id: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Convert room_id (taxonomy ID 3000-3999 or "0000") to class index (0 to num_room_classes-1).
    
    Args:
        room_id: Room ID tensor, shape () or (N,). Can be int or string "0000"
        ignore_index: Value to use for unknown room IDs
    
    Returns:
        Class index tensor, same shape as input
    """
    if isinstance(room_id, torch.Tensor):
        room_id_np = room_id.cpu().numpy()
        device = room_id.device
    else:
        room_id_np = np.array(room_id)
        device = torch.device("cpu")
    
    # Helper to look up class index - handle both string "0000" and int 0
    def lookup_room_id(rid):
        # Try as string first (for "0000")
        rid_str = str(rid).zfill(4)  # Pad to 4 digits
        if rid_str in ROOM_ID_TO_CLASS:
            return ROOM_ID_TO_CLASS[rid_str]
        # Try as int
        rid_int = int(rid)
        return ROOM_ID_TO_CLASS.get(rid_int, ignore_index)
    
    # Convert to Python value(s) for lookup
    if room_id_np.ndim == 0:
        class_idx = lookup_room_id(room_id_np.item())
        return torch.tensor(class_idx, dtype=torch.long, device=device)
    else:
        # Batch case
        class_indices = torch.tensor(
            [lookup_room_id(rid) for rid in room_id_np.flatten()],
            dtype=torch.long,
            device=device
        )
        return class_indices.reshape(room_id.shape)

