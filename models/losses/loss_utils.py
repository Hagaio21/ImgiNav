"""
Utilities for loss computation.
Loads mappings from YAML config files for segmentation (RGB -> class) and classification (room_id -> class).
"""

import torch
import numpy as np
import yaml
from pathlib import Path

# Load RGB to class mapping from YAML
_LOSS_DIR = Path(__file__).parent
_RGB_CONFIG_PATH = _LOSS_DIR / "rgb_to_class.yaml"
_ROOM_CONFIG_PATH = _LOSS_DIR / "room_id_to_class.yaml"

def _load_rgb_to_class():
    """Load RGB to class mapping from YAML file."""
    with open(_RGB_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    rgb_to_class = {}
    for entry in config.values():
        rgb_tuple = tuple(entry["rgb"])
        class_idx = entry["class_index"]
        rgb_to_class[rgb_tuple] = class_idx
    
    return rgb_to_class

def _load_room_id_to_class():
    """Load room_id to class mapping from YAML file."""
    with open(_ROOM_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    room_id_to_class = {}
    for entry in config.values():
        room_id = entry["room_id"]
        class_idx = entry["class_index"]
        # Handle both string "0000" and int
        if isinstance(room_id, str):
            room_id_to_class[room_id] = class_idx
        room_id_to_class[int(room_id)] = class_idx  # Also map as int
    
    return room_id_to_class

# Load mappings at module import
RGB_TO_CLASS = _load_rgb_to_class()
ROOM_ID_TO_CLASS = _load_room_id_to_class()

# Number of classes = number of unique class indices
NUM_CLASSES = len(set(RGB_TO_CLASS.values()))
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

