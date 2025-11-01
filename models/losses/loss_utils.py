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
_ROOM_CONFIG_PATH = _LOSS_DIR / "room_scene_to_class.yaml"  # Changed to binary room/scene mapping

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
NUM_ROOM_CLASSES = 2  # Binary classification: room (0) vs scene (1)

# Pre-compute RGB lookup tensors for batched operations
# Create tensors: (num_classes, 3) for RGB values and (num_classes,) for class indices
_rgb_values = []
_class_indices = []
for rgb_tuple, class_idx in RGB_TO_CLASS.items():
    _rgb_values.append(list(rgb_tuple))
    _class_indices.append(class_idx)

RGB_VALUES_TENSOR = torch.tensor(_rgb_values, dtype=torch.uint8)  # (num_classes, 3)
CLASS_INDICES_TENSOR = torch.tensor(_class_indices, dtype=torch.long)  # (num_classes,)

# Cache lookup tensors per device to avoid repeated transfers
_device_rgb_cache = {}
_device_class_cache = {}

def _get_lookup_tensors(device):
    """Get or create lookup tensors for a device, with caching."""
    # Convert device to string key for caching (handles both str and torch.device)
    device_key = str(device)
    if device_key not in _device_rgb_cache:
        _device_rgb_cache[device_key] = RGB_VALUES_TENSOR.to(device)
        _device_class_cache[device_key] = CLASS_INDICES_TENSOR.to(device)
    return _device_rgb_cache[device_key], _device_class_cache[device_key]


def create_seg_mask(image: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    """
    Convert RGB image tensor to segmentation mask.
    Optimized GPU version that keeps all operations on device.
    
    Accepts images in either:
    - [-1, 1] range (normalized for tanh)
    - [0, 1] range (normalized for sigmoid)
    - [0, 255] range (uint8)
    """
    device = image.device
    
    # Normalize to [0, 255] uint8 if needed
    if image.dtype == torch.float32 or image.dtype == torch.float64:
        # Check if image is in [-1, 1] range (tanh normalization)
        # If min < -0.5, assume it's [-1, 1], convert to [0, 1] first
        if image.min() < -0.5:
            image = (image + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        # Convert [0, 1] to [0, 255]
        image = (image * 255).clamp(0, 255).to(torch.uint8)
    else:
        image = image.to(torch.uint8)
    
    # Ensure image is on correct device
    image = image.to(device)
    
    # Handle batch dimension
    if image.dim() == 4:  # (B, 3, H, W)
        B, C, H, W = image.shape
        seg_list = []
        for b in range(B):
            seg_list.append(_segment_single(image[b], ignore_index=ignore_index))
        return torch.stack(seg_list)
    elif image.dim() == 3:
        if image.shape[0] == 3:  # (3, H, W)
            return _segment_single(image, ignore_index=ignore_index)
        else:  # (H, W, 3)
            return _segment_single(image.permute(2, 0, 1), ignore_index=ignore_index)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def _segment_single(image: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    """Segment a single (3, H, W) image tensor - fully vectorized batched version."""
    _, H, W = image.shape
    
    # Ensure image is uint8 on the same device
    # Handle both [-1, 1] and [0, 1] float ranges
    if image.dtype != torch.uint8:
        # Check if image is in [-1, 1] range (tanh normalization)
        if image.min() < -0.5:
            image = (image + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        # Convert [0, 1] to [0, 255] uint8
        image = (image * 255).clamp(0, 255).to(torch.uint8)
    
    device = image.device
    
    # Reshape to (H*W, 3) for vectorized processing
    image_flat = image.permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3)
    
    # Get cached lookup tensors (avoids repeated device transfers)
    rgb_values, class_indices = _get_lookup_tensors(device)  # (num_classes, 3), (num_classes,)
    
    # Batched comparison: compare all pixels against all RGB values at once
    # image_flat: (H*W, 3), rgb_values: (num_classes, 3)
    # Expand dimensions for broadcasting: (H*W, 1, 3) vs (1, num_classes, 3)
    image_expanded = image_flat.unsqueeze(1)  # (H*W, 1, 3)
    rgb_expanded = rgb_values.unsqueeze(0)  # (1, num_classes, 3)
    
    # Compare all pixels with all RGB values: (H*W, num_classes, 3)
    matches = (image_expanded == rgb_expanded).all(dim=2)  # (H*W, num_classes)
    
    # Find which class (if any) each pixel matches
    # matches is (H*W, num_classes) boolean tensor
    # We want to find the first True value in each row (pixel)
    match_indices = matches.long().argmax(dim=1)  # (H*W,) - index of matching class, or 0 if no match
    has_match = matches.any(dim=1)  # (H*W,) - whether pixel matched any class
    
    # Assign class indices: use matched class if found, otherwise ignore_index
    seg = torch.where(
        has_match,
        class_indices[match_indices],
        torch.full((H * W,), ignore_index, dtype=torch.long, device=device)
    )
    
    return seg.reshape(H, W)


def room_id_to_class_index(room_id: torch.Tensor, ignore_index: int = 0) -> torch.Tensor:
    """
    Convert room_id to binary class index: room=0, scene=1
    Scene layouts have room_id "0000" or 0 → class 1 (scene)
    All other room_ids → class 0 (room)
    
    Args:
        room_id: Room ID tensor, shape () or (N,). Can be int or string "0000"
        ignore_index: Value to use for unknown room IDs (defaults to 0 for room)
    
    Returns:
        Class index tensor, same shape as input. 0=room, 1=scene
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
        if rid_int in ROOM_ID_TO_CLASS:
            return ROOM_ID_TO_CLASS[rid_int]
        # Default: any other room_id is a room (class 0)
        return 0
    
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

