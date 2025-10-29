from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
import torch
from PIL import Image

# This transform will be used to convert layout images to tensors
layout_transform = transforms.ToTensor()

def _is_empty_batch(batch):
    """Check if batch is empty or None."""
    return not batch or len(batch) == 0

def collate_skip_none(batch):
    """Collate function that handles empty batches and skips None values."""
    if _is_empty_batch(batch):
        return None
    
    # Filter out None values from batch
    filtered_batch = [item for item in batch if item is not None]
    if not filtered_batch:
        return None
        
    return default_collate(filtered_batch)

def collate_fn(batch):
    # Check if batch is empty
    if _is_empty_batch(batch):
        return {}

    # Separate items
    layout_items = []
    pov_items = []
    graph_items = []

    for sample in batch:
        if "layout" in sample:
            layout_items.append(sample["layout"])
        if "pov" in sample:
            pov_items.append(sample["pov"])
        if "graph" in sample:
            graph_items.append(sample["graph"])

    collated_batch = {}

    # Handle "layout" items
    if layout_items:
        # Check the type of the first item
        if isinstance(layout_items[0], torch.Tensor):
            # All are tensors, stack them
            collated_batch["layout"] = torch.stack(layout_items)
        elif isinstance(layout_items[0], Image.Image):
            # They are images, transform and stack
            tensor_layouts = [layout_transform(img) for img in layout_items]
            collated_batch["layout"] = torch.stack(tensor_layouts)
        else:
            # Handle unexpected type
            raise TypeError(f"Unexpected type for layout: {type(layout_items[0])}")
    
    # Return pov and graph items as lists
    if pov_items:
        collated_batch["pov"] = pov_items
    
    if graph_items:
        collated_batch["graph"] = graph_items

    return collated_batch
