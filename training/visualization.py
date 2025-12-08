#!/usr/bin/env python3
"""
Visualization utilities for diffusion model training.

Provides image grid creation, comparison grids, and latent-to-RGB conversion.
"""

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


def create_image_grid(images, grid_cols=4, border_width=2, border_color=(128, 128, 128),
                      padding=0, background_color=(255, 255, 255)):
    """
    Create an image grid with borders around each image.

    Args:
        images: List of PIL Images
        grid_cols: Number of columns in grid
        border_width: Width of border around each image (pixels)
        border_color: RGB tuple for border color (default: gray)
        padding: Extra padding between images (pixels)
        background_color: RGB tuple for background color

    Returns:
        PIL Image of the grid
    """
    if not images:
        return None

    img_size = images[0].size[0]
    num_images = len(images)
    grid_rows = (num_images + grid_cols - 1) // grid_cols

    # Calculate cell size (image + border on all sides)
    cell_size = img_size + 2 * border_width + padding

    # Create grid
    grid_width = cell_size * grid_cols - padding  # No padding after last column
    grid_height = cell_size * grid_rows - padding  # No padding after last row
    grid = Image.new('RGB', (grid_width, grid_height), background_color)

    for idx, img in enumerate(images):
        row = idx // grid_cols
        col = idx % grid_cols

        # Calculate position
        x = col * cell_size
        y = row * cell_size

        # Draw border (by filling a rectangle and pasting image on top)
        if border_width > 0:
            draw = ImageDraw.Draw(grid)
            draw.rectangle(
                [x, y, x + img_size + 2 * border_width - 1, y + img_size + 2 * border_width - 1],
                fill=border_color
            )

        # Paste image inside border
        grid.paste(img, (x + border_width, y + border_width))

    return grid


def create_comparison_grid(target_images, generated_images, grid_cols=4,
                           border_width=2, border_color=(128, 128, 128),
                           label_height=30, add_labels=True):
    """
    Create a side-by-side comparison grid with targets on left, generated on right.

    Args:
        target_images: List of PIL target images
        generated_images: List of PIL generated images
        grid_cols: Number of columns per side
        border_width: Border width around each image
        border_color: RGB tuple for border color
        label_height: Height of label area at top
        add_labels: Whether to add "Target" and "Generated" labels

    Returns:
        PIL Image of the comparison
    """
    if not target_images or not generated_images:
        return None

    # Create individual grids
    target_grid = create_image_grid(
        target_images[:len(generated_images)],
        grid_cols=grid_cols,
        border_width=border_width,
        border_color=border_color
    )
    generated_grid = create_image_grid(
        generated_images[:len(target_images)],
        grid_cols=grid_cols,
        border_width=border_width,
        border_color=border_color
    )

    if target_grid is None or generated_grid is None:
        return None

    # Add separator between grids
    separator_width = 4

    # Calculate total dimensions
    total_width = target_grid.width + separator_width + generated_grid.width
    total_height = target_grid.height + (label_height if add_labels else 0)

    # Create combined image
    comparison = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    y_offset = label_height if add_labels else 0

    # Paste grids
    comparison.paste(target_grid, (0, y_offset))
    comparison.paste(generated_grid, (target_grid.width + separator_width, y_offset))

    # Draw separator line
    draw = ImageDraw.Draw(comparison)
    separator_x = target_grid.width + separator_width // 2
    draw.line(
        [(separator_x, y_offset), (separator_x, total_height)],
        fill=(64, 64, 64),
        width=separator_width
    )

    # Add labels
    if add_labels:
        font = _get_font()

        target_label = "Target"
        generated_label = "Generated"

        # Center labels above each grid
        if font:
            target_bbox = draw.textbbox((0, 0), target_label, font=font)
            generated_bbox = draw.textbbox((0, 0), generated_label, font=font)
            target_text_width = target_bbox[2] - target_bbox[0]
            generated_text_width = generated_bbox[2] - generated_bbox[0]
        else:
            target_text_width = len(target_label) * 8
            generated_text_width = len(generated_label) * 8

        target_x = (target_grid.width - target_text_width) // 2
        generated_x = target_grid.width + separator_width + (generated_grid.width - generated_text_width) // 2

        draw.text((target_x, 5), target_label, fill=(0, 0, 0), font=font)
        draw.text((generated_x, 5), generated_label, fill=(0, 0, 0), font=font)

    return comparison


def _get_font(size=20):
    """Get a font for labels, with fallbacks."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, size)
        except:
            pass
    try:
        return ImageFont.load_default()
    except:
        return None


def latents2rgb(model, output_dict, warning_prefix="Decoder"):
    """
    Convert latents to RGB images from model output.

    Handles two cases:
    1. Output already contains "rgb" key - use it directly
    2. Output contains "latent" key - decode using model.decoder

    Args:
        model: DiffusionModel with decoder
        output_dict: Dictionary containing either "rgb" or "latent" key
        warning_prefix: Prefix for warning message if decoding fails

    Returns:
        torch.Tensor: RGB images in [0, 1] range, or None if decoding fails
    """
    if "rgb" in output_dict:
        rgb = output_dict["rgb"]
        # Normalize from [-1, 1] to [0, 1] if needed
        if rgb.min() < 0:
            rgb = (rgb + 1.0) / 2.0
        rgb = torch.clamp(rgb, 0.0, 1.0)
        return rgb
    elif "latent" in output_dict:
        with torch.no_grad():
            # Clamp latents before decoding (same as during sampling)
            latents = output_dict["latent"]
            if hasattr(model, '_latent_clamp_min') and hasattr(model, '_latent_clamp_max'):
                clamp_min = model._latent_clamp_min
                clamp_max = model._latent_clamp_max
                latents = torch.clamp(latents, clamp_min, clamp_max)

            decoded = model.decoder({"latent": latents})
            if "rgb" in decoded:
                rgb = (decoded["rgb"] + 1.0) / 2.0
                rgb = torch.clamp(rgb, 0.0, 1.0)
                return rgb
            else:
                print(f"  Warning: {warning_prefix} did not produce RGB output")
                return None
    else:
        print(f"  Warning: {warning_prefix} output missing both 'rgb' and 'latent' keys")
        return None


def tensor_to_pil_images(tensor, normalize=True):
    """
    Convert a batch of tensors to PIL images.

    Args:
        tensor: Tensor of shape [B, C, H, W] with values in [0, 1] or [-1, 1]
        normalize: Whether to normalize from [-1, 1] to [0, 1]

    Returns:
        List of PIL Images
    """
    import numpy as np

    if normalize and tensor.min() < 0:
        tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0.0, 1.0)

    np_images = (tensor.cpu().numpy() * 255.0).astype(np.uint8)
    images = []
    for i in range(np_images.shape[0]):
        img = Image.fromarray(np_images[i].transpose(1, 2, 0))
        images.append(img)
    return images
