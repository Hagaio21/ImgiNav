"""
Metrics utilities for evaluating diffusion models.
Includes KID and LPIPS calculation.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from PIL import Image
from typing import List, Optional
import warnings

try:
    from cleanfid import fid
    CLEANFID_AVAILABLE = True
except ImportError:
    CLEANFID_AVAILABLE = False
    warnings.warn("cleanfid not available. Install with: pip install clean-fid")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    warnings.warn("lpips not available. Install with: pip install lpips")


def calculate_kid(
    real_images: List[Image.Image],
    generated_images: List[Image.Image],
    device: Optional[torch.device] = None,
    batch_size: int = 50,
) -> float:
    """
    Calculate KID (Kernel Inception Distance) between real and generated images.
    More reliable than FID at low sample counts (even 500 samples can work).
    
    Args:
        real_images: List of PIL Images (real/target images)
        generated_images: List of PIL Images (generated images)
        device: Device to use for computation
        batch_size: Batch size for processing images
    
    Returns:
        KID score (lower is better)
    """
    if not CLEANFID_AVAILABLE:
        raise ImportError(
            "cleanfid is required for KID calculation. "
            "Install with: pip install clean-fid"
        )
    
    if len(real_images) == 0 or len(generated_images) == 0:
        raise ValueError("Both real_images and generated_images must be non-empty")
    
    # Use minimum length to ensure equal comparison
    min_len = min(len(real_images), len(generated_images))
    real_images = real_images[:min_len]
    generated_images = generated_images[:min_len]
    
    import tempfile
    import shutil
    
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="kid_temp_")
        temp_path = Path(temp_dir)
        real_dir = temp_path / "real"
        gen_dir = temp_path / "generated"
        real_dir.mkdir()
        gen_dir.mkdir()
        
        # Save images to temporary directories
        for i, img in enumerate(real_images):
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(real_dir / f"img_{i:05d}.png")
        
        for i, img in enumerate(generated_images):
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(gen_dir / f"img_{i:05d}.png")
        
        # Calculate KID using clean-fid
        try:
            kid_score = fid.compute_kid(
                str(real_dir),
                str(gen_dir),
                mode="clean",
                batch_size=batch_size,
                device=device,
            )
        except Exception as e:
            try:
                kid_score = fid.compute_kid(
                    str(real_dir),
                    str(gen_dir),
                    mode="clean",
                    batch_size=batch_size,
                )
            except Exception as e2:
                raise RuntimeError(f"Failed to compute KID: {e2}") from e2
        
        return float(kid_score)
    finally:
        if temp_dir and Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                warnings.warn(f"Failed to clean up temporary KID directory {temp_dir}: {e}")


def compute_kid_from_tensors(
    real_tensors: torch.Tensor,
    generated_tensors: torch.Tensor,
    device: Optional[torch.device] = None,
    batch_size: int = 50,
) -> float:
    """
    Calculate KID from tensors (assumed to be in [0, 1] range, CHW format).
    
    Args:
        real_tensors: Tensor of shape [N, C, H, W] with values in [0, 1]
        generated_tensors: Tensor of shape [M, C, H, W] with values in [0, 1]
        device: Device to use
        batch_size: Batch size for processing
    
    Returns:
        KID score
    """
    real_images = []
    generated_images = []
    
    real_np = (real_tensors.cpu().numpy() * 255.0).astype(np.uint8)
    for i in range(real_np.shape[0]):
        img_array = real_np[i].transpose(1, 2, 0)
        img = Image.fromarray(img_array)
        real_images.append(img)
    
    gen_np = (generated_tensors.cpu().numpy() * 255.0).astype(np.uint8)
    for i in range(gen_np.shape[0]):
        img_array = gen_np[i].transpose(1, 2, 0)
        img = Image.fromarray(img_array)
        generated_images.append(img)
    
    return calculate_kid(real_images, generated_images, device=device, batch_size=batch_size)


def compute_lpips(
    real_tensors: torch.Tensor,
    generated_tensors: torch.Tensor,
    device: Optional[torch.device] = None,
    net: str = 'alex',
) -> float:

    if not LPIPS_AVAILABLE:
        raise ImportError(
            "lpips is required for LPIPS calculation. "
            "Install with: pip install lpips"
        )
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure same number of samples
    min_len = min(real_tensors.shape[0], generated_tensors.shape[0])
    real_tensors = real_tensors[:min_len].to(device)
    generated_tensors = generated_tensors[:min_len].to(device)
    
    # Normalize from [0, 1] to [-1, 1] for LPIPS
    real_tensors = real_tensors * 2.0 - 1.0
    generated_tensors = generated_tensors * 2.0 - 1.0
    
    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net=net).to(device)
    lpips_model.eval()
    
    # Compute LPIPS for each pair
    with torch.no_grad():
        distances = []
        for i in range(min_len):
            dist = lpips_model(real_tensors[i:i+1], generated_tensors[i:i+1])
            distances.append(dist.item())
    
    return float(np.mean(distances))