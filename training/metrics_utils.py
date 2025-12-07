"""
Metrics utilities for evaluating diffusion models.
Includes FID, KID, LPIPS, and CLIP score calculation.

Note on disk space:
- Temporary image directories are automatically cleaned up after each metric computation
- clean-fid may cache Inception features in ~/.cache/clean-fid/ which can grow over time
- If disk space is an issue, consider clearing the clean-fid cache periodically:
  rm -rf ~/.cache/clean-fid/
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from PIL import Image
from typing import List, Optional, Union, Dict
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

try:
    import timm
    CLIP_AVAILABLE = True
    CLIP_USE_TIMM = True
except ImportError:
    try:
        import clip
        CLIP_AVAILABLE = True
        CLIP_USE_TIMM = False
    except ImportError:
        CLIP_AVAILABLE = False
        CLIP_USE_TIMM = False
        warnings.warn("CLIP not available. Install with: pip install timm (recommended) or pip install git+https://github.com/openai/CLIP.git")


def calculate_fid(
    real_images: List[Image.Image],
    generated_images: List[Image.Image],
    device: Optional[torch.device] = None,
    batch_size: int = 50,
) -> float:
    """
    Calculate FID (Fréchet Inception Distance) between real and generated images.
    
    Args:
        real_images: List of PIL Images (real/target images)
        generated_images: List of PIL Images (generated images)
        device: Device to use for computation (default: cuda if available)
        batch_size: Batch size for processing images
    
    Returns:
        FID score (lower is better)
    """
    if not CLEANFID_AVAILABLE:
        raise ImportError(
            "cleanfid is required for FID calculation. "
            "Install with: pip install clean-fid"
        )
    
    if len(real_images) == 0 or len(generated_images) == 0:
        raise ValueError("Both real_images and generated_images must be non-empty")
    
    # Use minimum length to ensure equal comparison
    min_len = min(len(real_images), len(generated_images))
    real_images = real_images[:min_len]
    generated_images = generated_images[:min_len]
    
    # Create temporary directories for clean-fid
    import tempfile
    import shutil
    
    temp_dir = None
    try:
        # Use a custom temp directory that we can explicitly control
        temp_dir = tempfile.mkdtemp(prefix="fid_temp_")
        temp_path = Path(temp_dir)
        real_dir = temp_path / "real"
        gen_dir = temp_path / "generated"
        real_dir.mkdir()
        gen_dir.mkdir()
        
        # Save images to temporary directories
        for i, img in enumerate(real_images):
            # Ensure image is RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(real_dir / f"img_{i:05d}.png")
        
        for i, img in enumerate(generated_images):
            # Ensure image is RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(gen_dir / f"img_{i:05d}.png")
        
        # Calculate FID using clean-fid
        try:
            fid_score = fid.compute_fid(
                str(real_dir),
                str(gen_dir),
                mode="clean",
                batch_size=batch_size,
                device=device,
            )
        except Exception as e:
            # Fallback: try without device specification
            try:
                fid_score = fid.compute_fid(
                    str(real_dir),
                    str(gen_dir),
                    mode="clean",
                    batch_size=batch_size,
                )
            except Exception as e2:
                raise RuntimeError(f"Failed to compute FID: {e2}") from e2
        
        return float(fid_score)
    finally:
        # Explicitly clean up temporary directory
        if temp_dir and Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                # Log but don't fail if cleanup fails
                import warnings
                warnings.warn(f"Failed to clean up temporary FID directory {temp_dir}: {e}")


def compute_fid_from_tensors(
    real_tensors: torch.Tensor,
    generated_tensors: torch.Tensor,
    device: Optional[torch.device] = None,
    batch_size: int = 50,
) -> float:
    """
    Calculate FID from tensors (assumed to be in [0, 1] range, CHW format).
    
    Args:
        real_tensors: Tensor of shape [N, C, H, W] with values in [0, 1]
        generated_tensors: Tensor of shape [M, C, H, W] with values in [0, 1]
        device: Device to use
        batch_size: Batch size for processing
    
    Returns:
        FID score
    """
    # Convert tensors to PIL Images
    real_images = []
    generated_images = []
    
    # Convert real tensors
    real_np = (real_tensors.cpu().numpy() * 255.0).astype(np.uint8)
    for i in range(real_np.shape[0]):
        # Convert from CHW to HWC
        img_array = real_np[i].transpose(1, 2, 0)
        img = Image.fromarray(img_array)
        real_images.append(img)
    
    # Convert generated tensors
    gen_np = (generated_tensors.cpu().numpy() * 255.0).astype(np.uint8)
    for i in range(gen_np.shape[0]):
        # Convert from CHW to HWC
        img_array = gen_np[i].transpose(1, 2, 0)
        img = Image.fromarray(img_array)
        generated_images.append(img)
    
    return calculate_fid(real_images, generated_images, device=device, batch_size=batch_size)


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
    
    # Create temporary directories for clean-fid
    import tempfile
    import shutil
    
    temp_dir = None
    try:
        # Use a custom temp directory that we can explicitly control
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
        # Explicitly clean up temporary directory
        if temp_dir and Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                # Log but don't fail if cleanup fails
                import warnings
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
    # Convert tensors to PIL Images
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
    net: str = 'alex',  # 'alex', 'vgg', or 'squeeze'
) -> float:
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity) between real and generated images.
    Measures perceptual distance - good for paired data.
    
    Args:
        real_tensors: Tensor of shape [N, C, H, W] with values in [0, 1]
        generated_tensors: Tensor of shape [N, C, H, W] with values in [0, 1]
        device: Device to use
        net: Network to use ('alex', 'vgg', or 'squeeze')
    
    Returns:
        Average LPIPS score (lower is better, 0 = identical)
    """
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


def compute_clip_score(
    images: torch.Tensor,
    text_embeddings: torch.Tensor,
    device: Optional[torch.device] = None,
) -> float:
    """
    Compute similarity score between images and text/graph embeddings.
    Uses CLIP's image encoder and compares with provided text embeddings.
    Note: This uses graph embeddings (from sentence-transformers), not CLIP's text encoder.
    Still provides a useful semantic similarity metric.
    
    Args:
        images: Tensor of shape [N, C, H, W] with values in [0, 1]
        text_embeddings: Tensor of shape [N, D] - text/graph embeddings (e.g., from all-MiniLM-L6-v2)
        device: Device to use
    
    Returns:
        Average similarity score (higher is better, range typically [-1, 1])
    """
    if not CLIP_AVAILABLE:
        raise ImportError(
            "CLIP is required for CLIP score calculation. "
            "Install with: pip install timm (recommended) or pip install git+https://github.com/openai/CLIP.git"
        )
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure same number of samples
    min_len = min(images.shape[0], text_embeddings.shape[0])
    images = images[:min_len].to(device)
    text_embeddings = text_embeddings[:min_len].to(device)
    
    # Resize images to 224x224 for CLIP
    images_resized = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
    
    # Get image features from CLIP
    with torch.no_grad():
        if CLIP_USE_TIMM:
            # Use timm's CLIP model (works with modern PyTorch)
            clip_model = timm.create_model('vit_base_patch32_clip_224', pretrained=True, num_classes=0)
            clip_model = clip_model.to(device).eval()
            
            # timm CLIP expects images in [0, 1] range
            # Normalize using ImageNet stats (timm handles this internally, but we need to ensure [0, 1])
            image_features = clip_model.forward_features(images_resized)
            # Extract CLS token or global pool
            if isinstance(image_features, dict):
                image_features = image_features.get('cls_token', image_features.get('pooler_output', list(image_features.values())[0]))
            # If it's a tuple, take the first element
            if isinstance(image_features, tuple):
                image_features = image_features[0]
            # Global average pool if needed
            if image_features.dim() == 4:  # [B, C, H, W]
                image_features = F.adaptive_avg_pool2d(image_features, 1).flatten(1)
        else:
            # Use original OpenAI CLIP (if available)
            import clip
            clip_model, _ = clip.load("ViT-B/32", device=device)
            clip_model.eval()
            
            # CLIP expects images in [-1, 1] range
            images_normalized = images_resized * 2.0 - 1.0
            image_features = clip_model.encode_image(images_normalized)
        
        image_features = F.normalize(image_features, dim=1)
    
    # Normalize text embeddings
    text_features = F.normalize(text_embeddings, dim=1)
    
    # Compute cosine similarity (CLIP score)
    similarities = (image_features * text_features).sum(dim=1)
    
    return float(similarities.mean().item())

