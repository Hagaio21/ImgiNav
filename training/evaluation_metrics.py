"""
Evaluation metrics for diffusion model training:
- CLIP Score: Semantic alignment between generated layouts and text/POV inputs
- FID: Fréchet Inception Distance for distribution quality
- mIoU: Mean Intersection over Union for spatial correctness
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

# Set cache directories to use work space instead of home directory
# This avoids "No space left on device" errors in /zhome
work_cache_dir = "/work3/s233249/ImgiNav/.cache"
os.makedirs(work_cache_dir, exist_ok=True)

# Shared directory for evaluation models (used by all experiments)
SHARED_EVAL_MODELS_DIR = os.path.join(work_cache_dir, "eval_models")
os.makedirs(SHARED_EVAL_MODELS_DIR, exist_ok=True)

# Set HuggingFace cache directory
hf_cache = os.path.join(work_cache_dir, "huggingface")
os.makedirs(hf_cache, exist_ok=True)
os.makedirs(os.path.join(hf_cache, "datasets"), exist_ok=True)

os.environ["HF_HOME"] = work_cache_dir
os.environ["TRANSFORMERS_CACHE"] = hf_cache
os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_cache, "datasets")

# Set PyTorch hub cache directory
torch_cache = os.path.join(work_cache_dir, "torch")
os.makedirs(torch_cache, exist_ok=True)
os.makedirs(os.path.join(torch_cache, "hub"), exist_ok=True)
os.environ["TORCH_HOME"] = torch_cache

try:
    from transformers import CLIPModel, CLIPProcessor
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    warnings.warn("transformers library not available. CLIP Score will be disabled.")

try:
    from scipy import linalg
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. FID computation will be disabled.")

try:
    from torchvision.models import inception_v3
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    # Try to import weights enum (available in newer torchvision versions)
    try:
        from torchvision.models import Inception_V3_Weights
    except ImportError:
        Inception_V3_Weights = None
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    warnings.warn("torchvision not available. FID computation will be disabled.")
    Inception_V3_Weights = None


# Global CLIP model cache
_clip_model = None
_clip_processor = None


def get_clip_model(device="cuda", model_dir=None):
    """Get or load CLIP model for CLIP Score computation.
    
    Args:
        device: Device to load model on
        model_dir: Deprecated - models are now saved in shared location
    
    Returns:
        (clip_model, clip_processor) or (None, None) if unavailable
    """
    global _clip_model, _clip_processor
    
    if not CLIP_AVAILABLE:
        return None, None
    
    if _clip_model is None:
        try:
            model_name = "openai/clip-vit-base-patch32"
            clip_dir = Path(SHARED_EVAL_MODELS_DIR) / "clip"
            clip_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if model exists in shared location
            if (clip_dir / "config.json").exists():
                try:
                    _clip_model = CLIPModel.from_pretrained(str(clip_dir))
                    _clip_processor = CLIPProcessor.from_pretrained(str(clip_dir))
                    print(f"  Loaded CLIP model from shared cache: {clip_dir}")
                except Exception as e:
                    warnings.warn(f"Failed to load CLIP from {clip_dir}, downloading: {e}")
                    clip_dir = None  # Fall back to downloading
            
            # Download if not found locally
            if _clip_model is None:
                _clip_model = CLIPModel.from_pretrained(model_name, cache_dir=None)
                _clip_processor = CLIPProcessor.from_pretrained(model_name, cache_dir=None)
                
                # Save to shared location
                clip_dir = Path(SHARED_EVAL_MODELS_DIR) / "clip"
                _clip_model.save_pretrained(str(clip_dir))
                _clip_processor.save_pretrained(str(clip_dir))
                print(f"  Saved CLIP model to shared cache: {clip_dir} (~150MB)")
            
            _clip_model = _clip_model.to(device)
            _clip_model.eval()
        except Exception as e:
            warnings.warn(f"Failed to load CLIP model: {e}")
            return None, None
    
    return _clip_model, _clip_processor


def compute_clip_score(
    images: torch.Tensor,
    text_emb: Optional[torch.Tensor] = None,
    pov_emb: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    use_text_prompt: bool = False
) -> Dict[str, float]:
    """
    Compute CLIP Score: semantic alignment between images and text/POV embeddings.
    
    Args:
        images: Generated images [B, C, H, W] in range [0, 1] or [0, 255]
        text_emb: Text embeddings [B, D] or None
        pov_emb: POV embeddings [B, D] or None
        device: Device to run on
        use_text_prompt: If True, uses a generic prompt instead of embeddings
    
    Returns:
        Dictionary with 'clip_score' (average cosine similarity)
    """
    if not CLIP_AVAILABLE:
        return {}
    
    if device is None:
        device = images.device if isinstance(images, torch.Tensor) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    clip_model, clip_processor = get_clip_model(device)
    if clip_model is None:
        return {}
    
    # Convert images to PIL format for CLIP processor
    # Images should be [B, C, H, W] in range [0, 1] or [0, 255]
    B = images.shape[0]
    
    # Normalize to [0, 1] if needed
    if images.max() > 1.1:
        images = images / 255.0
    
    # Convert to PIL Images
    pil_images = []
    for i in range(B):
        img_tensor = images[i].cpu()
        # Convert from [C, H, W] to [H, W, C] and to numpy
        if img_tensor.shape[0] == 3:
            img_np = img_tensor.permute(1, 2, 0).numpy()
        else:
            img_np = img_tensor.numpy()
        img_np = (img_np * 255).astype(np.uint8)
        pil_images.append(Image.fromarray(img_np))
    
    # Process images through CLIP
    with torch.no_grad():
        inputs = clip_processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        image_features = clip_model.get_image_features(**inputs)
        image_features = F.normalize(image_features, p=2, dim=1)
    
    # For text, we can either:
    # 1. Use a generic prompt (if use_text_prompt=True)
    # 2. Try to reconstruct text from embeddings (not straightforward)
    # 3. Use embeddings directly if CLIP has a way to do that
    
    # For now, use a generic prompt as placeholder
    # In practice, you'd want to use actual text prompts or find a way to map embeddings to text
    if use_text_prompt:
        text_prompts = ["a room layout"] * B
        text_inputs = clip_processor(text=text_prompts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_features = clip_model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, p=2, dim=1)
        
        # Compute cosine similarity
        clip_scores = (image_features * text_features).sum(dim=1)  # [B]
        avg_score = clip_scores.mean().item()
        
        return {"clip_score": avg_score}
    
    # If we have embeddings but no text, we can't compute CLIP score directly
    # Return empty dict to indicate metric unavailable
    return {}


# Global Inception model cache
_inception_model = None
_inception_transform = None


def get_inception_model(device="cuda", model_dir=None):
    """Get or load Inception v3 model for FID computation.
    
    Args:
        device: Device to load model on
        model_dir: Deprecated - models are now saved in shared location
    
    Returns:
        (inception_model, inception_transform) or (None, None) if unavailable
    """
    global _inception_model, _inception_transform
    
    if not TORCHVISION_AVAILABLE:
        return None, None
    
    if _inception_model is None:
        try:
            inception_dir = Path(SHARED_EVAL_MODELS_DIR) / "inception"
            inception_dir.mkdir(parents=True, exist_ok=True)
            inception_path = inception_dir / "inception_v3.pth"
            
            # Check if model exists in shared location
            if inception_path.exists():
                try:
                    _inception_model = inception_v3(pretrained=False, transform_input=False)
                    # Remove fc layer before loading to match saved state
                    _inception_model.fc = torch.nn.Identity()
                    # Load state dict with strict=False to ignore fc layer if present
                    saved_state = torch.load(inception_path, map_location=device)
                    # Filter out fc layer keys if they exist
                    filtered_state = {k: v for k, v in saved_state.items() if not k.startswith('fc.')}
                    _inception_model.load_state_dict(filtered_state, strict=False)
                    print(f"  Loaded Inception model from shared cache: {inception_path}")
                except Exception as e:
                    warnings.warn(f"Failed to load Inception from {inception_path}, downloading: {e}")
                    _inception_model = None  # Will fall back to downloading
            
            # Download if not found locally or loading failed
            if _inception_model is None:
                # Use newer weights API if available, fall back to pretrained for older versions
                try:
                    _inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
                except (AttributeError, TypeError):
                    # Fall back to deprecated pretrained parameter for older torchvision
                    _inception_model = inception_v3(pretrained=True, transform_input=False)
                _inception_model.fc = torch.nn.Identity()  # Remove final classification layer
                
                # Save to shared location (without fc layer)
                torch.save(_inception_model.state_dict(), inception_path)
                print(f"  Saved Inception model to shared cache: {inception_path} (~100MB)")
            
            _inception_model = _inception_model.to(device)
            _inception_model.eval()
            
            _inception_transform = Compose([
                Resize(299),
                CenterCrop(299),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except Exception as e:
            warnings.warn(f"Failed to load Inception model: {e}")
            return None, None
    
    return _inception_model, _inception_transform


def compute_fid_features(images: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Extract Inception features for FID computation.
    
    Args:
        images: Images [B, C, H, W] in range [0, 1]
        device: Device to run on
    
    Returns:
        Features [B, 2048]
    """
    if not TORCHVISION_AVAILABLE or not SCIPY_AVAILABLE:
        return None
    
    if device is None:
        device = images.device if isinstance(images, torch.Tensor) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inception_model, transform = get_inception_model(device)
    if inception_model is None:
        return None
    
    B = images.shape[0]
    
    # Normalize to [0, 1] if needed
    if images.max() > 1.1:
        images = images / 255.0
    
    # Process each image through Inception
    features_list = []
    with torch.no_grad():
        for i in range(B):
            img_tensor = images[i].cpu()
            # Convert from [C, H, W] to PIL Image
            if img_tensor.shape[0] == 3:
                img_np = img_tensor.permute(1, 2, 0).numpy()
            else:
                img_np = img_tensor.numpy()
            img_np = (img_np * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            # Transform and extract features
            img_transformed = transform(pil_img).unsqueeze(0).to(device)
            feat = inception_model(img_transformed)
            features_list.append(feat.cpu())
    
    if len(features_list) == 0:
        return None
    
    return torch.cat(features_list, dim=0)


def compute_fid(
    real_features: torch.Tensor,
    fake_features: torch.Tensor
) -> float:
    """
    Compute Fréchet Inception Distance (FID).
    
    Args:
        real_features: Features from real images [N, 2048]
        fake_features: Features from generated images [M, 2048]
    
    Returns:
        FID score (lower is better)
    """
    if not SCIPY_AVAILABLE:
        return float('inf')
    
    try:
        # Validate inputs
        if real_features.shape[0] < 2 or fake_features.shape[0] < 2:
            warnings.warn(f"FID requires at least 2 samples per set. Got {real_features.shape[0]} real and {fake_features.shape[0]} fake samples.")
            return float('inf')
        
        # Check for NaN or Inf values
        if not torch.isfinite(real_features).all() or not torch.isfinite(fake_features).all():
            warnings.warn("FID features contain NaN or Inf values. Skipping FID computation.")
            return float('inf')
        
        # Convert to numpy
        real_np = real_features.numpy()
        fake_np = fake_features.numpy()
        
        # Compute mean and covariance
        mu1 = real_np.mean(axis=0)
        mu2 = fake_np.mean(axis=0)
        
        # Compute covariance with regularization to avoid singular matrices
        # Add small epsilon to diagonal for numerical stability
        eps = 1e-6
        sigma1 = np.cov(real_np, rowvar=False)
        sigma2 = np.cov(fake_np, rowvar=False)
        
        # Regularize covariance matrices
        sigma1 += np.eye(sigma1.shape[0]) * eps
        sigma2 += np.eye(sigma2.shape[0]) * eps
        
        # Compute FID
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        
        # Take real part (matrix square root can produce complex values)
        covmean = np.real(covmean)
        
        # Check if covmean is valid
        if not np.isfinite(covmean).all():
            # Try with larger regularization
            eps = 1e-4
            sigma1_reg = sigma1 + np.eye(sigma1.shape[0]) * eps
            sigma2_reg = sigma2 + np.eye(sigma2.shape[0]) * eps
            covmean, _ = linalg.sqrtm(sigma1_reg @ sigma2_reg, disp=False)
            covmean = np.real(covmean)
        
        # Final validation
        if not np.isfinite(covmean).all():
            warnings.warn("FID computation produced invalid covmean. Returning inf.")
            return float('inf')
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        # Ensure result is real and finite
        fid = np.real(fid)
        
        # Validate final result is reasonable (FID should typically be < 1000 for similar images)
        if not np.isfinite(fid) or fid < 0 or fid > 1e6:
            warnings.warn(f"FID computation produced unreasonable value: {fid}. This may indicate numerical issues.")
            return float('inf')
        
        return float(fid)
    except Exception as e:
        warnings.warn(f"FID computation failed: {e}")
        return float('inf')


def compute_miou(
    pred_seg: np.ndarray,
    gt_seg: np.ndarray,
    num_classes: Optional[int] = None
) -> float:
    """
    Compute mean Intersection over Union (mIoU) between segmentation maps.
    
    Since layouts are geometric, mIoU provides a much more accurate assessment of 
    spatial correctness than pixel-wise MSE. It measures how well the model preserves 
    object boundaries and spatial relationships by computing IoU for each class and 
    averaging across all classes.
    
    Args:
        pred_seg: Predicted segmentation map [H, W] with class IDs
        gt_seg: Ground truth segmentation map [H, W] with class IDs
        num_classes: Number of classes (if None, inferred from data)
    
    Returns:
        mIoU score (higher is better, range [0, 1])
    """
    if pred_seg.shape != gt_seg.shape:
        # Resize pred to match gt
        from scipy.ndimage import zoom
        zoom_factors = (gt_seg.shape[0] / pred_seg.shape[0], gt_seg.shape[1] / pred_seg.shape[1])
        pred_seg = zoom(pred_seg, zoom_factors, order=0)  # Nearest neighbor
    
    # Get unique classes
    if num_classes is None:
        all_classes = np.unique(np.concatenate([pred_seg.flatten(), gt_seg.flatten()]))
        num_classes = len(all_classes)
        class_map = {cls: idx for idx, cls in enumerate(all_classes)}
    else:
        all_classes = np.arange(num_classes)
        class_map = {cls: idx for idx, cls in enumerate(all_classes)}
    
    # Compute IoU for each class
    ious = []
    for cls in all_classes:
        pred_mask = (pred_seg == cls)
        gt_mask = (gt_seg == cls)
        
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union > 0:
            iou = intersection / union
            ious.append(iou)
    
    # Return mean IoU
    if len(ious) == 0:
        return 0.0
    return float(np.mean(ious))


def compute_evaluation_metrics(
    pred_images: torch.Tensor,
    gt_images: torch.Tensor,
    text_emb: Optional[torch.Tensor] = None,
    pov_emb: Optional[torch.Tensor] = None,
    taxonomy=None,
    device: Optional[torch.device] = None,
    compute_clip: bool = True,
    compute_fid: bool = True,
    compute_miou: bool = True,
    model_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        pred_images: Generated images [B, C, H, W]
        gt_images: Ground truth images [B, C, H, W]
        text_emb: Text embeddings [B, D] or None
        pov_emb: POV embeddings [B, D] or None
        taxonomy: Taxonomy instance for mIoU (required if compute_miou=True)
        device: Device to run on
        compute_clip: Whether to compute CLIP Score
        compute_fid: Whether to compute FID
        compute_miou: Whether to compute mIoU
        model_dir: Deprecated - models are now saved in shared location
    
    Returns:
        Dictionary of metric values
    """
    metrics = {}
    
    # Store references to functions before parameters shadow them
    # Access the functions from globals to avoid parameter shadowing
    compute_fid_func = globals()['compute_fid']
    compute_miou_func = globals()['compute_miou']
    
    if device is None:
        device = pred_images.device if isinstance(pred_images, torch.Tensor) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Validate inputs
    if pred_images.shape != gt_images.shape:
        raise ValueError(f"Image shape mismatch: pred_images {pred_images.shape} vs gt_images {gt_images.shape}")
    
    # Debug: Check if images are actually different (sanity check)
    pixel_diff = torch.abs(pred_images - gt_images).mean().item()
    if pixel_diff < 0.01:
        warnings.warn(
            f"WARNING: pred_images and gt_images are very similar (mean abs diff={pixel_diff:.6f}). "
            f"Metrics may be unreliable. Pred range: [{pred_images.min().item():.3f}, {pred_images.max().item():.3f}], "
            f"GT range: [{gt_images.min().item():.3f}, {gt_images.max().item():.3f}]"
        )
    
    # CLIP Score - Compare generated images to ground truth images (image-to-image similarity)
    # This is more meaningful than comparing to a generic text prompt
    # IMPORTANT: pred_images = GENERATED, gt_images = TARGETS
    # For bad generated images, CLIP score should be LOW (bad similarity)
    if compute_clip:
        try:
            # Compute CLIP features for both pred and gt images
            clip_model, clip_processor = get_clip_model(device)
            if clip_model is not None:
                B = pred_images.shape[0]
                
                # Debug: Print image stats to verify we have the right images
                pred_img_mean = pred_images.mean().item()
                gt_img_mean = gt_images.mean().item()
                print(f"  [DEBUG] CLIP computation:")
                print(f"    pred_images (GENERATED) mean: {pred_img_mean:.3f}")
                print(f"    gt_images (TARGETS) mean: {gt_img_mean:.3f}")
                
                # Convert images to PIL format
                pred_pil = []
                gt_pil = []
                for i in range(B):
                    pred_img = pred_images[i].cpu()
                    gt_img = gt_images[i].cpu()
                    
                    # Normalize to [0, 1] if needed
                    if pred_img.max() > 1.1:
                        pred_img = pred_img / 255.0
                    if gt_img.max() > 1.1:
                        gt_img = gt_img / 255.0
                    
                    # Convert to numpy and PIL
                    if pred_img.shape[0] == 3:
                        pred_np = pred_img.permute(1, 2, 0).numpy()
                        gt_np = gt_img.permute(1, 2, 0).numpy()
                    else:
                        pred_np = pred_img.numpy()
                        gt_np = gt_img.numpy()
                    
                    pred_np = (pred_np * 255).astype(np.uint8)
                    gt_np = (gt_np * 255).astype(np.uint8)
                    pred_pil.append(Image.fromarray(pred_np))
                    gt_pil.append(Image.fromarray(gt_np))
                
                # Get CLIP features for both
                # IMPORTANT: pred_features = features from GENERATED images
                #           gt_features = features from TARGET images
                with torch.no_grad():
                    pred_inputs = clip_processor(images=pred_pil, return_tensors="pt", padding=True)
                    pred_inputs = {k: v.to(device) for k, v in pred_inputs.items()}
                    pred_features = clip_model.get_image_features(**pred_inputs)
                    pred_features = F.normalize(pred_features, p=2, dim=1)
                    
                    gt_inputs = clip_processor(images=gt_pil, return_tensors="pt", padding=True)
                    gt_inputs = {k: v.to(device) for k, v in gt_inputs.items()}
                    gt_features = clip_model.get_image_features(**gt_inputs)
                    gt_features = F.normalize(gt_features, p=2, dim=1)
                
                # Compute cosine similarity between pred (GENERATED) and gt (TARGET) image features
                # For bad generated images, this should be LOW (close to 0)
                # For good generated images, this should be HIGH (close to 1)
                clip_scores = (pred_features * gt_features).sum(dim=1)  # [B]
                avg_clip_score = clip_scores.mean().item()
                metrics["clip_score"] = avg_clip_score
                
                print(f"    CLIP score: {avg_clip_score:.6f} (higher is better, should be LOW for bad images)")
        except Exception as e:
            warnings.warn(f"CLIP score computation failed: {e}", exc_info=True)
    
    # FID (requires accumulating features across batches - this is a per-batch approximation)
    if compute_fid:
        try:
            B = pred_images.shape[0]
            
            # FID requires sufficient samples for reliable statistics
            # With very small batches, FID can be misleadingly low (appear "good" when it shouldn't)
            # FID typically needs 1000+ samples for reliable estimates
            if B < 16:
                warnings.warn(
                    f"FID computation skipped: batch size ({B}) is too small for reliable FID. "
                    f"FID requires at least 16 samples per set (ideally 1000+). "
                    f"With <16 samples, FID can be misleadingly low. "
                    f"Consider accumulating features across all validation batches for accurate FID."
                )
            else:
                pred_features = compute_fid_features(pred_images, device)
                gt_features = compute_fid_features(gt_images, device)
                if pred_features is not None and gt_features is not None:
                    # Debug: Check feature statistics
                    pred_feat_mean = pred_features.mean(dim=0).mean().item()
                    pred_feat_std = pred_features.std().item()
                    gt_feat_mean = gt_features.mean(dim=0).mean().item()
                    gt_feat_std = gt_features.std().item()
                    
                    # Compute feature distance as sanity check
                    feat_diff = torch.norm(pred_features.mean(dim=0) - gt_features.mean(dim=0)).item()
                    
                    # Note: This is a per-batch FID, not the full dataset FID
                    # For accurate FID, you need to accumulate features across all validation batches
                    # With small batches, FID can be unreliable (may be misleadingly low)
                    # IMPORTANT: gt_features = real (ground truth), pred_features = fake (generated)
                    fid_score = compute_fid_func(gt_features, pred_features)
                    if np.isfinite(fid_score):
                        metrics["fid"] = fid_score
                        
                        # Debug output
                        print(f"  [DEBUG] FID computation:")
                        print(f"    GT features: mean={gt_feat_mean:.3f}, std={gt_feat_std:.3f}")
                        print(f"    Pred features: mean={pred_feat_mean:.3f}, std={pred_feat_std:.3f}")
                        print(f"    Feature mean distance: {feat_diff:.3f}")
                        print(f"    FID score: {fid_score:.2f}")
                        
                        # Warn if FID seems suspiciously low for bad images
                        # For noise vs real images, FID should typically be > 50-100
                        if fid_score < 20.0:
                            warnings.warn(
                                f"FID score ({fid_score:.2f}) is suspiciously low. "
                                f"For bad/generated images vs real images, FID should typically be > 50-100. "
                                f"This may indicate: (1) small batch size ({B}) causing unreliable estimates, "
                                f"(2) features are too similar (mean distance={feat_diff:.3f}), "
                                f"or (3) images are being compared incorrectly."
                            )
                    else:
                        warnings.warn(f"FID score is not finite: {fid_score}")
                else:
                    if pred_features is None:
                        warnings.warn("FID computation skipped: pred_features is None (Inception model may not be available)")
                    if gt_features is None:
                        warnings.warn("FID computation skipped: gt_features is None (Inception model may not be available)")
        except Exception as e:
            warnings.warn(f"FID computation failed: {e}", exc_info=True)
    
    # mIoU (mean Intersection over Union) - Critical for geometric/spatial correctness
    # Since layouts are geometric, mIoU between generated and ground truth segmentation maps
    # provides a much more accurate assessment of spatial correctness than pixel-wise MSE.
    # It measures how well the model preserves object boundaries and spatial relationships.
    if compute_miou:
        try:
            # Try to get taxonomy - use provided one or load from default path
            from common.taxonomy import Taxonomy
            taxonomy_obj = taxonomy
            if taxonomy_obj is None:
                # Try to load from default path
                DEFAULT_TAXONOMY_PATH = "/work3/s233249/ImgiNav/ImgiNav/config/taxonomy.json"
                taxonomy_path = DEFAULT_TAXONOMY_PATH
                if not Path(taxonomy_path).exists():
                    # Try relative path as fallback
                    rel_path = Path("config/taxonomy.json")
                    if rel_path.exists():
                        taxonomy_path = str(rel_path)
                    else:
                        raise FileNotFoundError(f"Taxonomy not found at {DEFAULT_TAXONOMY_PATH} or {rel_path}")
                taxonomy_obj = Taxonomy(taxonomy_path)
            elif not isinstance(taxonomy_obj, Taxonomy):
                # If taxonomy is a string/Path, load it
                if isinstance(taxonomy_obj, (str, Path)):
                    taxonomy_path = str(taxonomy_obj)
                    if not Path(taxonomy_path).exists():
                        raise FileNotFoundError(f"Taxonomy path does not exist: {taxonomy_path}")
                    taxonomy_obj = Taxonomy(taxonomy_path)
                else:
                    raise TypeError(f"taxonomy must be a Taxonomy instance, str, Path, or None, got {type(taxonomy_obj)}")
            
            from data_preparation.utils.layout_analysis import LayoutSegmentor
            segmentor = LayoutSegmentor(taxonomy_obj, mode="category")
            
            # Debug: Verify we're comparing the right images
            print(f"  [DEBUG] mIoU computation:")
            print(f"    pred_images (GENERATED) mean: {pred_images.mean().item():.3f}")
            print(f"    gt_images (TARGETS) mean: {gt_images.mean().item():.3f}")
            
            ious = []
            B = pred_images.shape[0]
            for i in range(B):
                # Convert tensors to numpy arrays
                pred_img = pred_images[i].cpu()
                gt_img = gt_images[i].cpu()
                
                # Convert from [C, H, W] to [H, W, C] and to uint8
                if pred_img.shape[0] == 3:
                    pred_np = pred_img.permute(1, 2, 0).numpy()
                    gt_np = gt_img.permute(1, 2, 0).numpy()
                else:
                    pred_np = pred_img.numpy()
                    gt_np = gt_img.numpy()
                
                # Normalize to [0, 255] if needed
                if pred_np.max() <= 1.0:
                    pred_np = (pred_np * 255).astype(np.uint8)
                    gt_np = (gt_np * 255).astype(np.uint8)
                else:
                    pred_np = pred_np.astype(np.uint8)
                    gt_np = gt_np.astype(np.uint8)
                
                # Segment images to category maps
                pred_seg = segmentor.segment(pred_np)
                gt_seg = segmentor.segment(gt_np)
                
                # Compute mean IoU across all classes
                # IMPORTANT: pred_seg = segmentation of GENERATED images
                #           gt_seg = segmentation of TARGET images
                # For bad generated images, mIoU should be LOW (close to 0)
                # For good generated images, mIoU should be HIGH (close to 1)
                iou = compute_miou_func(pred_seg, gt_seg)
                ious.append(iou)
            
            if len(ious) > 0:
                avg_miou = float(np.mean(ious))
                metrics["miou"] = avg_miou
                print(f"    mIoU: {avg_miou:.6f} (higher is better, should be LOW for bad images)")
            else:
                warnings.warn("mIoU computation skipped: no valid IoU values computed")
        except Exception as e:
            warnings.warn(f"mIoU computation failed: {e}", exc_info=True)
    
    # Layout-specific metrics (coverage, class matching, color matching)
    # These don't require external models - they analyze geometric properties directly
    # Default taxonomy path (always use this path)
    DEFAULT_TAXONOMY_PATH = "/work3/s233249/ImgiNav/ImgiNav/config/taxonomy.json"
    
    try:
        from common.taxonomy import Taxonomy
        from analysis.evaluation_metrics import LayoutEvaluator
        from data_preparation.utils.layout_analysis import LayoutSegmentor
        
        # Always use the default taxonomy path
        taxonomy_path = DEFAULT_TAXONOMY_PATH
        if not Path(taxonomy_path).exists():
            # Try relative path as fallback
            rel_path = Path("config/taxonomy.json")
            if rel_path.exists():
                taxonomy_path = str(rel_path)
                taxonomy_obj = Taxonomy(taxonomy_path)
            else:
                warnings.warn(f"Taxonomy path does not exist: {DEFAULT_TAXONOMY_PATH}, trying to use provided taxonomy object")
                # If taxonomy is already a Taxonomy instance, use it
                if taxonomy is not None and not isinstance(taxonomy, (str, Path)):
                    taxonomy_obj = taxonomy
                else:
                    raise ValueError(f"Taxonomy path does not exist: {DEFAULT_TAXONOMY_PATH} and no valid taxonomy object provided")
        else:
            taxonomy_obj = Taxonomy(taxonomy_path)
        
        # Verify taxonomy_obj is valid
        if taxonomy_obj is None:
            raise ValueError("taxonomy_obj is None after initialization")
        if not isinstance(taxonomy_obj, Taxonomy):
            raise ValueError(f"taxonomy_obj is not a Taxonomy instance, got {type(taxonomy_obj)}")
        
        # Verify methods exist and are callable
        if not hasattr(taxonomy_obj, 'resolve_super'):
            raise ValueError("taxonomy_obj does not have resolve_super method")
        if not callable(getattr(taxonomy_obj, 'resolve_super', None)):
            raise ValueError("taxonomy_obj.resolve_super is not callable")
        
        if not hasattr(taxonomy_obj, 'get_color'):
            raise ValueError("taxonomy_obj does not have get_color method")
        if not callable(getattr(taxonomy_obj, 'get_color', None)):
            raise ValueError("taxonomy_obj.get_color is not callable")
        
        # Images are colored with category colors, but evaluator expects super-category colors
        # Use LayoutSegmentor to convert: category colors -> category IDs -> super-category IDs -> super-category colors
        try:
            category_segmentor = LayoutSegmentor(taxonomy_obj, mode="category")
        except Exception as e:
            raise ValueError(f"Failed to create LayoutSegmentor: {e}")
        
        if category_segmentor is None:
            raise ValueError("category_segmentor is None after initialization")
        if not hasattr(category_segmentor, 'segment') or not callable(getattr(category_segmentor, 'segment', None)):
            raise ValueError("category_segmentor.segment is not callable")
        
        try:
            evaluator = LayoutEvaluator(taxonomy_obj, mode="super", cooccurrence_radius=0.15)
        except Exception as e:
            raise ValueError(f"Failed to create LayoutEvaluator: {e}")
        
        if evaluator is None:
            raise ValueError("evaluator is None after initialization")
        
        print(f"  Computing layout-specific metrics (coverage, class matching)...")
        
        # Accumulate metrics across batch
        all_dist_metrics = []
        all_density_diffs = []
        all_class_ious = []
        
        B = pred_images.shape[0]
        for i in range(B):
            # Convert tensors to PIL Images
            pred_img = pred_images[i].cpu()
            gt_img = gt_images[i].cpu()
            
            # Convert from [C, H, W] to [H, W, C] and to uint8
            if pred_img.shape[0] == 3:
                pred_np = pred_img.permute(1, 2, 0).numpy()
                gt_np = gt_img.permute(1, 2, 0).numpy()
            else:
                pred_np = pred_img.numpy()
                gt_np = gt_img.numpy()
            
            # Normalize to [0, 255] if needed
            if pred_np.max() <= 1.0:
                pred_np = (pred_np * 255).astype(np.uint8)
                gt_np = (gt_np * 255).astype(np.uint8)
            else:
                pred_np = pred_np.astype(np.uint8)
                gt_np = gt_np.astype(np.uint8)
            
            # Convert to PIL Images
            pred_pil = Image.fromarray(pred_np)
            gt_pil = Image.fromarray(gt_np)
            
            # Segment category-colored images to get category IDs, then convert to super-category colors
            # LayoutSegmentor finds closest color for each pixel and assigns category ID
            pred_cat_ids = category_segmentor.segment(pred_pil)  # (H, W) array of category IDs
            gt_cat_ids = category_segmentor.segment(gt_pil)  # (H, W) array of category IDs
            
            # Convert category IDs to super-category IDs, then to super-category colors
            def convert_to_super_colors(cat_id_map):
                """Convert category ID map to super-category colored image."""
                H, W = cat_id_map.shape
                super_colors = np.zeros((H, W, 3), dtype=np.uint8)
                
                # Vectorized conversion: map each category ID to its super-category color
                unique_cat_ids = np.unique(cat_id_map)
                for cat_id in unique_cat_ids:
                    if cat_id == 0:  # Background/unknown
                        continue
                    try:
                        # Resolve category ID to super-category ID
                        super_id = taxonomy_obj.resolve_super(int(cat_id))
                        if super_id is None:
                            continue
                        # Get super-category color - use default mode (resolves to super automatically)
                        # get_color with any mode other than "category" will resolve to super-category
                        super_color = taxonomy_obj.get_color(super_id, mode="none")
                        if super_color is None:
                            continue
                        # Convert color tuple to numpy array
                        if isinstance(super_color, (list, tuple)) and len(super_color) == 3:
                            super_color_arr = np.array(super_color, dtype=np.uint8)
                        else:
                            continue
                        # Assign color to all pixels with this category ID
                        mask = (cat_id_map == cat_id)
                        super_colors[mask] = super_color_arr
                    except Exception as e:
                        # Skip this category if conversion fails
                        warnings.warn(f"Failed to convert category {cat_id} to super-category: {e}")
                        continue
                
                return Image.fromarray(super_colors)
            
            # Convert to super-category colored images
            pred_super_pil = convert_to_super_colors(pred_cat_ids)
            gt_super_pil = convert_to_super_colors(gt_cat_ids)
            
            # Now use evaluator with super-category colored images
            pred_counts = evaluator.count_objects(pred_super_pil)
            gt_counts = evaluator.count_objects(gt_super_pil)
            
            # Compare distributions (class matching metrics)
            dist_metrics = evaluator.compare_distributions(pred_counts, gt_counts)
            all_dist_metrics.append(dist_metrics)
            
            # Coverage/density analysis - measures spatial coverage (objects per pixel)
            pred_density = evaluator.analyze_bbox_density(pred_super_pil)
            gt_density = evaluator.analyze_bbox_density(gt_super_pil)
            density_diff = pred_density["overall_density"] - gt_density["overall_density"]
            all_density_diffs.append(density_diff)
            
            # Class set IoU (class matching)
            pred_classes = set(pred_counts.keys())
            gt_classes = set(gt_counts.keys())
            intersection = len(pred_classes & gt_classes)
            union = len(pred_classes | gt_classes)
            class_iou = intersection / union if union > 0 else 0.0
            all_class_ious.append(class_iou)
        
        # Average metrics across batch
        if len(all_dist_metrics) > 0:
            # Class matching metrics (from distribution comparison)
            metrics["class_kl_divergence"] = float(np.mean([d["kl_divergence"] for d in all_dist_metrics]))
            metrics["class_total_variation"] = float(np.mean([d["total_variation"] for d in all_dist_metrics]))
            metrics["class_l1_distance"] = float(np.mean([d["l1_distance"] for d in all_dist_metrics]))
            metrics["class_iou"] = float(np.mean(all_class_ious))  # Intersection over Union of class sets
            
            # Coverage metric (density difference) - positive means more objects, negative means fewer
            metrics["coverage_diff"] = float(np.mean(all_density_diffs))
            print(f"  Layout metrics computed: coverage_diff={metrics['coverage_diff']:.6f}, class_iou={metrics['class_iou']:.4f}")
        else:
            warnings.warn("Layout-specific metrics: no valid samples processed")
    except Exception as e:
        # Print full traceback for debugging
        import traceback
        error_msg = f"Layout-specific metrics computation failed: {e}\n{traceback.format_exc()}"
        warnings.warn(error_msg)
        print(f"  ERROR: {error_msg}")  # Also print to stdout for visibility
    
    return metrics

