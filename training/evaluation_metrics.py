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
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    warnings.warn("torchvision not available. FID computation will be disabled.")


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
                    _inception_model.load_state_dict(torch.load(inception_path, map_location=device))
                    _inception_model.fc = torch.nn.Identity()
                    print(f"  Loaded Inception model from shared cache: {inception_path}")
                except Exception as e:
                    warnings.warn(f"Failed to load Inception from {inception_path}, downloading: {e}")
                    inception_path = None  # Fall back to downloading
            
            # Download if not found locally
            if _inception_model is None:
                _inception_model = inception_v3(pretrained=True, transform_input=False)
                _inception_model.fc = torch.nn.Identity()  # Remove final classification layer
                
                # Save to shared location
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
        # Compute mean and covariance
        mu1 = real_features.mean(dim=0).numpy()
        sigma1 = np.cov(real_features.numpy(), rowvar=False)
        
        mu2 = fake_features.mean(dim=0).numpy()
        sigma2 = np.cov(fake_features.numpy(), rowvar=False)
        
        # Compute FID
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        
        if not np.isfinite(covmean).all():
            # Add small epsilon to diagonal if singular
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
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
    
    if device is None:
        device = pred_images.device if isinstance(pred_images, torch.Tensor) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # CLIP Score
    if compute_clip:
        clip_metrics = compute_clip_score(pred_images, text_emb, pov_emb, device, use_text_prompt=True)
        metrics.update(clip_metrics)
    
    # FID (requires accumulating features across batches - this is a per-batch approximation)
    if compute_fid:
        try:
            pred_features = compute_fid_features(pred_images, device)
            gt_features = compute_fid_features(gt_images, device)
            if pred_features is not None and gt_features is not None:
                # Note: This is a per-batch FID, not the full dataset FID
                # For accurate FID, you need to accumulate features across all validation batches
                fid_score = compute_fid(gt_features, pred_features)
                if np.isfinite(fid_score):
                    metrics["fid"] = fid_score
        except Exception as e:
            warnings.warn(f"FID computation failed: {e}")
    
    # mIoU (mean Intersection over Union) - Critical for geometric/spatial correctness
    # Since layouts are geometric, mIoU between generated and ground truth segmentation maps
    # provides a much more accurate assessment of spatial correctness than pixel-wise MSE.
    # It measures how well the model preserves object boundaries and spatial relationships.
    if compute_miou and taxonomy is not None:
        try:
            from data_preparation.utils.layout_analysis import LayoutSegmentor
            segmentor = LayoutSegmentor(taxonomy, mode="category")
            
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
                iou = compute_miou(pred_seg, gt_seg)
                ious.append(iou)
            
            if len(ious) > 0:
                metrics["miou"] = float(np.mean(ious))
        except Exception as e:
            warnings.warn(f"mIoU computation failed: {e}")
    
    # Layout-specific metrics (coverage, class matching, color matching)
    # These don't require external models - they analyze geometric properties directly
    if taxonomy is not None:
        try:
            from common.taxonomy import Taxonomy
            from analysis.evaluation_metrics import LayoutEvaluator
            from data_preparation.utils.layout_analysis import LayoutSegmentor
            
            # Ensure taxonomy is a Taxonomy instance
            if isinstance(taxonomy, (str, Path)):
                taxonomy_obj = Taxonomy(taxonomy)
            else:
                taxonomy_obj = taxonomy
            
            # Images are colored with category colors, but evaluator expects super-category colors
            # Use LayoutSegmentor to convert: category colors -> category IDs -> super-category IDs -> super-category colors
            try:
                category_segmentor = LayoutSegmentor(taxonomy_obj, mode="category")
            except Exception as e:
                raise ValueError(f"Failed to create LayoutSegmentor: {e}")
            
            try:
                evaluator = LayoutEvaluator(taxonomy_obj, mode="super", cooccurrence_radius=0.15)
            except Exception as e:
                raise ValueError(f"Failed to create LayoutEvaluator: {e}")
            
            # Verify methods exist
            if not hasattr(taxonomy_obj, 'resolve_super') or taxonomy_obj.resolve_super is None:
                raise ValueError("taxonomy_obj.resolve_super is None or doesn't exist")
            if not hasattr(taxonomy_obj, 'get_color') or taxonomy_obj.get_color is None:
                raise ValueError("taxonomy_obj.get_color is None or doesn't exist")
            
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

