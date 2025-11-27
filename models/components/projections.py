# models/components/projections.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from .base_component import BaseComponent


# Projection Registry (defined early for use in classes)
PROJECTION_REGISTRY = {}


# -----------------------
# Helper functions for common projection patterns
# -----------------------

def infer_batch_and_device(text_emb=None, pov_emb=None, batch_size=None, device=None, fallback_tensor=None):
    """
    Infer batch size and device from available inputs.
    
    Args:
        text_emb: Text embeddings tensor or None
        pov_emb: POV embeddings tensor or None
        batch_size: Explicit batch size or None
        device: Explicit device or None
        fallback_tensor: Tensor to use for inference if text_emb and pov_emb are None
    
    Returns:
        Tuple (batch_size, device, dtype)
    """
    if text_emb is not None:
        return text_emb.shape[0], text_emb.device, text_emb.dtype
    elif pov_emb is not None:
        return pov_emb.shape[0], pov_emb.device, pov_emb.dtype
    elif fallback_tensor is not None:
        return fallback_tensor.shape[0], fallback_tensor.device, fallback_tensor.dtype
    elif batch_size is not None and device is not None:
        dtype = torch.float32  # Default dtype
        return batch_size, device, dtype
    else:
        raise ValueError(
            "At least one of text_emb, pov_emb, or (batch_size, device) must be provided, "
            "or provide fallback_tensor"
        )


def get_dtype_from_module(module):
    """Get dtype from module parameters."""
    return next(module.parameters()).dtype


def project_to_spatial(emb, projection_module, target_size, batch_size=None, device=None, normalize=True):
    """
    Project embedding to spatial dimensions.
    
    Args:
        emb: Embedding tensor [B, dim] or None
        projection_module: Module to project with
        target_size: Target spatial size (H, W)
        batch_size: Batch size (required if emb is None)
        device: Device (required if emb is None)
        normalize: Whether to normalize after projection
    
    Returns:
        Spatial tensor [B, projection_dim, H, W]
    """
    if emb is None:
        if batch_size is None or device is None:
            raise ValueError("batch_size and device required when emb is None")
        proj_dtype = get_dtype_from_module(projection_module)
        # Get output dimension from projection module
        if hasattr(projection_module, 'out_features'):
            out_dim = projection_module.out_features
        elif hasattr(projection_module, 'projection_dim'):
            out_dim = projection_module.projection_dim
        else:
            # Try to infer from Sequential module
            last_layer = projection_module[-1] if isinstance(projection_module, nn.Sequential) else projection_module
            out_dim = last_layer.out_features if hasattr(last_layer, 'out_features') else 256
        proj = torch.zeros(batch_size, out_dim, device=device, dtype=proj_dtype)
    else:
        # Flatten if needed
        emb_flat = emb.flatten(start_dim=1) if emb.dim() > 2 else emb
        proj_dtype = get_dtype_from_module(projection_module)
        emb_flat = emb_flat.to(dtype=proj_dtype)
        proj = projection_module(emb_flat)
        if normalize:
            proj = F.normalize(proj, p=2, dim=1)
    
    # Expand to spatial dimensions
    H, W = target_size
    spatial = proj.unsqueeze(-1).unsqueeze(-1)  # [B, dim, 1, 1]
    spatial = F.interpolate(spatial, size=(H, W), mode='bilinear', align_corners=False)  # [B, dim, H, W]
    if normalize:
        spatial = F.normalize(spatial, p=2, dim=1)
    
    return spatial


def combine_embeddings(emb1, emb2, method="average"):
    """
    Combine two embeddings using specified method.
    
    Args:
        emb1: First embedding tensor or None
        emb2: Second embedding tensor or None
        method: "add", "average", or "concat"
    
    Returns:
        Combined embedding tensor
    """
    if emb1 is not None and emb2 is not None:
        if method == "add":
            combined = emb1 + emb2
        elif method == "average":
            combined = (emb1 + emb2) / 2.0
        elif method == "concat":
            combined = torch.cat([emb1, emb2], dim=1)
        else:
            # Default to average
            combined = (emb1 + emb2) / 2.0
    elif emb1 is not None:
        combined = emb1
    elif emb2 is not None:
        combined = emb2
    else:
        raise ValueError("At least one embedding must be provided")
    
    return combined


def register_projection(cls):
    """Decorator to register projection classes."""
    PROJECTION_REGISTRY[cls.__name__] = cls
    return cls


def from_config_with_registry(cfg, registry=None):
    """
    Create a component from config using a registry.
    
    Args:
        cfg: Config dict with "type" field
        registry: Registry dict mapping type names to classes (default: PROJECTION_REGISTRY)
    
    Returns:
        Instance of the registered class
    """
    if registry is None:
        registry = PROJECTION_REGISTRY
    
    cfg = cfg.get("model", cfg) if isinstance(cfg, dict) else cfg
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a dict, got {type(cfg)}")
    
    comp_type = cfg.get("type")
    if comp_type is None:
        raise ValueError("Config must have 'type' field")
    
    if comp_type not in registry:
        raise ValueError(
            f"Unknown type '{comp_type}' in registry. "
            f"Available types: {list(registry.keys())}"
        )
    
    comp_cls = registry[comp_type]
    comp_cfg = {k: v for k, v in cfg.items() if k != "type"}
    return comp_cls(**comp_cfg)


class BaseProjection(BaseComponent):
    """
    Base class for projection layers that map embeddings to joint CLIP space.
    
    Handles shared logic:
    - Projection network (Linear layers with normalization)
    - None input handling (returns zeros)
    - Flattening and normalization
    - Dtype matching for mixed precision
    
    Subclasses should define:
    - input_dim: Dimension of input embeddings
    - projection_dim: Dimension of output joint space
    """
    def _build(self):
        self.projection_dim = self._init_kwargs.get("projection_dim", 256)
        
        # Build projection network
        # This will be overridden by subclasses to set input_dim
        self.proj = None
    
    def _build_projection(self, input_dim):
        """Build the projection network."""
        self.proj = nn.Sequential(
            nn.Linear(input_dim, self.projection_dim * 2),
            nn.LayerNorm(self.projection_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.projection_dim * 2, self.projection_dim),
            nn.LayerNorm(self.projection_dim)
        )
    
    def forward(self, emb, batch_size=None, device=None):
        """
        Project embeddings to joint space.
        
        Args:
            emb: Embeddings [B, input_dim] or [B, ..., input_dim], or None
            batch_size: Batch size (required if emb is None)
            device: Device for output tensor (required if emb is None)
        
        Returns:
            Projected embeddings [B, projection_dim] or [B, ..., projection_dim]
            If emb is None, returns zeros of shape [batch_size, projection_dim]
        """
        if emb is None:
            if batch_size is None:
                raise ValueError(
                    f"{self.__class__.__name__} received None input but batch_size is not provided. "
                    "Either provide emb or specify batch_size and device."
                )
            if device is None:
                # Try to get device from projection parameters
                device = next(self.proj.parameters()).device
            
            # Return zeros of projection shape
            proj_dtype = get_dtype_from_module(self.proj)
            return torch.zeros(batch_size, self.projection_dim, device=device, dtype=proj_dtype)
        
        # Flatten if needed
        if emb.dim() > 2:
            original_shape = emb.shape
            emb = emb.flatten(start_dim=1)
            # Ensure we have the right dimension
            input_dim = self._get_input_dim()
            if emb.shape[-1] != input_dim:
                raise ValueError(
                    f"Expected input_dim={input_dim}, got {emb.shape[-1]}. "
                    f"Input shape: {original_shape}"
                )
        
        # Ensure dtype matches projection parameters (for mixed precision training)
        proj_dtype = get_dtype_from_module(self.proj)
        emb = emb.to(dtype=proj_dtype)
        
        projected = self.proj(emb)
        projected = F.normalize(projected, p=2, dim=-1)
        
        return projected
    
    def _get_input_dim(self):
        """Get input dimension. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _get_input_dim()")


class TextProjection(BaseProjection):
    """
    Projects text/graph embeddings to joint CLIP space.
    
    Config:
        text_dim: Dimension of input text embeddings (default: 384)
        projection_dim: Dimension of output joint space (default: 256)
    """
    def _build(self):
        super()._build()
        self.text_dim = self._init_kwargs.get("text_dim", 384)
        self._build_projection(self.text_dim)
    
    def _get_input_dim(self):
        """Get input dimension."""
        return self.text_dim
    
    def to_config(self):
        cfg = super().to_config()
        cfg.update({
            "text_dim": self.text_dim,
        })
        return cfg


class ImageProjection(BaseProjection):
    """
    Projects image/POV embeddings to joint CLIP space.
    
    Config:
        pov_dim: Dimension of input POV/image embeddings (default: 512)
        projection_dim: Dimension of output joint space (default: 256)
    """
    def _build(self):
        super()._build()
        self.pov_dim = self._init_kwargs.get("pov_dim", 512)
        self._build_projection(self.pov_dim)
    
    def _get_input_dim(self):
        """Get input dimension."""
        return self.pov_dim
    
    def to_config(self):
        cfg = super().to_config()
        cfg.update({
            "pov_dim": self.pov_dim,
        })
        return cfg


class LatentProjection(BaseProjection):
    """
    Projects VAE latent features to joint CLIP space.
    
    Can work in two modes:
    - Global mode: Uses Linear layers (for flattened/pooled features)
    - Spatial mode: Uses Conv2d layers (for spatial features [B, C, H, W])
    
    Config:
        latent_dim: Dimension of input latent features (required)
        projection_dim: Dimension of output joint space (default: 256)
        spatial_mode: If True, uses Conv2d for spatial features (default: False)
    """
    def _build(self):
        super()._build()
        self.latent_dim = self._init_kwargs.get("latent_dim", None)
        self.spatial_mode = self._init_kwargs.get("spatial_mode", False)
        
        if self.latent_dim is None:
            raise ValueError("LatentProjection requires latent_dim")
        
        if self.spatial_mode:
            # Use Conv2d for spatial features
            self.proj = nn.Sequential(
                nn.Conv2d(self.latent_dim, self.projection_dim * 2, 1),
                nn.GroupNorm(8, self.projection_dim * 2),
                nn.GELU(),
                nn.Dropout2d(0.1),
                nn.Conv2d(self.projection_dim * 2, self.projection_dim, 1),
                nn.GroupNorm(8, self.projection_dim)
            )
        else:
            # Use Linear for global features
            self._build_projection(self.latent_dim)
    
    def _get_input_dim(self):
        """Get input dimension."""
        return self.latent_dim
    
    def forward(self, latent_features, batch_size=None, device=None):
        """
        Project latent features to joint space.
        
        Args:
            latent_features: VAE features [B, C, H, W] (spatial) or [B, D] (global), or None
            batch_size: Batch size (required if latent_features is None)
            device: Device for output tensor (required if latent_features is None)
        
        Returns:
            Projected features [B, projection_dim, H, W] (spatial) or [B, projection_dim] (global)
            If latent_features is None, returns zeros
        """
        if latent_features is None:
            if batch_size is None:
                raise ValueError(
                    "LatentProjection received None input but batch_size is not provided. "
                    "Either provide latent_features or specify batch_size and device."
                )
            if device is None:
                device = next(self.proj.parameters()).device
            
            proj_dtype = get_dtype_from_module(self.proj)
            if self.spatial_mode:
                # Return zeros with spatial dimensions (need to know H, W)
                # For now, return a placeholder - caller should handle this
                raise ValueError(
                    "LatentProjection in spatial_mode cannot handle None input without spatial dimensions. "
                    "Provide latent_features or handle None case in calling code."
                )
            else:
                return torch.zeros(batch_size, self.projection_dim, device=device, dtype=proj_dtype)
        
        # Handle spatial vs global mode
        if self.spatial_mode:
            # Expect [B, C, H, W]
            if latent_features.dim() != 4:
                raise ValueError(
                    f"LatentProjection in spatial_mode expects 4D input [B, C, H, W], "
                    f"got {latent_features.shape}"
                )
            B, C, H, W = latent_features.shape
            if C != self.latent_dim:
                raise ValueError(
                    f"Expected latent_dim={self.latent_dim}, got C={C}. "
                    f"Input shape: {latent_features.shape}"
                )
            
            proj_dtype = next(self.proj.parameters()).dtype
            latent_features = latent_features.to(dtype=proj_dtype)
            projected = self.proj(latent_features)  # [B, projection_dim, H, W]
            projected = F.normalize(projected, p=2, dim=1)
            return projected
        else:
            # Global mode: use base class forward
            return super().forward(latent_features, batch_size=batch_size, device=device)
    
    def to_config(self):
        cfg = super().to_config()
        cfg.update({
            "latent_dim": self.latent_dim,
            "spatial_mode": self.spatial_mode,
        })
        return cfg


class CLIPProjections(BaseComponent):
    """
    Standalone CLIP projection layers that can be attached to a model.
    These project VAE features, text embeddings, and POV embeddings to a joint space.
    
    Supports both global and spatial alignment modes:
    - Global mode (default): Pools spatial features to global, aligns global-to-global
    - Spatial mode: Preserves spatial structure, projects global conditions to spatial dimensions
    """
    def _build(self):
        self.projection_dim = self._init_kwargs.get("projection_dim", 256)
        self.text_dim = self._init_kwargs.get("text_dim", 384)
        self.pov_dim = self._init_kwargs.get("pov_dim", 512)
        self._latent_dim = self._init_kwargs.get("latent_dim", None)
        self.spatial_alignment = self._init_kwargs.get("spatial_alignment", False)
        
        # Build text and image projections as separate components
        # Support both direct config and registry-based config
        text_proj_cfg = self._init_kwargs.get("text_projection", {})
        if not text_proj_cfg:
            text_proj_cfg = {
                "type": "TextProjection",
                "text_dim": self.text_dim,
                "projection_dim": self.projection_dim,
            }
        elif isinstance(text_proj_cfg, dict) and "type" not in text_proj_cfg:
            # Legacy: if no type specified, assume TextProjection
            text_proj_cfg = {**text_proj_cfg, "type": "TextProjection"}
        
        if isinstance(text_proj_cfg, dict) and "type" in text_proj_cfg:
            self.text_proj = from_config_with_registry(text_proj_cfg, PROJECTION_REGISTRY)
        else:
            # Direct instantiation (backward compatibility)
            self.text_proj = TextProjection(**text_proj_cfg) if text_proj_cfg else TextProjection(
                text_dim=self.text_dim, projection_dim=self.projection_dim
            )
        
        pov_proj_cfg = self._init_kwargs.get("image_projection", {})
        if not pov_proj_cfg:
            pov_proj_cfg = {
                "type": "ImageProjection",
                "pov_dim": self.pov_dim,
                "projection_dim": self.projection_dim,
            }
        elif isinstance(pov_proj_cfg, dict) and "type" not in pov_proj_cfg:
            # Legacy: if no type specified, assume ImageProjection
            pov_proj_cfg = {**pov_proj_cfg, "type": "ImageProjection"}
        
        if isinstance(pov_proj_cfg, dict) and "type" in pov_proj_cfg:
            self.pov_proj = from_config_with_registry(pov_proj_cfg, PROJECTION_REGISTRY)
        else:
            # Direct instantiation (backward compatibility)
            self.pov_proj = ImageProjection(**pov_proj_cfg) if pov_proj_cfg else ImageProjection(
                pov_dim=self.pov_dim, projection_dim=self.projection_dim
            )
        
        # VAE latent features -> joint space (will be initialized dynamically)
        # Support both direct config and registry-based config
        latent_proj_cfg = self._init_kwargs.get("latent_projection", {})
        self.latent_proj = None
        if self._latent_dim is not None:
            if not latent_proj_cfg:
                latent_proj_cfg = {
                    "type": "LatentProjection",
                    "latent_dim": self._latent_dim,
                    "projection_dim": self.projection_dim,
                    "spatial_mode": False,  # Will be set dynamically if needed
                }
            elif isinstance(latent_proj_cfg, dict) and "type" not in latent_proj_cfg:
                latent_proj_cfg = {**latent_proj_cfg, "type": "LatentProjection"}
            
            if isinstance(latent_proj_cfg, dict) and "type" in latent_proj_cfg:
                # Use registry
                latent_proj_cfg["latent_dim"] = self._latent_dim
                latent_proj_cfg["projection_dim"] = self.projection_dim
                self.latent_proj = from_config_with_registry(latent_proj_cfg, PROJECTION_REGISTRY)
            else:
                # Legacy: use old method
                self._init_latent_proj(self._latent_dim)
        
        # Spatial projection for global conditions (only used in spatial_alignment mode)
        # Projects global embeddings to spatial feature maps
        self.spatial_text_proj = None
        self.spatial_pov_proj = None
        # Initialize spatial dimension tracking (needed for loading checkpoints)
        self._spatial_h = None
        self._spatial_w = None
    
    def _init_latent_proj(self, latent_dim, device=None, spatial_mode=False):
        """
        Initialize latent projection (legacy method for backward compatibility).
        
        Creates a LatentProjection instance.
        """
        if self.latent_proj is None or self._latent_dim != latent_dim:
            self._latent_dim = latent_dim
            
            # Use LatentProjection class
            proj = LatentProjection(
                latent_dim=latent_dim,
                projection_dim=self.projection_dim,
                spatial_mode=spatial_mode
            )
            
            if device is not None:
                proj = proj.to(device)
            self.latent_proj = proj
    
    def _init_spatial_projections(self, h, w, device=None):
        """Initialize spatial projections for global conditions."""
        # Validate dimensions to prevent memory issues
        if h is None or w is None:
            raise ValueError(f"Invalid spatial dimensions: h={h}, w={w}")
        
        # Convert to int if they're tensors
        if isinstance(h, torch.Tensor):
            h = int(h.item())
        if isinstance(w, torch.Tensor):
            w = int(w.item())
        
        h, w = int(h), int(w)
        
        # Safety check: if dimensions are unreasonably large, something is wrong
        # Latent features should typically be 32x32, 64x64, or at most 128x128
        MAX_SPATIAL_DIM = 512  # Reasonable upper bound
        if h > MAX_SPATIAL_DIM or w > MAX_SPATIAL_DIM:
            raise ValueError(
                f"Spatial dimensions are too large: H={h}, W={w}. "
                f"This suggests latent_features has wrong shape. "
                f"Expected latent features (e.g., 32x32, 64x64), got {h}x{w}. "
                f"Check that latent_features is from encoder output, not raw image."
            )
        
        if self._spatial_h == h and self._spatial_w == w and self.spatial_text_proj is not None:
            return  # Already initialized
        
        self._spatial_h = h
        self._spatial_w = w
        
        # For spatial mode, create separate spatial projections
        # These are similar to text/image projections but without final LayerNorm
        # (since they'll be expanded spatially and normalized later)
        # Create a simpler version without the final LayerNorm for spatial use
        self.spatial_text_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.projection_dim * 2),
            nn.LayerNorm(self.projection_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.projection_dim * 2, self.projection_dim)
        )
        
        self.spatial_pov_proj = nn.Sequential(
            nn.Linear(self.pov_dim, self.projection_dim * 2),
            nn.LayerNorm(self.projection_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.projection_dim * 2, self.projection_dim)
        )
        
        if device is not None:
            self.spatial_text_proj = self.spatial_text_proj.to(device)
            self.spatial_pov_proj = self.spatial_pov_proj.to(device)
    
    def forward(self, latent_features, text_emb, pov_emb, combine_method="average"):
        """
        Project all embeddings to joint space.
        
        Args:
            latent_features: VAE features [B, C, H, W] or [B, D]
            text_emb: Text embeddings [B, text_dim]
            pov_emb: POV embeddings [B, pov_dim] or None (for scenes)
            combine_method: How to combine text and POV ("add", "concat", "average")
        
        Returns:
            Global mode: (latent_proj, combined_emb) where both are [B, projection_dim]
            Spatial mode: (latent_proj, combined_emb) where both are [B, projection_dim, H, W]
        """
        if self.spatial_alignment and latent_features.dim() == 4:
            # Spatial alignment mode: preserve spatial structure
            B, C, H, W = latent_features.shape
            
            # Validate input shape - latent features should be small (e.g., 32x32, 64x64)
            if H > 512 or W > 512:
                raise ValueError(
                    f"latent_features has unreasonably large spatial dimensions: {H}x{W}. "
                    f"Expected latent features from encoder (typically 32x32, 64x64, or at most 128x128). "
                    f"Got shape: {latent_features.shape}. "
                    f"This suggests the input might be a raw image instead of encoder features. "
                    f"Check that you're passing encoder output, not the original image."
                )
            
            # Initialize spatial projections if needed
            self._init_spatial_projections(H, W, latent_features.device)
            
            # Project VAE features spatially: [B, C, H, W] -> [B, projection_dim, H, W]
            # Check if we need to create/update latent projection
            needs_update = (
                self.latent_proj is None or 
                self._latent_dim != C or
                not isinstance(self.latent_proj, LatentProjection) or
                not self.latent_proj.spatial_mode
            )
            
            if needs_update:
                self._latent_dim = C
                # Create or update LatentProjection in spatial mode
                self.latent_proj = LatentProjection(
                    latent_dim=C,
                    projection_dim=self.projection_dim,
                    spatial_mode=True
                ).to(latent_features.device)
            
            # Use LatentProjection forward
            latent_proj = self.latent_proj(latent_features)  # [B, projection_dim, H, W]
            
            # Project global conditions to joint space, then expand to spatial dimensions
            # Handle None inputs by using zeros
            if self.spatial_text_proj is not None:
                text_spatial = project_to_spatial(
                    text_emb, 
                    self.spatial_text_proj, 
                    (H, W),
                    batch_size=B,
                    device=latent_features.device
                )
            else:
                proj_dtype = get_dtype_from_module(self.latent_proj)
                text_spatial = torch.zeros(B, self.projection_dim, H, W, device=latent_features.device, dtype=proj_dtype)
            
            if pov_emb is None:
                combined_emb = text_spatial
            else:
                if self.spatial_pov_proj is not None:
                    pov_spatial = project_to_spatial(
                        pov_emb,
                        self.spatial_pov_proj,
                        (H, W),
                        batch_size=B,
                        device=latent_features.device
                    )
                else:
                    proj_dtype = get_dtype_from_module(self.latent_proj)
                    pov_spatial = torch.zeros(B, self.projection_dim, H, W, device=latent_features.device, dtype=proj_dtype)
                
                combined_emb = combine_embeddings(text_spatial, pov_spatial, method=combine_method)
                combined_emb = F.normalize(combined_emb, p=2, dim=1)
            
            return latent_proj, combined_emb
        
        # Global alignment mode (original behavior)
        if latent_features.dim() > 2:
            if latent_features.dim() == 4:
                latent_features = F.adaptive_avg_pool2d(latent_features, 1).squeeze(-1).squeeze(-1)
            else:
                latent_features = latent_features.flatten(start_dim=1)
        
        latent_dim = latent_features.shape[1]
        if self.latent_proj is None or self._latent_dim != latent_dim:
            # Create or update LatentProjection in global mode
            self._latent_dim = latent_dim
            self.latent_proj = LatentProjection(
                latent_dim=latent_dim,
                projection_dim=self.projection_dim,
                spatial_mode=False
            ).to(latent_features.device)
        
        # Use text and image projection components
        # Infer batch size and device from available inputs
        batch_size, device, _ = infer_batch_and_device(
            text_emb=text_emb, 
            pov_emb=pov_emb, 
            fallback_tensor=latent_features
        )
        
        # Project text embeddings (handle None)
        if text_emb is not None:
            text_proj = self.text_proj(text_emb)
        else:
            text_proj = self.text_proj(None, batch_size=batch_size, device=device)
        
        # Project POV embeddings (handle None)
        if pov_emb is not None:
            pov_proj = self.pov_proj(pov_emb)
        else:
            pov_proj = self.pov_proj(None, batch_size=batch_size, device=device)
        
        # Combine text and POV projections
        combined_emb = combine_embeddings(text_proj, pov_proj, method=combine_method)
        
        combined_emb = F.normalize(combined_emb, p=2, dim=1)
        
        # Use LatentProjection forward
        latent_proj = self.latent_proj(latent_features)
        
        return latent_proj, combined_emb
    
    def save_checkpoint(self, path, include_config=True, global_only=False):
        """
        Save CLIP projections checkpoint.
        
        Args:
            path: Path to save checkpoint
            include_config: Whether to include model config
            global_only: If True, only save global projections (text_proj, pov_proj)
                        without spatial projections. Useful for diffusion models.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state_dict = self.state_dict()
        
        if global_only:
            # Filter to only include global projections (text_proj, pov_proj)
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("text_proj.") or key.startswith("pov_proj."):
                    filtered_state_dict[key] = value
            state_dict = filtered_state_dict
        else:
            # Save all projections including spatial if they exist
            # spatial_text_proj and spatial_pov_proj are included in state_dict() if they exist
            pass
        
        payload = {"state_dict": state_dict}
        if include_config:
            cfg = self.to_config()
            if global_only:
                cfg["global_only"] = True
            payload["config"] = cfg
        
        torch.save(payload, path)
    
    @classmethod
    def load_checkpoint(cls, path, map_location="cpu", global_only=False):
        """
        Load CLIP projections checkpoint.
        
        Args:
            path: Path to checkpoint
            map_location: Device to load on
            global_only: If True, create a global-only version (spatial_alignment=False)
                        and only load text_proj and pov_proj. Useful for diffusion models.
        
        Returns:
            CLIPProjections instance
        """
        path = Path(path)
        payload = torch.load(path, map_location=map_location)
        
        config = payload.get("config")
        state_dict = payload.get("state_dict", payload)
        
        # If global_only is requested or config indicates global_only, create global-only version
        if global_only or (config and config.get("global_only", False)):
            if config:
                global_config = config.copy()
            else:
                global_config = {
                    "projection_dim": 256,
                    "text_dim": 384,
                    "pov_dim": 512,
                }
            
            global_config["spatial_alignment"] = False
            global_config["latent_dim"] = None
            
            instance = cls(**global_config)
            
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("text_proj.") or key.startswith("pov_proj."):
                    filtered_state_dict[key] = value
            
            instance.load_state_dict(filtered_state_dict, strict=False)
            
            instance.spatial_text_proj = None
            instance.spatial_pov_proj = None
            instance._spatial_h = None
            instance._spatial_w = None
            
            return instance
        else:
            if config:
                instance = cls.from_config(config)
            else:
                instance = cls()
            
            # Check if spatial projections exist in state dict BEFORE loading
            # If they do, we need to create them first so they can be loaded
            # Keys might be prefixed (e.g., "clip_projections.spatial_text_proj.") or direct ("spatial_text_proj.")
            has_spatial_text = any("spatial_text_proj." in k for k in state_dict.keys())
            has_spatial_pov = any("spatial_pov_proj." in k for k in state_dict.keys())
            
            # Also check config for spatial_alignment (it might be True even if instance doesn't have it yet)
            config_spatial_alignment = False
            if isinstance(config, dict):
                config_spatial_alignment = config.get("spatial_alignment", False)
                # Also check in clip_projection sub-config
                if "clip_projection" in config:
                    clip_cfg = config["clip_projection"]
                    if isinstance(clip_cfg, dict):
                        config_spatial_alignment = clip_cfg.get("spatial_alignment", config_spatial_alignment)
            
            # If config says spatial_alignment=True, set it on the instance
            if config_spatial_alignment and not instance.spatial_alignment:
                instance.spatial_alignment = True
            
            # If spatial projections exist in checkpoint OR config says spatial_alignment=True, initialize them
            if (has_spatial_text or has_spatial_pov) or (instance.spatial_alignment or config_spatial_alignment):
                # Spatial projections exist in checkpoint or config indicates spatial mode
                # We need to initialize them with dummy dimensions so they can be loaded
                # Use a default size (they'll be re-initialized with correct size on first forward if needed)
                default_h, default_w = 64, 64  # Common latent size
                instance._init_spatial_projections(default_h, default_w, device=map_location)
                print(f"[INFO] Initialized spatial projections for loading (will use dimensions from checkpoint)")
            
            # Load state dict - this will include spatial projections if they exist
            missing_keys, unexpected_keys = instance.load_state_dict(state_dict, strict=False)
            
            # Verify spatial projections were loaded
            if has_spatial_text or has_spatial_pov:
                if instance.spatial_text_proj is not None or instance.spatial_pov_proj is not None:
                    print(f"[OK] Loaded spatial CLIP projections from checkpoint")
                else:
                    print(f"[WARNING] Spatial projections in checkpoint but not loaded properly")
            elif instance.spatial_alignment:
                # Spatial alignment is enabled but projections weren't in checkpoint
                # They'll be created dynamically when forward() is called
                print(f"[INFO] Spatial alignment enabled but projections not in checkpoint - will be created on first forward()")
            
            return instance
    
    def extract_global_projections(self):
        """
        Create a new CLIPProjections instance with only global projections.
        Useful for extracting text_proj and pov_proj for diffusion models.
        
        Returns:
            New CLIPProjections instance with only global projections
        """
        global_proj = CLIPProjections(
            projection_dim=self.projection_dim,
            text_dim=self.text_dim,
            pov_dim=self.pov_dim,
            latent_dim=None,
            spatial_alignment=False
        )
        
        # Copy state dicts from text and image projections
        # Copy state dicts from text and image projections
        global_proj.text_proj.load_state_dict(self.text_proj.state_dict())
        global_proj.pov_proj.load_state_dict(self.pov_proj.state_dict())
        
        # Note: latent_proj is not copied as it's not needed for global-only version
        
        return global_proj


class BaseEmbeddingToSpatial(BaseComponent):
    """
    Base class for embedding-to-spatial projection components.
    
    Handles common logic:
    - Batch size and device inference
    - None input handling
    - Embedding combination
    - Final spatial projection to [B, output_channels, H, W]
    
    Subclasses should implement:
    - _project_embeddings(): Project text_emb and pov_emb to intermediate representation
    - _final_spatial_projection(): Project intermediate representation to spatial features
    """
    
    def _build(self):
        self.output_channels = self._init_kwargs.get("output_channels", 96)
        self.spatial_size = self._init_kwargs.get("spatial_size", (64, 64))
        self.combine_method = self._init_kwargs.get("combine_method", "average")
    
    def _project_embeddings(self, text_emb, pov_emb, batch_size, device):
        """
        Project text and POV embeddings to intermediate representation.
        Must be implemented by subclasses.
        
        Returns:
            Tuple (text_proj, pov_proj) - both can be None
        """
        raise NotImplementedError
    
    def _final_spatial_projection(self, combined, batch_size, device):
        """
        Project combined representation to final spatial features.
        Must be implemented by subclasses.
        
        Args:
            combined: Combined embedding tensor
            batch_size: Batch size
            device: Device
        
        Returns:
            Spatial tensor [B, output_channels, H, W]
        """
        raise NotImplementedError
    
    def forward(self, text_emb, pov_emb, batch_size=None, device=None):
        """
        Convert embeddings to spatial feature map.
        
        Args:
            text_emb: Text/graph embeddings [B, text_dim] or None
            pov_emb: POV embeddings [B, pov_dim] or None
            batch_size: Batch size (required if both text_emb and pov_emb are None)
            device: Device (required if both text_emb and pov_emb are None)
        
        Returns:
            Spatial feature map [B, output_channels, H, W]
        """
        # Infer batch size and device from available inputs
        B, device, _ = infer_batch_and_device(
            text_emb=text_emb,
            pov_emb=pov_emb,
            batch_size=batch_size,
            device=device
        )
        
        # Project embeddings (subclass-specific)
        text_proj, pov_proj = self._project_embeddings(text_emb, pov_emb, B, device)
        
        # Combine embeddings
        combined = combine_embeddings(text_proj, pov_proj, method=self.combine_method)
        
        # Final spatial projection (subclass-specific)
        spatial = self._final_spatial_projection(combined, B, device)
        
        return spatial


class CLIPEmbeddingToSpatial(BaseEmbeddingToSpatial):
    """
    Projects 1D embeddings to spatial feature maps using CLIP projections.
    
    First projects embeddings through CLIP joint space (from trained VAE),
    then converts to spatial features for cross-attention.
    
    This ensures spatial features (K, V) are in the same semantic space
    as VAE latents (Q), improving cross-attention alignment.
    
    Config:
        clip_projections: CLIPProjections instance or path to VAE checkpoint with CLIP projections
        output_channels: Number of output channels for spatial features (default: matches UNet base_channels)
        spatial_size: Target spatial size (H, W) - will be interpolated to match UNet resolution (default: (64, 64))
        combine_method: How to combine text and POV in CLIP space: "add", "average" (default: "average")
    """
    
    def _build(self):
        super()._build()  # Initialize base class (output_channels, spatial_size, combine_method)
        clip_projections = self._init_kwargs.get("clip_projections", None)
        
        # Load or use provided CLIP projections
        if clip_projections is None:
            raise ValueError(
                "CLIPEmbeddingToSpatial requires clip_projections. "
                "Can be: CLIPProjections instance, path to CLIP projection checkpoint, or path to VAE checkpoint with CLIP projections"
            )
        
        if isinstance(clip_projections, str) or isinstance(clip_projections, Path):
            checkpoint_path = Path(clip_projections)
            
            # Try loading as CLIP projection checkpoint first
            try:
                # Load full CLIP projections (including spatial if available)
                self.clip_projections = CLIPProjections.load_checkpoint(
                    checkpoint_path, map_location="cpu", global_only=False
                )
                print(f"✓ Loaded CLIP projections from standalone checkpoint: {checkpoint_path}")
                if self.clip_projections.spatial_alignment:
                    print(f"  [INFO] Spatial CLIP projections detected - will use spatial projections")
            except Exception as e1:
                # Fall back to loading from VAE checkpoint
                try:
                    from models.autoencoder import Autoencoder
                    autoencoder = Autoencoder.load_checkpoint(checkpoint_path, map_location="cpu")
                    if not hasattr(autoencoder, 'clip_projections') or autoencoder.clip_projections is None:
                        raise ValueError(f"VAE checkpoint {checkpoint_path} does not have CLIP projections")
                    
                    # Use full CLIP projections from VAE (including spatial if trained)
                    self.clip_projections = autoencoder.clip_projections
                    print(f"✓ Loaded full CLIP projections from VAE checkpoint: {checkpoint_path}")
                    if self.clip_projections.spatial_alignment:
                        print(f"  [INFO] Spatial CLIP projections detected - will use spatial projections")
                except Exception as e2:
                    raise ValueError(
                        f"Failed to load CLIP projections from {checkpoint_path}. "
                        f"Tried as CLIP checkpoint: {e1}. "
                        f"Tried as VAE checkpoint: {e2}"
                    ) from e2
        else:
            # Use provided CLIPProjections instance (keep full instance, including spatial)
            self.clip_projections = clip_projections
            if hasattr(clip_projections, 'spatial_alignment') and clip_projections.spatial_alignment:
                print("✓ Using full CLIP projections with spatial alignment from provided instance")
        
        # Efficient projection from joint space (256-dim) to spatial features
        # Use a simple learned projection: 256 -> output_channels
        # This is trainable and learns to convert CLIP embeddings to spatial features
        # Small enough to not collapse, but expressive enough to learn useful mappings
        # For spatial CLIP projections, we'll use 1x1 conv to preserve spatial structure
        # For global CLIP projections, we'll use Linear and broadcast
        self.spatial_proj = nn.Linear(256, self.output_channels)
        self.spatial_proj_conv = nn.Conv2d(256, self.output_channels, kernel_size=1)
        
        # Initialize to preserve information (Xavier/Glorot initialization)
        nn.init.xavier_uniform_(self.spatial_proj.weight)
        if self.spatial_proj.bias is not None:
            nn.init.zeros_(self.spatial_proj.bias)
        
        nn.init.xavier_uniform_(self.spatial_proj_conv.weight)
        if self.spatial_proj_conv.bias is not None:
            nn.init.zeros_(self.spatial_proj_conv.bias)
    
    def _project_embeddings(self, text_emb, pov_emb, batch_size, device):
        """Project embeddings through CLIP joint space."""
        # Check if CLIP projections have spatial alignment
        use_spatial_projections = (
            hasattr(self.clip_projections, 'spatial_alignment') and 
            self.clip_projections.spatial_alignment
        )
        
        if use_spatial_projections:
            # Use spatial projections - return spatial tensors directly
            H, W = self.spatial_size
            self.clip_projections._init_spatial_projections(H, W, device)
            
            text_proj = project_to_spatial(
                text_emb,
                self.clip_projections.spatial_text_proj,
                (H, W),
                batch_size=batch_size,
                device=device
            )
            
            pov_proj = project_to_spatial(
                pov_emb,
                self.clip_projections.spatial_pov_proj,
                (H, W),
                batch_size=batch_size,
                device=device
            )
            
            # Store flag for final projection
            self._use_spatial_mode = True
            return text_proj, pov_proj
        else:
            # Use global projections - return 1D embeddings
            if text_emb is not None:
                text_proj = self.clip_projections.text_proj(text_emb)
            else:
                text_proj = self.clip_projections.text_proj(None, batch_size=batch_size, device=device)
            
            if pov_emb is not None:
                pov_proj = self.clip_projections.pov_proj(pov_emb, batch_size=batch_size, device=device)
            else:
                pov_proj = self.clip_projections.pov_proj(None, batch_size=batch_size, device=device)
            
            self._use_spatial_mode = False
            return text_proj, pov_proj
    
    def _final_spatial_projection(self, combined, batch_size, device):
        """Project combined CLIP embeddings to final spatial features."""
        if self._use_spatial_mode:
            # Combined is already spatial [B, 256, H, W] - use conv
            combined = F.normalize(combined, p=2, dim=1)
            spatial_proj_dtype = get_dtype_from_module(self.spatial_proj_conv)
            combined = combined.to(dtype=spatial_proj_dtype)
            return self.spatial_proj_conv(combined)  # [B, output_channels, H, W]
        else:
            # Combined is 1D [B, 256] - use linear then broadcast
            combined = F.normalize(combined, p=2, dim=1)
            spatial_proj_dtype = get_dtype_from_module(self.spatial_proj)
            combined = combined.to(dtype=spatial_proj_dtype)
            spatial_flat = self.spatial_proj(combined)  # [B, output_channels]
            spatial = spatial_flat.unsqueeze(-1).unsqueeze(-1)  # [B, output_channels, 1, 1]
            spatial = F.interpolate(spatial, size=self.spatial_size, mode='bilinear', align_corners=False)
            return spatial
    
    def to_config(self):
        cfg = super().to_config()
        cfg.update({
            "output_channels": self.output_channels,
            "spatial_size": self.spatial_size,
            "combine_method": self.combine_method,
        })
        return cfg


class EmbeddingToSpatial(BaseEmbeddingToSpatial):
    """
    Projects 1D embeddings to spatial feature maps for cross-attention.
    
    Combines text and POV embeddings into a single spatial feature map
    that can be used in cross-attention layers.
    
    Config:
        text_dim: Dimension of text/graph embeddings (default: 384)
        pov_dim: Dimension of POV embeddings (default: 512)
        output_channels: Number of output channels for spatial features (default: matches UNet base_channels)
        spatial_size: Target spatial size (H, W) - will be interpolated to match UNet resolution (default: (64, 64))
        combine_method: How to combine text and POV embeddings: "add", "concat", "concat_proj" (default: "concat_proj")
    """
    
    def _build(self):
        super()._build()  # Initialize base class (output_channels, spatial_size, combine_method)
        text_dim = self._init_kwargs.get("text_dim", 384)
        pov_dim = self._init_kwargs.get("pov_dim", 512)
        
        self.text_dim = text_dim
        self.pov_dim = pov_dim
        
        if self.combine_method == "add":
            # Add embeddings (requires same dimension)
            if text_dim != pov_dim:
                # Project to common dimension
                common_dim = max(text_dim, pov_dim)
                self.text_proj = nn.Linear(text_dim, common_dim)
                self.pov_proj = nn.Linear(pov_dim, common_dim)
                combined_dim = common_dim
            else:
                self.text_proj = nn.Identity()
                self.pov_proj = nn.Identity()
                combined_dim = text_dim
        elif self.combine_method == "concat":
            # Concatenate embeddings
            combined_dim = text_dim + pov_dim
            self.text_proj = nn.Identity()
            self.pov_proj = nn.Identity()
        elif self.combine_method == "concat_proj":
            # Concatenate then project (most flexible)
            combined_dim = text_dim + pov_dim
            self.text_proj = nn.Identity()
            self.pov_proj = nn.Identity()
        else:
            raise ValueError(f"Unknown combine_method: {self.combine_method}")
        
        # Project combined embeddings to spatial features
        # Output: [B, output_channels, H, W]
        spatial_elements = self.output_channels * self.spatial_size[0] * self.spatial_size[1]
        
        # Safety check: prevent unreasonably large spatial projections
        MAX_SPATIAL_ELEMENTS = 10_000_000  # 10M elements max
        if spatial_elements > MAX_SPATIAL_ELEMENTS:
            raise ValueError(
                f"spatial_elements is too large: {spatial_elements} "
                f"(output_channels={self.output_channels}, spatial_size={self.spatial_size}). "
                f"This would create a Linear layer with {spatial_elements * 2} input features, "
                f"requiring ~{spatial_elements * 2 * spatial_elements * 4 / 1e9:.1f}GB of memory. "
                f"Check that spatial_size is set to latent dimensions (e.g., (64, 64)), not image dimensions."
            )
        
        self.spatial_proj = nn.Sequential(
            nn.Linear(combined_dim, spatial_elements * 2),  # Intermediate layer
            nn.SiLU(),
            nn.Linear(spatial_elements * 2, spatial_elements)
        )
        
    def _project_embeddings(self, text_emb, pov_emb, batch_size, device):
        """Project text and POV embeddings directly (no CLIP space)."""
        _, _, dtype = infer_batch_and_device(
            text_emb=text_emb,
            pov_emb=pov_emb,
            batch_size=batch_size,
            device=device
        )
        
        # Project embeddings (handle None inputs)
        if text_emb is not None:
            text_feat = self.text_proj(text_emb)
        else:
            zero_text = torch.zeros(batch_size, self.text_dim, device=device, dtype=dtype)
            text_feat = self.text_proj(zero_text)
        
        if pov_emb is not None:
            pov_feat = self.pov_proj(pov_emb)
        else:
            zero_pov = torch.zeros(batch_size, self.pov_dim, device=device, dtype=dtype)
            pov_feat = self.pov_proj(zero_pov)
        
        return text_feat, pov_feat
    
    def _final_spatial_projection(self, combined, batch_size, device):
        """Project combined embeddings directly to spatial features."""
        # Project to spatial features
        spatial_flat = self.spatial_proj(combined)  # [B, output_channels * H * W]
        spatial = spatial_flat.view(batch_size, self.output_channels, self.spatial_size[0], self.spatial_size[1])
        return spatial
    
    def to_config(self):
        cfg = super().to_config()
        cfg.update({
            "text_dim": self.text_dim,
            "pov_dim": self.pov_dim,
            "output_channels": self.output_channels,
            "spatial_size": self.spatial_size,
            "combine_method": self.combine_method,
        })
        return cfg


# Register all projection classes at module level
# This must be done after all classes are defined
PROJECTION_REGISTRY.update({
    "BaseProjection": BaseProjection,
    "TextProjection": TextProjection,
    "ImageProjection": ImageProjection,
    "LatentProjection": LatentProjection,
    "CLIPProjections": CLIPProjections,
    "CLIPEmbeddingToSpatial": CLIPEmbeddingToSpatial,
    "EmbeddingToSpatial": EmbeddingToSpatial,
})

