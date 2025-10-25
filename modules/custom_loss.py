# modules/custom_loss.py
import torch
import torch.nn.functional as F
import json


class VAELoss:
    """Base class for VAE losses."""
    
    def __call__(self, x, recon, mu, logvar):
        """
        Args:
            x: Input tensor
            recon: Reconstructed tensor
            mu: Latent mean
            logvar: Latent log variance
            
        Returns:
            Tuple of (total_loss, *individual_components, metrics_dict)
        """
        raise NotImplementedError

class StandardVAELoss(VAELoss):
    """Standard VAE loss with MSE reconstruction + KL divergence."""
    
    def __init__(self, kl_weight=1e-6):
        self.kl_weight = kl_weight
    
    def __call__(self, x, recon, mu, logvar):
        mse_loss = F.mse_loss(recon, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = mse_loss + self.kl_weight * kl_loss
        
        metrics = {
            "mse": mse_loss.item(),
            "kl": kl_loss.item(),
        }
        
        return total_loss, mse_loss, kl_loss, metrics

class SegmentationVAELoss(VAELoss):
    """VAE loss with segmentation preservation for RGB outputs."""
    
    def __init__(self, id_to_color, kl_weight=1e-6, lambda_seg=1.0, lambda_mse=1.0):
        """
        Args:
            id_to_color: Dict mapping class IDs (as strings) to RGB colors [R, G, B]
            kl_weight: Weight for KL divergence term
            lambda_seg: Weight for segmentation consistency term
            lambda_mse: Weight for RGB MSE term
        """
        self.id_to_color = id_to_color
        self.kl_weight = kl_weight
        self.lambda_seg = lambda_seg
        self.lambda_mse = lambda_mse
        self._class_colors = None
        self._color_to_id = None
        
        # Build inverse mapping and class colors tensor
        self._build_color_mappings()
    
    def _build_color_mappings(self):
        """Build color tensor and inverse mapping from id_to_color."""
        # Convert id_to_color to color_to_id
        self._color_to_id = {}
        colors = []
        ids = []
        
        for id_str, rgb in self.id_to_color.items():
            rgb_tuple = tuple(rgb)
            self._color_to_id[rgb_tuple] = int(id_str)
            colors.append(rgb)
            ids.append(int(id_str))
        
        print(f"[SegmentationVAELoss] Loaded {len(colors)} unique colors")
        print(f"[SegmentationVAELoss] ID range: {min(ids)} - {max(ids)}")
    
    def _get_class_colors(self, device):
        """Lazy initialization of class color tensor."""
        if self._class_colors is None or self._class_colors.device != device:
            # Create tensor with colors normalized to [0, 1]
            colors = []
            for id_str in sorted(self.id_to_color.keys(), key=lambda x: int(x)):
                rgb = self.id_to_color[id_str]
                colors.append([c / 255.0 for c in rgb])
            
            self._class_colors = torch.tensor(colors, dtype=torch.float32, device=device)
        
        return self._class_colors
    
    def _rgb_to_class_index(self, tensor_img):
        """Convert RGB to class indices (non-differentiable, for metrics only)."""
        if tensor_img.max() <= 1:
            tensor_img = (tensor_img * 255).byte()
        else:
            tensor_img = tensor_img.byte()
        
        B, _, H, W = tensor_img.shape
        class_map = torch.zeros((B, H, W), dtype=torch.long, device=tensor_img.device)
        
        for color, idx in self._color_to_id.items():
            mask = (tensor_img == torch.tensor(color, dtype=torch.uint8, device=tensor_img.device).view(1, 3, 1, 1)).all(dim=1)
            class_map[mask] = idx
        
        return class_map
    
    def __call__(self, x, recon, mu, logvar):
        # 1. RGB MSE loss
        mse_loss = F.mse_loss(recon, x)
        
        # 2. KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 3. Segmentation consistency (differentiable)
        class_colors = self._get_class_colors(x.device)  # (num_colors, 3)
        B, C, H, W = recon.shape
        
        # Reshape for distance computation
        recon_flat = recon.view(B, 3, H * W).permute(0, 2, 1)  # (B, H*W, 3)
        
        # Compute L2 distance from each reconstructed pixel to each class color
        # distances: (B, H*W, num_colors)
        distances = torch.cdist(recon_flat, class_colors.unsqueeze(0).expand(B, -1, -1))
        
        # Find nearest class color for each pixel
        nearest_idx = distances.argmin(dim=-1)  # (B, H*W)
        nearest_colors = class_colors[nearest_idx]  # (B, H*W, 3)
        nearest_colors = nearest_colors.permute(0, 2, 1).view(B, 3, H, W)  # (B, 3, H, W)
        
        # Segmentation loss: encourage recon to be close to nearest valid color
        # This is differentiable and provides gradient signal!
        seg_loss = F.mse_loss(recon, nearest_colors.detach())
        
        # 4. Total loss
        total_loss = self.lambda_mse * mse_loss + self.kl_weight * kl_loss + self.lambda_seg * seg_loss
        
        # 5. Compute accuracy for monitoring (non-differentiable, no gradients)
        with torch.no_grad():
            true_classes = self._rgb_to_class_index(x)
            pred_classes = self._rgb_to_class_index(torch.clamp(recon, 0, 1))
            seg_acc = (true_classes == pred_classes).float().mean().item()
        
        metrics = {
            "mse": mse_loss.item(),
            "kl": kl_loss.item(),
            "seg": seg_loss.item(),
            "acc": seg_acc,
        }
        
        return total_loss, mse_loss, kl_loss, seg_loss, metrics