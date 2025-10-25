import torch
import torch.nn.functional as F
import json


class VAELoss:
    """Base class for VAE losses."""
    
    def __call__(self, x, recon, mu, logvar):
        raise NotImplementedError


class StandardVAELoss(VAELoss):
    """Standard VAE loss with MSE reconstruction + KL divergence."""
    
    def __init__(self, kl_weight=1e-6):
        self.kl_weight = kl_weight
    
    def __call__(self, x, outputs): # <-- CHANGE
        # Unpack the outputs dictionary
        recon = outputs.get("recon")
        mu = outputs.get("mu")
        logvar = outputs.get("logvar")
        
        if recon is None or mu is None or logvar is None:
            raise ValueError("StandardVAELoss requires 'recon', 'mu', and 'logvar' in outputs")

        mse_loss = F.mse_loss(recon, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = mse_loss + self.kl_weight * kl_loss
        metrics = {"mse": mse_loss.item(), "kl": kl_loss.item()}
        return total_loss, mse_loss, kl_loss, metrics


class SegmentationVAELoss(VAELoss):
    """VAE loss with segmentation preservation for RGB outputs."""
    
    def __init__(self, id_to_color, kl_weight=1e-6, lambda_seg=1.0, lambda_mse=1.0):
        self.id_to_color = id_to_color
        self.kl_weight = kl_weight
        self.lambda_seg = lambda_seg
        self.lambda_mse = lambda_mse
        self._class_colors = None
        self._color_to_id = None
        self._build_color_mappings()
    
    # -------------------- internal utilities --------------------
    def _build_color_mappings(self):
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
        if self._class_colors is None or self._class_colors.device != device:
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

    # -------------------- NEW: layout → logits tensor --------------------
    def layout_to_logits(self, layout_rgb):
        """
        Convert an RGB layout tensor to per-class logits (one-hot encoded).

        Args:
            layout_rgb: (B,3,H,W) tensor in [0,1] or [0,255]
        Returns:
            logits: (B,num_classes,H,W) float tensor with 1 at the matching class color
        """
        if layout_rgb.max() <= 1:
            layout_rgb = (layout_rgb * 255).byte()
        else:
            layout_rgb = layout_rgb.byte()
        B, _, H, W = layout_rgb.shape

        class_colors = self._get_class_colors(layout_rgb.device)  # (num_classes,3)
        num_classes = class_colors.shape[0]

        logits = torch.zeros((B, num_classes, H, W), device=layout_rgb.device, dtype=torch.float32)

        for class_idx, color in enumerate((class_colors * 255).byte()):
            mask = (layout_rgb == color.view(1, 3, 1, 1)).all(dim=1)
            logits[:, class_idx, :, :] = mask.float()

        return logits

    # -------------------- main loss --------------------
    def __call__(self, x, outputs):
        """
        Compute the total VAE loss for dual-head (RGB + segmentation) decoder.

        Args:
            x: input RGB layout tensor (B,3,H,W)
            outputs: dict containing {
                "recon": (B,3,H,W) reconstructed RGB,
                "seg_logits": (B,num_classes,H,W) predicted segmentation logits (optional),
                "mu": latent mean,
                "logvar": latent log variance
            }
        """
        recon = outputs.get("recon")
        seg_logits = outputs.get("seg_logits", None)
        mu = outputs["mu"]
        logvar = outputs["logvar"]

        # 1. Reconstruction loss
        mse_loss = F.mse_loss(recon, x)

        # 2. KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # 3. Segmentation consistency loss
        if seg_logits is not None:
            # Compute ground-truth one-hot tensor from input layout
            target_logits = self.layout_to_logits(x)
            seg_loss = F.binary_cross_entropy_with_logits(seg_logits, target_logits)
        else:
            # Fallback differentiable color alignment
            class_colors = self._get_class_colors(x.device)
            B, C, H, W = recon.shape
            recon_flat = recon.view(B, 3, H * W).permute(0, 2, 1)
            distances = torch.cdist(recon_flat, class_colors.unsqueeze(0).expand(B, -1, -1))
            nearest_idx = distances.argmin(dim=-1)
            nearest_colors = class_colors[nearest_idx]
            nearest_colors = nearest_colors.permute(0, 2, 1).view(B, 3, H, W)
            seg_loss = F.mse_loss(recon, nearest_colors.detach())

        # 4. Total loss
        total_loss = (
            self.lambda_mse * mse_loss +
            self.lambda_seg * seg_loss +
            self.kl_weight * kl_loss
        )

        # 5. Metrics for logging
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
