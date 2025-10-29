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


class DiffusionLoss:
    """Standard diffusion loss (unconditional)."""

    def __call__(self, pred_noise, true_noise):
        """
        pred_noise: predicted noise (B,C,H,W)
        true_noise: ground-truth noise (B,C,H,W)
        """
        mse_loss = F.mse_loss(pred_noise, true_noise)
        metrics = {"mse": mse_loss.item()}
        return mse_loss, metrics


class CorrLoss:
    """Correlation alignment loss between two tensors."""

    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, x, y):
        """
        Compute 1 - Pearson correlation over batch.
        x, y: tensors of shape (B, ...) with same batch dimension.
        """
        B = x.shape[0]
        x_f = x.view(B, -1)
        y_f = y.view(B, -1)

        # zero-mean normalize
        x_f = x_f - x_f.mean(dim=1, keepdim=True)
        y_f = y_f - y_f.mean(dim=1, keepdim=True)

        num = (x_f * y_f).sum(dim=1)
        denom = torch.sqrt((x_f.pow(2).sum(dim=1) + self.eps) *
                           (y_f.pow(2).sum(dim=1) + self.eps))
        corr = num / denom

        # minimize (1 - correlation)
        return 1 - corr.mean()

class VGGPerceptualLoss(nn.Module):
    """
    VGG16-based perceptual loss.
    Assumes inputs are RGB tensors in [0, 1] range.
    """
    def __init__(self, layer_indices=None, resize=True):
        super().__init__()
        # Load VGG16 features
        vgg = models.vgg16(pretrained=True).features.eval()
        self.resize = resize
        
        # Use standard conv layers before max-pooling
        if layer_indices is None:
            # Corresponds to conv1_2, conv2_2, conv3_3, conv4_3
            self.layer_indices = [3, 8, 15, 22] 
        else:
            self.layer_indices = layer_indices
            
        # Create a module list of VGG features
        self.features = nn.ModuleList([vgg[i] for i in range(max(self.layer_indices) + 1)])
        
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
        # VGG normalization
        # VGG was trained on ImageNet, which has a specific mean and std
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        """Normalize input for VGG."""
        return (x - self.mean) / self.std

    def forward(self, x, y):
        """
        x, y: Input tensors (B, 3, H, W) in [0, 1] range.
        """
        if self.resize:
            # VGG expects 224x224
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)

        x = self.normalize(x)
        y = self.normalize(y)

        loss = 0.0
        for i, layer in enumerate(self.features):
            x = layer(x)
            y = layer(y)
            # Add L1 loss at specified layers
            if i in self.layer_indices:
                loss += F.l1_loss(x, y)
        
        return loss




