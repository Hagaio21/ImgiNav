"""
Additional loss functions for sharper reconstructions.

Includes:
- L1Loss: Less blurry than MSE
- PerceptualLoss: VGG-based perceptual loss for texture/edge preservation
- LPIPSLoss: Learned perceptual loss (requires lpips package)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_loss import LossComponent, register_loss


@register_loss
class L1Loss(LossComponent):
    """
    L1 (Mean Absolute Error) loss.
    
    Less blurry than MSE because it doesn't over-penalize outliers,
    so the model doesn't converge to the mean as strongly.
    
    Config:
        key: Key in preds for predictions (default: "rgb")
        target_key: Key in targets for ground truth (default: "rgb")
        weight: Loss weight (default: 1.0)
    """
    def _build(self):
        super()._build()
        self.criterion = nn.L1Loss()

    def forward(self, preds, targets):
        if self.key not in preds or self.target_key not in targets:
            device = next(iter(preds.values())).device if preds else torch.device("cpu")
            return torch.tensor(0.0, device=device, requires_grad=True), {}
        
        loss = self.criterion(preds[self.key], targets[self.target_key]) * self.weight
        return loss, {f"L1_{self.key}": loss.detach()}


@register_loss
class PerceptualLoss(LossComponent):
    """
    VGG-based perceptual loss for sharper reconstructions.
    
    Compares features from a pretrained VGG network rather than raw pixels.
    This encourages preservation of edges, textures, and high-level structure.
    
    Config:
        key: Key in preds for predictions (default: "rgb")
        target_key: Key in targets for ground truth (default: "rgb")
        weight: Loss weight (default: 0.1)
        layers: List of VGG layers to use (default: ["conv1_2", "conv2_2", "conv3_3", "conv4_3"])
        normalize_input: Whether to normalize input to ImageNet stats (default: True)
        layer_weights: Dict of layer -> weight, or None for equal weights
    """
    
    # VGG layer name to index mapping
    LAYER_MAP = {
        "conv1_1": 0, "relu1_1": 1, "conv1_2": 2, "relu1_2": 3, "pool1": 4,
        "conv2_1": 5, "relu2_1": 6, "conv2_2": 7, "relu2_2": 8, "pool2": 9,
        "conv3_1": 10, "relu3_1": 11, "conv3_2": 12, "relu3_2": 13, 
        "conv3_3": 14, "relu3_3": 15, "conv3_4": 16, "relu3_4": 17, "pool3": 18,
        "conv4_1": 19, "relu4_1": 20, "conv4_2": 21, "relu4_2": 22,
        "conv4_3": 23, "relu4_3": 24, "conv4_4": 25, "relu4_4": 26, "pool4": 27,
        "conv5_1": 28, "relu5_1": 29, "conv5_2": 30, "relu5_2": 31,
        "conv5_3": 32, "relu5_3": 33, "conv5_4": 34, "relu5_4": 35, "pool5": 36,
    }
    
    def _build(self):
        super()._build()
        
        self.layers = self._init_kwargs.get("layers", ["conv1_2", "conv2_2", "conv3_3", "conv4_3"])
        self.normalize_input = self._init_kwargs.get("normalize_input", True)
        self.layer_weights = self._init_kwargs.get("layer_weights", None)
        
        # Load pretrained VGG19
        try:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        except ImportError:
            # Fallback for older torchvision
            from torchvision.models import vgg19
            vgg = vgg19(pretrained=True).features
        
        # Freeze VGG weights
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Get max layer index needed
        max_idx = max(self.LAYER_MAP.get(layer, 0) for layer in self.layers)
        
        # Only keep layers up to max needed
        self.vgg = nn.Sequential(*list(vgg.children())[:max_idx + 1])
        self.vgg.eval()
        
        # Store layer indices for extraction
        self.layer_indices = [self.LAYER_MAP[layer] for layer in self.layers]
        
        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _normalize(self, x):
        """Normalize from [-1, 1] (tanh output) to ImageNet stats."""
        # First convert from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        # Then normalize to ImageNet stats
        return (x - self.mean) / self.std
    
    def _extract_features(self, x):
        """Extract features from specified VGG layers."""
        if self.normalize_input:
            x = self._normalize(x)
        
        features = []
        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            if idx in self.layer_indices:
                features.append(x)
        
        return features
    
    def forward(self, preds, targets):
        if self.key not in preds or self.target_key not in targets:
            device = next(iter(preds.values())).device if preds else torch.device("cpu")
            return torch.tensor(0.0, device=device, requires_grad=True), {}
        
        pred = preds[self.key]
        target = targets[self.target_key]
        
        # Ensure VGG is on same device
        if self.mean.device != pred.device:
            self.to(pred.device)
        
        # Extract features
        with torch.no_grad():
            target_features = self._extract_features(target)
        pred_features = self._extract_features(pred)
        
        # Compute loss for each layer
        total_loss = 0.0
        logs = {}
        
        for i, (pred_feat, target_feat) in enumerate(zip(pred_features, target_features)):
            layer_name = self.layers[i]
            
            # Get layer weight
            if self.layer_weights and layer_name in self.layer_weights:
                layer_weight = self.layer_weights[layer_name]
            else:
                layer_weight = 1.0 / len(self.layers)
            
            # L1 loss on features
            layer_loss = F.l1_loss(pred_feat, target_feat.detach())
            total_loss = total_loss + layer_loss * layer_weight
            logs[f"perceptual_{layer_name}"] = layer_loss.detach()
        
        total_loss = total_loss * self.weight
        logs["perceptual_total"] = total_loss.detach()
        
        return total_loss, logs


@register_loss  
class SSIMLoss(LossComponent):
    """
    Structural Similarity Index (SSIM) loss.
    
    SSIM measures structural similarity rather than pixel-wise difference,
    which often correlates better with human perception.
    
    Config:
        key: Key in preds for predictions (default: "rgb")
        target_key: Key in targets for ground truth (default: "rgb")
        weight: Loss weight (default: 1.0)
        window_size: Size of SSIM window (default: 11)
        sigma: Gaussian sigma for window (default: 1.5)
    """
    def _build(self):
        super()._build()
        self.window_size = self._init_kwargs.get("window_size", 11)
        self.sigma = self._init_kwargs.get("sigma", 1.5)
        self.channel = 3  # Will be updated on first forward
        
        # Create Gaussian window
        self.register_buffer("window", self._create_window(self.window_size, self.channel))
    
    def _gaussian(self, window_size, sigma):
        """Create 1D Gaussian kernel."""
        gauss = torch.tensor([
            torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def _create_window(self, window_size, channel):
        """Create 2D Gaussian window."""
        _1D_window = self._gaussian(window_size, self.sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _ssim(self, img1, img2):
        """Compute SSIM between two images."""
        channel = img1.size(1)
        
        # Update window if channel count changed
        if channel != self.channel:
            self.channel = channel
            self.window = self._create_window(self.window_size, channel).to(img1.device)
        
        window = self.window.to(img1.device)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()
    
    def forward(self, preds, targets):
        if self.key not in preds or self.target_key not in targets:
            device = next(iter(preds.values())).device if preds else torch.device("cpu")
            return torch.tensor(0.0, device=device, requires_grad=True), {}
        
        pred = preds[self.key]
        target = targets[self.target_key]
        
        # SSIM is a similarity (higher = better), convert to loss (lower = better)
        ssim_val = self._ssim(pred, target)
        loss = (1 - ssim_val) * self.weight
        
        return loss, {
            f"SSIM_{self.key}": ssim_val.detach(),
            f"SSIM_loss_{self.key}": loss.detach(),
        }


@register_loss
class CombinedReconstructionLoss(LossComponent):
    """
    Combined reconstruction loss: L1 + Perceptual + SSIM.
    
    A convenience wrapper that combines multiple reconstruction losses
    with sensible defaults for sharp reconstructions.
    
    Config:
        key: Key in preds for predictions (default: "rgb")
        target_key: Key in targets for ground truth (default: "rgb")
        weight: Overall loss weight (default: 1.0)
        l1_weight: Weight for L1 loss (default: 1.0)
        perceptual_weight: Weight for perceptual loss (default: 0.1)
        ssim_weight: Weight for SSIM loss (default: 0.1)
        use_perceptual: Whether to include perceptual loss (default: True)
        use_ssim: Whether to include SSIM loss (default: True)
    """
    def _build(self):
        super()._build()
        
        self.l1_weight = self._init_kwargs.get("l1_weight", 1.0)
        self.perceptual_weight = self._init_kwargs.get("perceptual_weight", 0.1)
        self.ssim_weight = self._init_kwargs.get("ssim_weight", 0.1)
        self.use_perceptual = self._init_kwargs.get("use_perceptual", True)
        self.use_ssim = self._init_kwargs.get("use_ssim", True)
        
        # Create sub-losses
        self.l1_loss = L1Loss(key=self.key, target_key=self.target_key, weight=self.l1_weight)
        
        if self.use_perceptual:
            self.perceptual_loss = PerceptualLoss(
                key=self.key, 
                target_key=self.target_key, 
                weight=self.perceptual_weight
            )
        
        if self.use_ssim:
            self.ssim_loss = SSIMLoss(
                key=self.key,
                target_key=self.target_key,
                weight=self.ssim_weight
            )
    
    def forward(self, preds, targets):
        total_loss = torch.tensor(0.0, device=preds[self.key].device, requires_grad=True)
        logs = {}
        
        # L1 loss
        l1, l1_logs = self.l1_loss(preds, targets)
        total_loss = total_loss + l1
        logs.update(l1_logs)
        
        # Perceptual loss
        if self.use_perceptual:
            perc, perc_logs = self.perceptual_loss(preds, targets)
            total_loss = total_loss + perc
            logs.update(perc_logs)
        
        # SSIM loss
        if self.use_ssim:
            ssim, ssim_logs = self.ssim_loss(preds, targets)
            total_loss = total_loss + ssim
            logs.update(ssim_logs)
        
        total_loss = total_loss * self.weight
        logs["combined_reconstruction"] = total_loss.detach()
        
        return total_loss, logs
