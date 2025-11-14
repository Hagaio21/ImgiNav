"""
ControlNet Diffusion Model - A wrapper that combines ControlNet with the diffusion pipeline.

This model wraps:
- Decoder (from base diffusion model)
- Scheduler (from base diffusion model)
- ControlNet (frozen base UNet + trainable adapter)

Usage:
    # Load from a pretrained diffusion checkpoint
    base_model = DiffusionModel.load_checkpoint("path/to/diffusion.pt")
    
    # Create ControlNet wrapper
    model = ControlNetDiffusionModel.from_diffusion_model(
        base_model,
        adapter_config={
            "text_dim": 768,
            "pov_dim": 256,
            "base_channels": 32,
            "depth": 3
        }
    )
    
    # Training
    pred_noise = model(noisy_latents, t, text_emb=text_emb, pov_emb=pov_emb)
    
    # Sampling
    samples = model.sample(
        batch_size=4,
        text_emb=text_emb,
        pov_emb=pov_emb,
        num_steps=50
    )
"""

import torch
import torch.nn as nn
from pathlib import Path
import yaml

from models.components.base_model import BaseModel
from models.decoder import Decoder
from models.components.controlnet import ControlNet
from models.components.scheduler import SCHEDULER_REGISTRY


class ControlNetDiffusionModel(BaseModel):
    """
    Diffusion model with ControlNet for conditioned generation.
    
    This model wraps a pretrained diffusion model's decoder and scheduler,
    but replaces the UNet with a ControlNet that uses conditional inputs
    (text embeddings and POV embeddings).
    """
    
    def _build(self):
        # Get configs
        decoder_cfg = self._init_kwargs.get("decoder", None)
        sched_cfg = self._init_kwargs.get("scheduler", {})
        controlnet_cfg = self._init_kwargs.get("controlnet", {})
        
        # Build decoder
        if decoder_cfg:
            from models.autoencoder import Autoencoder
            decoder_checkpoint = decoder_cfg.get("checkpoint")
            if decoder_checkpoint:
                autoencoder = Autoencoder.load_checkpoint(decoder_checkpoint, map_location="cpu")
                self.decoder = autoencoder.decoder
            else:
                decoder_cfg_copy = decoder_cfg.copy()
                decoder_cfg_copy.pop("checkpoint", None)
                self.decoder = Decoder.from_config(decoder_cfg_copy)
        else:
            raise ValueError("ControlNetDiffusionModel requires 'decoder' config")
        
        # Freeze decoder if requested
        if decoder_cfg.get("frozen", False):
            self.decoder.freeze()
        
        # Build scheduler
        sched_type = sched_cfg.get("type", "CosineScheduler")
        if sched_type not in SCHEDULER_REGISTRY:
            raise ValueError(f"Unknown scheduler: {sched_type}")
        self.scheduler = SCHEDULER_REGISTRY[sched_type].from_config(sched_cfg)
        
        # Build ControlNet
        base_unet_cfg = controlnet_cfg.get("base_unet")
        adapter_cfg = controlnet_cfg.get("adapter")
        
        if base_unet_cfg is None or adapter_cfg is None:
            raise ValueError("ControlNetDiffusionModel requires 'controlnet.base_unet' and 'controlnet.adapter' configs")
        
        self.controlnet = ControlNet.from_config(controlnet_cfg)
        
        # ControlNet's base_unet should already be frozen, but ensure it
        for param in self.controlnet.base_unet.parameters():
            param.requires_grad = False
    
    @classmethod
    def from_diffusion_model(cls, diffusion_model, adapter_config, fuse_mode="add"):
        """
        Create ControlNetDiffusionModel from a pretrained DiffusionModel.
        
        Args:
            diffusion_model: Pretrained DiffusionModel instance
            adapter_config: Config for the ControlAdapter
                - text_dim: Text embedding dimension
                - pov_dim: POV embedding dimension
                - base_channels: Should match UNet base_channels
                - depth: Should match UNet depth
            fuse_mode: How to fuse control features ("add" or "concat")
        
        Returns:
            ControlNetDiffusionModel instance with weights copied from diffusion_model
        """
        # Get UNet config
        unet_config = diffusion_model.unet.to_config()
        
        # Build ControlNet config
        controlnet_config = {
            "base_unet": unet_config,
            "adapter": adapter_config,
            "fuse_mode": fuse_mode
        }
        
        # Create model
        model = cls(
            decoder=diffusion_model.decoder.to_config(),
            scheduler=diffusion_model.scheduler.to_config(),
            controlnet=controlnet_config
        )
        
        # Copy weights from trained UNet to ControlNet's base_unet
        model.controlnet.base_unet.load_state_dict(diffusion_model.unet.state_dict(), strict=False)
        
        # Ensure base_unet is frozen
        for param in model.controlnet.base_unet.parameters():
            param.requires_grad = False
        model.controlnet.base_unet.freeze_blocks(["downs", "ups", "bottleneck", "time_mlp", "final"])
        
        return model
    
    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x0_or_latents, t, text_emb=None, pov_emb=None, noise=None):
        """
        Forward diffusion training step with ControlNet.
        
        Args:
            x0_or_latents: Latents (should be pre-embedded)
            t: Timestep tensor
            text_emb: Text embeddings for conditioning [B, text_dim]
            pov_emb: POV embeddings for conditioning [B, pov_dim, H, W]
            noise: Optional noise tensor
        
        Returns:
            Dictionary with:
                - latent: Original clean latents
                - noisy_latent: Noisy latents
                - pred_noise: Predicted noise from ControlNet
                - noise: Noise that was used
        """
        # Input should already be latents (decoder-only mode)
        if isinstance(x0_or_latents, dict):
            if "latent" in x0_or_latents:
                latents = x0_or_latents["latent"]
            else:
                raise ValueError(f"Latent dict must contain 'latent'. Got: {list(x0_or_latents.keys())}")
        else:
            latents = x0_or_latents  # Direct tensor
        
        if noise is None:
            noise = self.scheduler.randn_like(latents)
        
        # Add noise using diffusion schedule
        result = self.scheduler.add_noise(latents, noise, t, return_scaled_noise=True)
        noisy_latents, noise_used = result
        
        # Predict noise using ControlNet (requires text_emb and pov_emb)
        if text_emb is None or pov_emb is None:
            raise ValueError("ControlNet requires text_emb and pov_emb for forward pass")
        
        pred_noise = self.controlnet(noisy_latents, t, text_emb, pov_emb)
        
        return {
            "latent": latents,
            "noisy_latent": noisy_latents,
            "pred_noise": pred_noise,
            "noise": noise_used,
        }
    
    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample(self, batch_size=1, latent_shape=None, text_emb=None, pov_emb=None,
               num_steps=50, method="ddim", eta=0.0, device=None, return_history=False, verbose=False):
        """
        Sample images using ControlNet with conditioning.
        
        Args:
            batch_size: Number of samples to generate
            latent_shape: Latent shape (C, H, W). Inferred from decoder if None.
            text_emb: Text embeddings [B, text_dim]
            pov_emb: POV embeddings [B, pov_dim, H, W]
            num_steps: Number of sampling steps
            method: "ddim" or "ddpm"
            eta: DDIM eta parameter (0 = deterministic)
            device: Device to use
            return_history: If True, return sampling history
            verbose: If True, print progress
        
        Returns:
            Dictionary with:
                - latent: Generated latents
                - rgb: Decoded RGB images (if decoder has RGB head)
                - history: Sampling history (if return_history=True)
        """
        if device is None:
            device = next(self.parameters()).device
        
        if text_emb is None or pov_emb is None:
            raise ValueError("ControlNet sampling requires text_emb and pov_emb")
        
        # Ensure batch sizes match
        if text_emb.shape[0] != batch_size:
            if text_emb.shape[0] == 1:
                text_emb = text_emb.expand(batch_size, -1)
            else:
                raise ValueError(f"text_emb batch size ({text_emb.shape[0]}) doesn't match batch_size ({batch_size})")
        
        if pov_emb.shape[0] != batch_size:
            if pov_emb.shape[0] == 1:
                pov_emb = pov_emb.expand(batch_size, -1, -1, -1)
            else:
                raise ValueError(f"pov_emb batch size ({pov_emb.shape[0]}) doesn't match batch_size ({batch_size})")
        
        # Infer latent shape from decoder config if not provided
        if latent_shape is None:
            latent_ch = self.decoder._init_kwargs.get('latent_channels', 4)
            up_steps = self.decoder._init_kwargs.get('upsampling_steps', 4)
            spatial_res = 512 // (2 ** up_steps)
            latent_shape = (latent_ch, spatial_res, spatial_res)
        
        self.scheduler = self.scheduler.to(device)
        # Start with standard normal noise N(0,1)
        dummy = torch.zeros((batch_size, *latent_shape), device=device)
        latents = self.scheduler.randn_like(dummy)
        
        # Create timestep schedule
        # For both DDIM and DDPM, we go from high noise (T-1) to low noise (0)
        if method == "ddim":
            step_size = self.scheduler.num_steps // num_steps
            # Create evenly spaced timesteps from T-1 down to 0
            timesteps = torch.arange(self.scheduler.num_steps - 1, -1, -step_size, device=device).long()
            # Ensure we always include t=0 as the final step
            if timesteps[-1] != 0:
                timesteps = torch.cat([timesteps, torch.tensor([0], device=device)])
        else:
            timesteps = torch.arange(self.scheduler.num_steps - 1, -1, -1, device=device).long()
        
        history = [] if return_history else None
        
        # Sampling loop
        for i, t in enumerate(timesteps):
            if verbose and (i % max(1, len(timesteps) // 10) == 0 or i == len(timesteps) - 1):
                print(f"  Sampling step {i+1}/{len(timesteps)} (t={t.item()})")
            
            t_batch = t.expand(batch_size)
            
            with torch.no_grad():
                self.controlnet.eval()
                pred_noise = self.controlnet(latents, t_batch, text_emb, pov_emb)
            
            # DDIM step
            if method == "ddim":
                alpha_bars = self.scheduler.alpha_bars.to(device)
                
                alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
                
                # Get previous timestep (next in sequence since we're going backwards)
                if i < len(timesteps) - 1:
                    t_prev = timesteps[i + 1]
                    alpha_bar_prev = alpha_bars[t_prev].view(-1, 1, 1, 1)
                else:
                    # Last step: use alpha_bar_0 = 1.0 (fully denoised)
                    alpha_bar_prev = torch.tensor(1.0, device=device, dtype=alpha_bar_t.dtype).view(-1, 1, 1, 1)
                
                # Predict x0 from current noisy latents
                pred_x0 = (latents - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt().clamp(min=1e-8)
                pred_x0 = torch.clamp(pred_x0, -10.0, 10.0)
                
                # DDIM update (deterministic if eta=0)
                # Standard DDIM formula: x_{t-1} = sqrt(alpha_bar_{t-1}) * pred_x0 + sqrt(1 - alpha_bar_{t-1} - sigma_t^2) * epsilon + sigma_t * z
                if eta > 0 and i < len(timesteps) - 1:
                    # Add stochastic noise if eta > 0
                    sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t).clamp(min=1e-8) * (1 - alpha_bar_t / alpha_bar_prev.clamp(min=1e-8))).sqrt()
                    sigma = torch.clamp(sigma, min=0.0, max=1.0)
                    pred_dir = (1 - alpha_bar_prev - sigma**2).sqrt().clamp(min=0.0) * pred_noise
                    noise_step = sigma * self.scheduler.randn_like(latents)
                else:
                    # Deterministic DDIM (eta=0): x_{t-1} = sqrt(alpha_bar_{t-1}) * pred_x0 + sqrt(1 - alpha_bar_{t-1}) * epsilon
                    pred_dir = (1 - alpha_bar_prev).sqrt() * pred_noise
                    noise_step = 0
                
                latents = alpha_bar_prev.sqrt() * pred_x0 + pred_dir + noise_step
            
            else:
                # DDPM step
                alpha_bars = self.scheduler.alpha_bars.to(device)
                alphas = self.scheduler.alphas.to(device)
                betas = self.scheduler.betas.to(device)
                
                alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
                beta_t = betas[t].view(-1, 1, 1, 1)
                
                if i < len(timesteps) - 1:
                    t_prev = timesteps[i+1]
                    alpha_bar_prev = alpha_bars[t_prev].view(-1, 1, 1, 1)
                    
                    # Predict x0
                    pred_x0 = (latents - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()
                    pred_x0 = torch.clamp(pred_x0, -10.0, 10.0)
                    
                    # Sample from posterior
                    posterior_mean = (alpha_bar_prev.sqrt() * beta_t / (1 - alpha_bar_t)) * pred_x0 + \
                                    (alphas[t_prev].sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar_t)) * latents
                    posterior_var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                    noise_sample = self.scheduler.randn_like(latents)
                    latents = posterior_mean + posterior_var.sqrt() * noise_sample
                else:
                    # Last step: predict x0
                    pred_x0 = (latents - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()
                    latents = pred_x0
            
            if return_history:
                history.append(latents.clone())
        
        # Build output dict
        result = {"latent": latents}
        
        # Decode to RGB if decoder is available
        with torch.no_grad():
            decoded_out = self.decoder({"latent": latents})
            if "rgb" in decoded_out:
                rgb = decoded_out["rgb"]
                # RGBHead uses tanh activation by default, outputting in [-1, 1] range
                # Denormalize from [-1, 1] to [0, 1] for image saving
                # This matches how images are saved during autoencoder training (train.py line 186-187)
                rgb = (rgb + 1.0) / 2.0
                # Clamp to ensure valid [0, 1] range (some values might be slightly outside due to numerical precision)
                rgb = torch.clamp(rgb, 0.0, 1.0)
                result["rgb"] = rgb
        
        if return_history:
            result["history"] = history
        
        return result
    
    # ------------------------------------------------------------------
    # Config I/O
    # ------------------------------------------------------------------
    def to_config(self):
        cfg = {
            "type": "ControlNetDiffusionModel",
            "decoder": self.decoder.to_config(),
            "scheduler": self.scheduler.to_config(),
            "controlnet": self.controlnet.to_config(),
        }
        return cfg
    
    @classmethod
    def load_checkpoint(cls, path, map_location="cpu", return_extra=False, config=None):
        """
        Load ControlNetDiffusionModel checkpoint.
        
        Args:
            path: Path to checkpoint
            map_location: Device to load on
            return_extra: If True, return tuple (model, extra_state_dict)
            config: Optional config dict to use instead of saved config
        
        Returns:
            If return_extra=False: just the model
            If return_extra=True: (model, extra_state_dict) tuple
        """
        path = Path(path)
        payload = torch.load(path, map_location=map_location)
        
        # Use provided config if available, otherwise use saved config
        model_config = config if config is not None else payload.get("config")
        model = cls.from_config(model_config) if model_config else cls()
        
        # Load state dict
        model.load_state_dict(payload["state_dict"], strict=False)
        
        if return_extra:
            extra_state = {k: v for k, v in payload.items() 
                          if k not in ["state_dict", "config"]}
            return model, extra_state
        
        return model
    
    def save_checkpoint(self, path, include_config=True, **extra_state):
        """
        Save ControlNetDiffusionModel checkpoint.
        
        Args:
            path: Path to save checkpoint
            include_config: If True, include model config in checkpoint
            **extra_state: Additional state to save (optimizer, epoch, etc.)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        payload = {
            "state_dict": self.state_dict(),
        }
        
        if include_config:
            payload["config"] = self.to_config()
        
        payload.update(extra_state)
        torch.save(payload, path)

