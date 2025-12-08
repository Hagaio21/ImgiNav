import torch
import torch.nn as nn
from pathlib import Path
import yaml
import gzip
import pickle

from models.components.base_model import BaseModel
from models.components.registry import create_component, COMPONENT_REGISTRY


class DiffusionModel(BaseModel):
    """Minimal end-to-end diffusion model for training."""

    def _load_model_from_checkpoint(self, checkpoint_path):
        """
        Load any BaseModel from checkpoint by inspecting its config.
        
        Returns:
            BaseModel instance with decoder attribute
        """
        path = Path(checkpoint_path)
        # Try to detect if file is compressed
        try:
            with gzip.open(path, 'rb') as f:
                payload = pickle.load(f)
        except (gzip.BadGzipFile, OSError):
            payload = torch.load(path, map_location="cpu")
        
        config = payload.get("config")
        if not config:
            raise ValueError(f"Checkpoint {checkpoint_path} has no config")
        
        # Get model type from config
        model_type = config.get("type")
        if not model_type:
            raise ValueError(f"Checkpoint config has no 'type' field")
        
        # Use component registry to get the model class
        if model_type not in COMPONENT_REGISTRY:
            raise ValueError(f"Unknown model type '{model_type}' in checkpoint. Available: {list(COMPONENT_REGISTRY.keys())}")
        
        model_cls = COMPONENT_REGISTRY[model_type]
        if not issubclass(model_cls, BaseModel):
            raise ValueError(f"Type '{model_type}' is not a BaseModel subclass")
        
        # Create model from config
        model = model_cls.from_config(config)
        
        # Load state dict
        state_dict = payload.get("state_dict", payload)
        model.load_state_dict(state_dict, strict=False)
        
        return model
    
    def _setup_decoder(self, decoder=None, decoder_cfg=None, autoencoder_cfg=None):
        """Setup decoder from various sources."""
        if decoder is not None and not isinstance(decoder, dict):
            # Dependency injection: use provided decoder instance
            self.add_component("decoder", decoder)
            frozen = self._init_kwargs.get("frozen", False)
        elif autoencoder_cfg is not None:
            # Load from autoencoder config
            ae_checkpoint = autoencoder_cfg.get("checkpoint")
            if ae_checkpoint:
                # Load model from checkpoint and extract decoder
                model = self._load_model_from_checkpoint(ae_checkpoint)
                if not hasattr(model, 'decoder'):
                    raise ValueError(f"Model loaded from {ae_checkpoint} has no 'decoder' attribute")
                self.add_component("decoder", model.decoder)
            else:
                # Build decoder from autoencoder config
                decoder_subcfg = autoencoder_cfg.get("decoder")
                if decoder_subcfg:
                    decoder = self.create_component_from_config(decoder_subcfg, default_type="Decoder")
                    self.add_component("decoder", decoder)
                else:
                    raise ValueError("Cannot build decoder: no checkpoint and no decoder config in autoencoder config")
            frozen = autoencoder_cfg.get("frozen", False)
        elif decoder_cfg is not None:
            # Load from decoder config
            decoder_checkpoint = decoder_cfg.get("checkpoint")
            if decoder_checkpoint:
                # Load model from checkpoint and extract decoder
                model = self._load_model_from_checkpoint(decoder_checkpoint)
                if not hasattr(model, 'decoder'):
                    raise ValueError(f"Model loaded from {decoder_checkpoint} has no 'decoder' attribute")
                self.add_component("decoder", model.decoder)
            else:
                # Build from decoder config
                decoder = self.create_component_from_config(decoder_cfg, default_type="Decoder")
                self.add_component("decoder", decoder)
            frozen = decoder_cfg.get("frozen", False)
        else:
            raise ValueError("DiffusionModel requires decoder (object, decoder config, or autoencoder config)")
        
        if frozen:
            self.decoder.freeze()

    def _build(self):
        # Setup decoder from various sources
        decoder = self._init_kwargs.get("decoder", None)
        decoder_cfg = self._init_kwargs.get("decoder", None)
        autoencoder_cfg = self._init_kwargs.get("autoencoder", None)
        
        self._setup_decoder(
            decoder=decoder if not isinstance(decoder, dict) else None,
            decoder_cfg=decoder_cfg if isinstance(decoder_cfg, dict) else None,
            autoencoder_cfg=autoencoder_cfg
        )
        
        unet_cfg = self._init_kwargs.get("unet", {})
        sched_cfg = self._init_kwargs.get("scheduler", {})

        # Build embedding projection
        embedding_proj_cfg = self._init_kwargs.get("embedding_projection", None)
        conditioning_channels = None
        
        if embedding_proj_cfg:
            # Set default output_channels from UNet config if not specified
            if "output_channels" not in embedding_proj_cfg:
                embedding_proj_cfg["output_channels"] = unet_cfg.get("base_channels", 96)
            conditioning_channels = embedding_proj_cfg.get("output_channels")
            
            # Create embedding projection using unified registry
            # create_component handles default type and all initialization
            embedding_proj = self.create_component("embedding_projection", default_type="EmbeddingToSpatial")
            if embedding_proj is not None:
                self.add_component("embedding_projection", embedding_proj)
        
        # Build UNet
        unet_type = unet_cfg.get("type", "").lower()
        if conditioning_channels is not None and unet_cfg.get("enable_cross_attention", False):
            unet_cfg = unet_cfg.copy()
            unet_cfg["conditioning_channels"] = conditioning_channels
        
        # Unified UnetWithAttention class handles both "unet" and "unetwithattention" types
        # If type is "unet" (old name), it's mapped to UnetWithAttention with use_attention=False
        # If type is "unetwithattention" or "unet_with_attention", ensure use_attention is set
        if unet_type in ("unetwithattention", "unet_with_attention"):
            unet_cfg = unet_cfg.copy()
            if "use_attention" not in unet_cfg:
                unet_cfg["use_attention"] = True
        elif unet_type == "unet":
            # Old "Unet" type - map to UnetWithAttention with use_attention=False
            unet_cfg = unet_cfg.copy()
            if "use_attention" not in unet_cfg:
                unet_cfg["use_attention"] = False
        
        # Use unified component registry for UNet
        if "type" not in unet_cfg:
            unet_cfg["type"] = "UnetWithAttention"
        self.add_component("unet", create_component(unet_cfg))

        # Build scheduler using unified component registry
        if "type" not in sched_cfg:
            sched_cfg["type"] = "CosineScheduler"
        self.add_component("scheduler", create_component(sched_cfg))
        
        self.scale_factor = self._init_kwargs.get("scale_factor", 1.0)
        self._latent_clamp_min = self._init_kwargs.get("latent_clamp_min", -6.0)
        self._latent_clamp_max = self._init_kwargs.get("latent_clamp_max", 6.0)
        
        # Write model statistics if save_path is available
        if hasattr(self, 'save_path') and self.save_path:
            self._write_model_statistics()

    
    def _get_component_statistics(self):
        """Get parameter statistics for all tracked components."""
        stats = {}
        for component, name in self._component_names.items():
            if component is not None:
                component_stats = self._get_module_statistics(component, label=name.capitalize())
                if component_stats:
                    # Add "(frozen)" suffix if all parameters are frozen
                    if component_stats["trainable"] == 0:
                        component_stats["label"] = f"{component_stats['label']} (frozen)"
                    stats[name] = component_stats
        return stats

    def forward(self, latents, t, noise=None, text_emb=None, pov_emb=None):
        """
        Forward pass of diffusion model.
        
        Args:
            latents: Latent tensor [B, C, H, W]
            t: Timestep tensor [B]
            noise: Optional noise tensor (generated if None)
            text_emb: Optional text embeddings (may have per-sample zeros from CFG dropout)
            pov_emb: Optional POV embeddings (may have per-sample zeros from CFG dropout)
        
        Returns:
            Dict with prediction outputs
        """
        # Prepare conditioning signal
        embedding_proj = getattr(self, 'embedding_projection', None)
        
        # Pass embeddings to projection if available
        # Per-sample CFG dropout zeros individual samples within the tensor
        # The projection and cross-attention handle zeroed samples naturally:
        # - Zero embeddings -> near-zero conditioning -> effectively unconditional
        if embedding_proj is not None and (text_emb is not None or pov_emb is not None):
            conditioning_signal = embedding_proj(text_emb, pov_emb)
        else:
            conditioning_signal = None
        
        # Prepare noise
        if noise is None or noise.shape != latents.shape:
            noise = self.scheduler.randn_like(latents)
        
        # Apply scale factor
        if self.scale_factor != 1.0:
            latents = latents * self.scale_factor

        result = self.scheduler.add_noise(latents, noise, t, return_scaled_noise=True)
        noisy_latents, noise_used = result
        pred_noise = self.unet(noisy_latents, t, conditioning_signal=conditioning_signal)

        device_obj = noisy_latents.device
        alpha_bars = self.scheduler.alpha_bars.to(device_obj)
        alpha_bar = alpha_bars[t].view(-1, 1, 1, 1)
        
        pred_latent = (noisy_latents - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt().clamp(min=1e-8)

        return {
            "latent": latents,
            "pred_latent": pred_latent,
            "noisy_latent": noisy_latents,
            "pred_noise": pred_noise,
            "noise": noise_used,
        }

    def sample(self, batch_size=1, latent_shape=None, num_steps=50, 
               method="ddim", eta=0.0, device=None, return_history=False, verbose=False, 
               guidance_scale=1.0, text_emb=None, pov_emb=None):
 
        if device is None:
            device = next(self.parameters()).device
        
        if latent_shape is None:
            latent_ch = self.decoder._init_kwargs.get('latent_channels', 4)
            up_steps = self.decoder._init_kwargs.get('upsampling_steps', 4)
            spatial_res = 512 // (2 ** up_steps)
            latent_shape = (latent_ch, spatial_res, spatial_res)
        
        self.scheduler = self.scheduler.to(device)
        dummy = torch.zeros((batch_size, *latent_shape), device=device)
        latents = self.scheduler.randn_like(dummy)
        
        if method == "ddim":
            step_size = self.scheduler.num_steps // num_steps
            timesteps = torch.arange(self.scheduler.num_steps - 1, -1, -step_size, device=device).long()
            if timesteps[-1] != 0:
                timesteps = torch.cat([timesteps, torch.tensor([0], device=device)])
        else:
            timesteps = torch.arange(self.scheduler.num_steps - 1, -1, -1, device=device).long()
        
        history = [] if return_history else None
        
        # Prepare conditioning signal for conditional pass
        embedding_proj = getattr(self, 'embedding_projection', None)
        
        # Check if embeddings are provided for inference
        has_text_emb = text_emb is not None
        has_pov_emb = pov_emb is not None
        
        if embedding_proj is not None and (has_text_emb or has_pov_emb):
            conditioning_signal = embedding_proj(text_emb, pov_emb)
            
            # For CFG, prepare unconditional signal
            use_cfg = guidance_scale > 1.0
            if use_cfg:
                # Match training: use zero tensors, not None
                # This ensures the unconditional path sees the same input as during 
                # training with per-sample CFG dropout (zeroed embeddings)
                model_dtype = next(self.parameters()).dtype
                
                if has_text_emb:
                    zero_text = torch.zeros_like(text_emb)
                else:
                    zero_text = torch.zeros((batch_size, 384), device=device, dtype=model_dtype)
                
                if has_pov_emb:
                    zero_pov = torch.zeros_like(pov_emb)
                else:
                    zero_pov = torch.zeros((batch_size, 512), device=device, dtype=model_dtype)
                
                unconditional_signal = embedding_proj(zero_text, zero_pov)
            else:
                unconditional_signal = None
        else:
            conditioning_signal = None
            unconditional_signal = None
            use_cfg = False
        
        for i, t in enumerate(timesteps):
            if verbose and (i % max(1, len(timesteps) // 10) == 0 or i == len(timesteps) - 1):
                print(f"  Sampling step {i+1}/{len(timesteps)} (t={t.item()})")
            
            t_batch = t.expand(batch_size)
            
            with torch.no_grad():
                self.unet.eval()
                
                # Always compute conditional prediction
                cond_pred = self.unet(latents, t_batch, conditioning_signal=conditioning_signal)
                
                # Apply CFG if needed
                if use_cfg:
                    uncond_pred = self.unet(latents, t_batch, conditioning_signal=unconditional_signal)
                    pred_noise = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
                else:
                    pred_noise = cond_pred
            
            if method == "ddim":
                alpha_bars = self.scheduler.alpha_bars.to(device)
                alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
                
                if i < len(timesteps) - 1:
                    t_prev = timesteps[i + 1]
                    alpha_bar_prev = alpha_bars[t_prev].view(-1, 1, 1, 1)
                else:
                    alpha_bar_prev = torch.tensor(1.0, device=device, dtype=alpha_bar_t.dtype).view(-1, 1, 1, 1)
                
                pred_x0 = (latents - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt().clamp(min=1e-8)
                
                if eta > 0 and i < len(timesteps) - 1:
                    sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t).clamp(min=1e-8) * (1 - alpha_bar_t / alpha_bar_prev.clamp(min=1e-8))).sqrt()
                    pred_dir = (1 - alpha_bar_prev - sigma**2).sqrt().clamp(min=0.0) * pred_noise
                    noise = sigma * self.scheduler.randn_like(latents)
                else:
                    pred_dir = (1 - alpha_bar_prev).sqrt() * pred_noise
                    noise = 0
                
                latents = alpha_bar_prev.sqrt() * pred_x0 + pred_dir + noise
            
            else:
                alpha_bars = self.scheduler.alpha_bars.to(device)
                alphas = self.scheduler.alphas.to(device)
                betas = self.scheduler.betas.to(device)
                
                alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
                beta_t = betas[t].view(-1, 1, 1, 1)
                
                if i < len(timesteps) - 1:
                    t_prev = timesteps[i+1]
                    alpha_bar_prev = alpha_bars[t_prev].view(-1, 1, 1, 1)
                    alpha_t = alphas[t].view(-1, 1, 1, 1)
                    
                    pred_mean = (1.0 / alpha_t.sqrt()) * (latents - (beta_t / (1 - alpha_bar_t).sqrt()) * pred_noise)
                    posterior_variance = ((1 - alpha_bar_prev) / (1 - alpha_bar_t).clamp(min=1e-8)) * beta_t
                    posterior_variance = torch.clamp(posterior_variance, min=1e-20)
                    
                    noise = self.scheduler.randn_like(latents)
                    latents = pred_mean + posterior_variance.sqrt() * noise
                else:
                    pred_x0 = (latents - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()
                    latents = pred_x0
            
            if return_history:
                history.append(latents.clone())
        
        # First, convert latents back to original VAE scale
        if self.scale_factor != 1.0:
            latents = latents / self.scale_factor
        
        # Then clamp in original VAE scale (where clamp values are defined)
        clamp_min = getattr(self, '_latent_clamp_min', -6.0)
        clamp_max = getattr(self, '_latent_clamp_max', 6.0)
        latents_clamped = torch.clamp(latents, clamp_min, clamp_max)
        
        result = {"latent": latents_clamped}
        
        with torch.no_grad():
            decoded_out = self.decoder({"latent": latents_clamped})
            if "rgb" in decoded_out:
                rgb = decoded_out["rgb"]
                rgb = (rgb + 1.0) / 2.0
                result["rgb"] = rgb
        
        if return_history:
            result["history"] = history
        
        return result

    def sample_from(
        self, 
        x_t: torch.Tensor,
        start_step: int,
        num_steps: int = 50,
        method: str = "ddim",
        eta: float = 0.0,
        device=None,
        guidance_scale: float = 1.0,
        text_emb=None,
        pov_emb=None,
        verbose: bool = False
    ):
        """
        Sample starting from a given noisy latent at a specific timestep.
        Used for iterative refinement: noise a prior, then denoise with new conditioning.
        
        Args:
            x_t: Starting noisy latent tensor [B, C, H, W]
            start_step: Timestep to start denoising from (0 to num_scheduler_steps-1)
            num_steps: Number of denoising steps (for DDIM)
            method: "ddim" or "ddpm"
            eta: DDIM eta parameter (0 = deterministic)
            device: Device to use
            guidance_scale: CFG guidance scale
            text_emb: Text embeddings for conditioning
            pov_emb: POV embeddings for conditioning
            verbose: Print progress
        
        Returns:
            Dict with 'latent' and 'rgb' keys
        """
        if device is None:
            device = x_t.device
        
        batch_size = x_t.shape[0]
        latents = x_t.to(device)
        
        self.scheduler = self.scheduler.to(device)
        
        # Build timestep schedule starting from start_step
        if method == "ddim":
            # DDIM: use subset of timesteps
            full_step_size = self.scheduler.num_steps // num_steps
            full_timesteps = torch.arange(
                self.scheduler.num_steps - 1, -1, -full_step_size, device=device
            ).long()
            if full_timesteps[-1] != 0:
                full_timesteps = torch.cat([full_timesteps, torch.tensor([0], device=device)])
            # Only keep timesteps <= start_step
            timesteps = full_timesteps[full_timesteps <= start_step]
        else:
            # DDPM: use all timesteps from start_step down
            timesteps = torch.arange(start_step, -1, -1, device=device).long()
        
        if len(timesteps) == 0:
            timesteps = torch.tensor([0], device=device)
        
        # Prepare conditioning (same as sample method)
        embedding_proj = getattr(self, 'embedding_projection', None)
        has_text_emb = text_emb is not None
        has_pov_emb = pov_emb is not None
        
        if embedding_proj is not None and (has_text_emb or has_pov_emb):
            conditioning_signal = embedding_proj(text_emb, pov_emb)
            
            use_cfg = guidance_scale > 1.0
            if use_cfg:
                model_dtype = next(self.parameters()).dtype
                if has_text_emb:
                    zero_text = torch.zeros_like(text_emb)
                else:
                    zero_text = torch.zeros((batch_size, 384), device=device, dtype=model_dtype)
                if has_pov_emb:
                    zero_pov = torch.zeros_like(pov_emb)
                else:
                    zero_pov = torch.zeros((batch_size, 512), device=device, dtype=model_dtype)
                unconditional_signal = embedding_proj(zero_text, zero_pov)
            else:
                unconditional_signal = None
        else:
            conditioning_signal = None
            unconditional_signal = None
            use_cfg = False
        
        # Denoising loop (same logic as sample method)
        for i, t in enumerate(timesteps):
            if verbose and (i % max(1, len(timesteps) // 5) == 0):
                print(f"  Refine step {i+1}/{len(timesteps)} (t={t.item()})")
            
            t_batch = t.expand(batch_size)
            
            with torch.no_grad():
                self.unet.eval()
                cond_pred = self.unet(latents, t_batch, conditioning_signal=conditioning_signal)
                
                if use_cfg:
                    uncond_pred = self.unet(latents, t_batch, conditioning_signal=unconditional_signal)
                    pred_noise = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
                else:
                    pred_noise = cond_pred
            
            # DDIM step
            if method == "ddim":
                alpha_bars = self.scheduler.alpha_bars.to(device)
                alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
                
                if i < len(timesteps) - 1:
                    t_prev = timesteps[i + 1]
                    alpha_bar_prev = alpha_bars[t_prev].view(-1, 1, 1, 1)
                else:
                    alpha_bar_prev = torch.tensor(1.0, device=device, dtype=alpha_bar_t.dtype).view(-1, 1, 1, 1)
                
                pred_x0 = (latents - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt().clamp(min=1e-8)
                
                if eta > 0 and i < len(timesteps) - 1:
                    sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t).clamp(min=1e-8) * (1 - alpha_bar_t / alpha_bar_prev.clamp(min=1e-8))).sqrt()
                    pred_dir = (1 - alpha_bar_prev - sigma**2).sqrt().clamp(min=0.0) * pred_noise
                    noise = sigma * self.scheduler.randn_like(latents)
                else:
                    pred_dir = (1 - alpha_bar_prev).sqrt() * pred_noise
                    noise = 0
                
                latents = alpha_bar_prev.sqrt() * pred_x0 + pred_dir + noise
            else:
                # DDPM step
                alpha_bars = self.scheduler.alpha_bars.to(device)
                alphas = self.scheduler.alphas.to(device)
                betas = self.scheduler.betas.to(device)
                
                alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
                beta_t = betas[t].view(-1, 1, 1, 1)
                
                if i < len(timesteps) - 1:
                    alpha_bar_prev = alpha_bars[timesteps[i+1]].view(-1, 1, 1, 1)
                    alpha_t = alphas[t].view(-1, 1, 1, 1)
                    
                    pred_mean = (1.0 / alpha_t.sqrt()) * (latents - (beta_t / (1 - alpha_bar_t).sqrt()) * pred_noise)
                    posterior_variance = ((1 - alpha_bar_prev) / (1 - alpha_bar_t).clamp(min=1e-8)) * beta_t
                    posterior_variance = torch.clamp(posterior_variance, min=1e-20)
                    
                    noise = self.scheduler.randn_like(latents)
                    latents = pred_mean + posterior_variance.sqrt() * noise
                else:
                    pred_x0 = (latents - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()
                    latents = pred_x0
        
        # Unscale and decode
        if self.scale_factor != 1.0:
            latents = latents / self.scale_factor
        
        clamp_min = getattr(self, '_latent_clamp_min', -6.0)
        clamp_max = getattr(self, '_latent_clamp_max', 6.0)
        latents_clamped = torch.clamp(latents, clamp_min, clamp_max)
        
        result = {"latent": latents_clamped}
        
        with torch.no_grad():
            decoded_out = self.decoder({"latent": latents_clamped})
            if "rgb" in decoded_out:
                rgb = decoded_out["rgb"]
                rgb = (rgb + 1.0) / 2.0
                result["rgb"] = rgb
        
        return result

    def add_noise_to_latent(
        self,
        latent: torch.Tensor,
        timestep: int,
        noise: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Add noise to a latent at a specific timestep.
        Used to prepare prior latent for refinement.
        
        Args:
            latent: Clean latent [B, C, H, W] (in VAE scale, not diffusion scale)
            timestep: Noise level (0 = clean, num_steps-1 = pure noise)
            noise: Optional noise tensor (generated if None)
        
        Returns:
            Noised latent at timestep t
        """
        device = latent.device
        
        # Scale latent if needed
        if self.scale_factor != 1.0:
            latent = latent * self.scale_factor
        
        if noise is None:
            noise = torch.randn_like(latent)
        
        t = torch.tensor([timestep], device=device)
        noised = self.scheduler.add_noise(latent, noise, t)
        
        return noised

    @classmethod
    def load_config(cls, cfg_path):
        cfg_path = Path(cfg_path)
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cls.from_config(cfg)

    def to_config(self):
        cfg = {"type": "DiffusionModel"}
        cfg = self._components_to_config(cfg)
        
        # Non-component config values
        if hasattr(self, 'scale_factor') and self.scale_factor != 1.0:
            cfg["scale_factor"] = self.scale_factor
        
        return cfg

    def save_checkpoint(self, path, include_config=True, use_compression=False, **extra_state):
        """
        Save diffusion model checkpoint with all components nested.
        
        Ensures decoder, UNet, scheduler, and embedding_projection are all included in state_dict,
        even if frozen. All components are nested within the diffusion model.
        
        The checkpoint includes:
        - decoder: Frozen decoder from VAE (for decoding latents to images)
        - embedding_projection: Contains CLIP projections (frozen, loaded from checkpoint) 
          and spatial projection layers (trained, convert CLIP embeddings to spatial features)
        - unet: Trained UNet (denoising network)
        - scheduler: Noise scheduler (no trainable params, but state is saved)
        """
        path = Path(path)
        state_dict = self.state_dict()
        
        has_decoder = any(k.startswith("decoder.") for k in state_dict.keys())
        has_unet = any(k.startswith("unet.") for k in state_dict.keys())
        has_scheduler = any(k.startswith("scheduler.") for k in state_dict.keys())
        has_embedding_proj = any(k.startswith("embedding_projection.") for k in state_dict.keys())
        
        if not has_decoder:
            raise RuntimeError("Decoder not found in state_dict - checkpoint incomplete!")
        if not has_unet:
            raise RuntimeError("UNet not found in state_dict - checkpoint incomplete!")
        if not has_scheduler:
            raise RuntimeError("Scheduler not found in state_dict - checkpoint incomplete!")
        # embedding_projection is optional (only if conditioning is used)
        # but if it exists, it should be in state_dict
        
        payload = {"state_dict": state_dict}
        if include_config:
            payload["config"] = self.to_config()
        
        payload.update(extra_state)
        
        if use_compression:
            # Save with gzip compression
            with gzip.open(path, 'wb') as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            torch.save(payload, path)
    
    @classmethod
    def load_checkpoint(cls, path, map_location="cpu", return_extra=False, config=None):
        """
        Load diffusion model checkpoint.
        
        When loading a diffusion checkpoint, the decoder config comes from the saved checkpoint,
        not from an external autoencoder checkpoint. All components (decoder, UNet, scheduler)
        are saved in the checkpoint's state_dict and config.
        """
        path = Path(path)
        # Try to detect if file is compressed
        try:
            with gzip.open(path, 'rb') as f:
                payload = pickle.load(f)
        except (gzip.BadGzipFile, OSError):
            payload = torch.load(path, map_location=map_location)
        
        state_dict = payload.get("state_dict", payload)
        has_decoder_state = any(key.startswith("decoder.") for key in state_dict.keys())
        saved_config = payload.get("config")
        
        if has_decoder_state and saved_config and isinstance(saved_config, dict):
            if config and isinstance(config, dict):
                merged_config = config.copy()
                
                saved_decoder_config = None
                if "decoder" in saved_config:
                    saved_decoder_config = saved_config["decoder"]
                elif "autoencoder" in saved_config:
                    if "decoder" in saved_config["autoencoder"]:
                        saved_decoder_config = saved_config["autoencoder"]["decoder"]
                    elif isinstance(saved_config["autoencoder"], dict) and "decoder" not in saved_config["autoencoder"]:
                        # Saved config might have decoder at top level when autoencoder is just metadata
                        if "decoder" in saved_config:
                            saved_decoder_config = saved_config["decoder"]
                
                if saved_decoder_config:
                    if "autoencoder" in merged_config:
                        # Preserve checkpoint and other fields from original autoencoder config
                        merged_config["autoencoder"] = {
                            **merged_config["autoencoder"],  # Preserve all original fields (checkpoint, etc.)
                            "decoder": saved_decoder_config,  # Override decoder with saved config
                            "frozen": merged_config["autoencoder"].get("frozen", False)
                        }
                    else:
                        merged_config["decoder"] = saved_decoder_config
                else:
                    raise ValueError("Saved checkpoint config missing decoder config - checkpoint incomplete!")
                
                if "scheduler" in config:
                    merged_config["scheduler"] = config["scheduler"]
                
                if "scale_factor" in config:
                    merged_config["scale_factor"] = config["scale_factor"]
                elif "scale_factor" in saved_config:
                    merged_config["scale_factor"] = saved_config["scale_factor"]
                
                model_config = merged_config
            else:
                model_config = saved_config
        else:
            if config is not None:
                model_config = config.copy()
                if "scale_factor" not in model_config and saved_config and "scale_factor" in saved_config:
                    model_config["scale_factor"] = saved_config["scale_factor"]
            else:
                model_config = saved_config
        
        model = cls.from_config(model_config) if model_config else cls()
        model.load_state_dict(payload["state_dict"], strict=False)

        if return_extra:
            extra_state = {k: v for k, v in payload.items() 
                          if k not in ["state_dict", "config"]}
            return model, extra_state
        
        return model