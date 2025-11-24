import torch
import torch.nn as nn
from pathlib import Path
import yaml

from models.components.base_model import BaseModel
from models.autoencoder import Autoencoder
from models.decoder import Decoder
from models.components.unet import Unet, DualUNet, UnetWithAttention  # DualUNet for backward compatibility
from models.components.scheduler import SCHEDULER_REGISTRY
from models.components.embedding_projection import EmbeddingToSpatial, CLIPEmbeddingToSpatial


class DiffusionModel(BaseModel):
    """Minimal end-to-end diffusion model for training."""

    def _build(self):
        ae_cfg = self._init_kwargs.get("autoencoder", None)
        decoder_cfg = self._init_kwargs.get("decoder", None)
        unet_cfg = self._init_kwargs.get("unet", {})
        sched_cfg = self._init_kwargs.get("scheduler", {})

        if ae_cfg:
            ae_checkpoint = ae_cfg.get("checkpoint")
            encoder_cfg = ae_cfg.get("encoder", {}) if isinstance(ae_cfg, dict) else {}
            self._is_vae = isinstance(encoder_cfg, dict) and encoder_cfg.get("variational", False)
            
            if ae_checkpoint:
                autoencoder = Autoencoder.load_checkpoint(ae_checkpoint, map_location="cpu")
                self.decoder = autoencoder.decoder
                if hasattr(autoencoder, 'encoder') and autoencoder.encoder is not None:
                    self._is_vae = getattr(autoencoder.encoder, 'variational', False)
            else:
                decoder_subcfg = ae_cfg.get("decoder")
                if decoder_subcfg:
                    decoder_subcfg = decoder_subcfg.copy()
                    decoder_subcfg.pop("checkpoint", None)
                    self.decoder = Decoder.from_config(decoder_subcfg)
                else:
                    raise ValueError("Cannot build decoder: no checkpoint path and no decoder config in autoencoder config")
            
            self.encoder = None
            self.autoencoder = None
            self._has_encoder = False
            if ae_cfg.get("frozen", False):
                self.decoder.freeze()
        elif decoder_cfg:
            decoder_checkpoint = decoder_cfg.get("checkpoint")
            self._is_vae = False
            if decoder_checkpoint:
                autoencoder = Autoencoder.load_checkpoint(decoder_checkpoint, map_location="cpu")
                self.decoder = autoencoder.decoder
                if hasattr(autoencoder, 'encoder') and autoencoder.encoder is not None:
                    self._is_vae = getattr(autoencoder.encoder, 'variational', False)
            else:
                decoder_cfg_copy = decoder_cfg.copy()
                decoder_cfg_copy.pop("checkpoint", None)
                self.decoder = Decoder.from_config(decoder_cfg_copy)
            
            self.encoder = None
            self.autoencoder = None
            self._has_encoder = False
            if decoder_cfg.get("frozen", False):
                self.decoder.freeze()
        else:
            raise ValueError("DiffusionModel requires either 'autoencoder' or 'decoder' config")

        # Build embedding projection
        embedding_proj_cfg = self._init_kwargs.get("embedding_projection", None)
        self.embedding_proj = None
        conditioning_channels = None
        
        if embedding_proj_cfg:
            if "output_channels" not in embedding_proj_cfg:
                embedding_proj_cfg["output_channels"] = unet_cfg.get("base_channels", 96)
            conditioning_channels = embedding_proj_cfg.get("output_channels")
            
            # Load CLIP projections from VAE if using CLIPEmbeddingToSpatial
            embedding_proj_type = embedding_proj_cfg.get("type", "EmbeddingToSpatial")
            if embedding_proj_type == "CLIPEmbeddingToSpatial":
                if not ae_cfg or not ae_cfg.get("checkpoint"):
                    raise ValueError(
                        "CLIPEmbeddingToSpatial requires autoencoder.checkpoint to load CLIP projections. "
                        "This experiment requires CLIP projections from the VAE."
                    )
                
                try:
                    autoencoder = Autoencoder.load_checkpoint(ae_cfg.get("checkpoint"), map_location="cpu")
                    if not hasattr(autoencoder, 'clip_projections') or autoencoder.clip_projections is None:
                        raise ValueError(
                            f"VAE checkpoint {ae_cfg.get('checkpoint')} does not have CLIP projections. "
                            "This experiment requires a VAE trained with CLIP projections."
                        )
                    embedding_proj_cfg["clip_projections"] = autoencoder.clip_projections
                    print("✓ Loaded CLIP projections from VAE checkpoint for embedding projection")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load CLIP projections from VAE checkpoint: {e}\n"
                        "This experiment requires CLIP projections. Cannot proceed without them."
                    ) from e
            
            # Create embedding projection
            if embedding_proj_type == "CLIPEmbeddingToSpatial":
                self.embedding_proj = CLIPEmbeddingToSpatial.from_config(embedding_proj_cfg)
                # Verify CLIP projections are actually set
                if not hasattr(self.embedding_proj, 'clip_projections') or self.embedding_proj.clip_projections is None:
                    raise RuntimeError(
                        "CLIPEmbeddingToSpatial was created but clip_projections is None. "
                        "This experiment requires CLIP projections to work."
                    )
                print("✓ CLIPEmbeddingToSpatial initialized with CLIP projections")
                
                # Freeze CLIP projections (from VAE, should not be trained)
                # But keep spatial_proj trainable (it learns to convert CLIP embeddings to spatial features)
                if hasattr(self.embedding_proj, 'clip_projections') and self.embedding_proj.clip_projections is not None:
                    for p in self.embedding_proj.clip_projections.parameters():
                        p.requires_grad = False
                    print("✓ CLIP projections frozen (from VAE)")
                
                # spatial_proj remains trainable - it learns to project CLIP joint space to spatial features
                spatial_proj_params = sum(p.numel() for p in self.embedding_proj.spatial_proj.parameters())
                print(f"✓ spatial_proj trainable ({spatial_proj_params:,} parameters)")
            else:
                self.embedding_proj = EmbeddingToSpatial.from_config(embedding_proj_cfg)
                # For non-CLIP embedding projection, freeze everything
                for p in self.embedding_proj.parameters():
                    p.requires_grad = False
                print("✓ Embedding projection frozen (only UNet will be trained)")
        
        # Build UNet
        unet_type = unet_cfg.get("type", "").lower()
        if conditioning_channels is not None and unet_cfg.get("enable_cross_attention", False):
            unet_cfg = unet_cfg.copy()
            unet_cfg["conditioning_channels"] = conditioning_channels
        
        if unet_type in ("dualunet", "dual_unet"):
            self.unet = DualUNet.from_config(unet_cfg)
        elif unet_type in ("unetwithattention", "unet_with_attention"):
            self.unet = UnetWithAttention.from_config(unet_cfg)
        else:
            self.unet = Unet.from_config(unet_cfg)
        
        # Freeze UNet if requested
        if unet_cfg.get("frozen", False):
            self.unet.freeze()
        if unet_cfg.get("freeze_downblocks", False):
            self.unet.freeze_downblocks()
        if unet_cfg.get("freeze_upblocks", False):
            self.unet.freeze_upblocks()
        freeze_blocks = unet_cfg.get("freeze_blocks", None)
        if freeze_blocks:
            self.unet.freeze_blocks(freeze_blocks)

        # Build scheduler
        sched_type = sched_cfg.get("type", "CosineScheduler")
        if sched_type not in SCHEDULER_REGISTRY:
            raise ValueError(f"Unknown scheduler: {sched_type}")
        self.scheduler = SCHEDULER_REGISTRY[sched_type].from_config(sched_cfg)
        
        self.scale_factor = self._init_kwargs.get("scale_factor", 1.0)
        self._latent_clamp_min = self._init_kwargs.get("latent_clamp_min", -6.0)
        self._latent_clamp_max = self._init_kwargs.get("latent_clamp_max", 6.0)
        
        self._write_model_statistics()

    def _write_model_statistics(self):
        """Write model parameter statistics to Statistics.txt file."""
        try:
            save_path = self._init_kwargs.get("save_path", None)
            if save_path is None:
                exp_cfg = self._init_kwargs.get("experiment", {})
                save_path = exp_cfg.get("save_path", None)
            
            if save_path is None:
                return
            
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            stats_file = save_path / "Statistics.txt"
            
            unet_trainable = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
            unet_total = sum(p.numel() for p in self.unet.parameters())
            unet_frozen = unet_total - unet_trainable
            
            decoder_trainable = 0
            decoder_total = 0
            if hasattr(self, 'decoder'):
                decoder_trainable = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
                decoder_total = sum(p.numel() for p in self.decoder.parameters())
            
            with open(stats_file, 'w') as f:
                f.write("Model Statistics\n")
                f.write("=" * 60 + "\n\n")
                f.write("UNet Parameters:\n")
                f.write(f"  Trainable: {unet_trainable:,} ({unet_trainable / 1_000_000:.2f}M)\n")
                f.write(f"  Total: {unet_total:,} ({unet_total / 1_000_000:.2f}M)\n")
                f.write(f"  Frozen: {unet_frozen:,} ({unet_frozen / 1_000_000:.2f}M)\n")
                if hasattr(self, 'decoder'):
                    f.write(f"\nDecoder Parameters (frozen):\n")
                    f.write(f"  Total: {decoder_total:,} ({decoder_total / 1_000_000:.2f}M)\n")
                f.write(f"\nTotal Trainable Parameters: {unet_trainable:,} ({unet_trainable / 1_000_000:.2f}M)\n")
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to write model statistics: {e}")

    def forward(self, x0_or_latents, t, cond=None, noise=None, text_emb=None, pov_emb=None):
        # Validate CLIP projections are being used if configured
        if self.embedding_proj is not None:
            if isinstance(self.embedding_proj, CLIPEmbeddingToSpatial):
                if not hasattr(self.embedding_proj, 'clip_projections') or self.embedding_proj.clip_projections is None:
                    raise RuntimeError(
                        "CLIPEmbeddingToSpatial is configured but clip_projections is None. "
                        "This experiment requires CLIP projections. Cannot proceed."
                    )
        
        if self.embedding_proj is not None and (text_emb is not None or pov_emb is not None):
            conditioning_signal = self.embedding_proj(text_emb, pov_emb)
        else:
            conditioning_signal = None
        
        if self._has_encoder:
            encoder_out = self.encoder(x0_or_latents)
            if "latent" in encoder_out:
                latents = encoder_out["latent"]
            elif "mu" in encoder_out and "logvar" in encoder_out:
                latents = encoder_out["mu"]
            else:
                raise ValueError(f"Encoder output must contain 'latent' or 'mu'/'logvar'. Got: {list(encoder_out.keys())}")
            
            if self.scale_factor != 1.0:
                latents = latents * self.scale_factor
            
            if noise is None:
                noise = self.scheduler.randn_like(latents)
            elif noise.shape != latents.shape:
                if noise.shape == x0_or_latents.shape:
                    noise_out = self.encoder(noise)
                    if "latent" in noise_out:
                        noise = noise_out["latent"]
                    elif "mu" in noise_out:
                        noise = noise_out["mu"]
                    else:
                        noise = self.scheduler.randn_like(latents)
                else:
                    noise = self.scheduler.randn_like(latents)
        else:
            if isinstance(x0_or_latents, dict):
                if "latent" in x0_or_latents:
                    latents = x0_or_latents["latent"]
                elif "mu" in x0_or_latents:
                    latents = x0_or_latents["mu"]
                else:
                    raise ValueError(f"Latent dict must contain 'latent' or 'mu'. Got: {list(x0_or_latents.keys())}")
            else:
                latents = x0_or_latents
            
            if self.scale_factor != 1.0:
                latents = latents * self.scale_factor
            
            if noise is None:
                noise = self.scheduler.randn_like(latents)

        result = self.scheduler.add_noise(latents, noise, t, return_scaled_noise=True)
        noisy_latents, noise_used = result
        pred_noise = self.unet(noisy_latents, t, cond, conditioning_signal=conditioning_signal)

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

    def sample(self, batch_size=1, latent_shape=None, cond=None, num_steps=50, 
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
        if self.embedding_proj is not None and (text_emb is not None or pov_emb is not None):
            conditioning_signal = self.embedding_proj(text_emb, pov_emb)
            
            # For CFG, prepare unconditional (zero) conditioning signal
            # This must match what was used during training: zero embeddings passed through embedding_proj
            # During training, CFG dropout sets both text_emb and pov_emb to zeros_like (if they exist)
            # or creates zero tensors (if they don't exist), then passes both through embedding_proj
            use_cfg = guidance_scale > 1.0
            if use_cfg:
                # Create zero embeddings matching the conditional embeddings
                # Get dtype and device from existing embeddings or model parameters
                if text_emb is not None:
                    batch_size_cfg = text_emb.shape[0]
                    device_cfg = text_emb.device
                    dtype_cfg = text_emb.dtype
                    zero_text_emb = torch.zeros_like(text_emb)
                elif pov_emb is not None:
                    batch_size_cfg = pov_emb.shape[0]
                    device_cfg = pov_emb.device
                    dtype_cfg = pov_emb.dtype
                    zero_text_emb = torch.zeros((batch_size_cfg, 384), device=device_cfg, dtype=dtype_cfg)
                else:
                    # Fallback: use batch_size and device from function args
                    batch_size_cfg = batch_size
                    device_cfg = device
                    param_dtype = next(self.embedding_proj.parameters()).dtype
                    dtype_cfg = param_dtype
                    zero_text_emb = torch.zeros((batch_size_cfg, 384), device=device_cfg, dtype=dtype_cfg)
                
                # Create zero pov_emb (always needed for embedding_proj)
                if pov_emb is not None:
                    zero_pov_emb = torch.zeros_like(pov_emb)
                else:
                    zero_pov_emb = torch.zeros((batch_size_cfg, 512), device=device_cfg, dtype=dtype_cfg)
                
                # Project zero embeddings through embedding_proj to get the same projected zero signal as training
                # This matches training behavior where zero embeddings are passed through the projection
                unconditional_signal = self.embedding_proj(zero_text_emb, zero_pov_emb)
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
                
                if use_cfg:
                    # Conditional prediction with actual conditioning signal
                    cond_pred = self.unet(latents, t_batch, cond, conditioning_signal=conditioning_signal)
                    # Unconditional prediction with projected zero signal (matches training)
                    uncond_pred = self.unet(latents, t_batch, cond, conditioning_signal=unconditional_signal)
                    pred_noise = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
                else:
                    pred_noise = self.unet(latents, t_batch, cond, conditioning_signal=conditioning_signal)
            
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
        
        clamp_min = getattr(self, '_latent_clamp_min', -6.0)
        clamp_max = getattr(self, '_latent_clamp_max', 6.0)
        latents_clamped = torch.clamp(latents, clamp_min, clamp_max)
        
        if self.scale_factor != 1.0:
            latents_clamped = latents_clamped / self.scale_factor
        
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

    @classmethod
    def load_config(cls, cfg_path):
        cfg_path = Path(cfg_path)
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cls.from_config(cfg)

    def to_config(self):
        cfg = {
            "type": "DiffusionModel",
            "unet": self.unet.to_config(),
            "scheduler": self.scheduler.to_config(),
        }
        if self._has_encoder:
            cfg["autoencoder"] = self.autoencoder.to_config()
        else:
            cfg["decoder"] = self.decoder.to_config()
        if hasattr(self, 'scale_factor') and self.scale_factor != 1.0:
            cfg["scale_factor"] = self.scale_factor
        if hasattr(self, 'embedding_proj') and self.embedding_proj is not None:
            embedding_cfg = self.embedding_proj.to_config()
            if isinstance(self.embedding_proj, CLIPEmbeddingToSpatial):
                embedding_cfg["type"] = "CLIPEmbeddingToSpatial"
            else:
                embedding_cfg["type"] = "EmbeddingToSpatial"
            cfg["embedding_projection"] = embedding_cfg
        return cfg

    def save_checkpoint(self, path, include_config=True, **extra_state):
        """
        Save diffusion model checkpoint with all components nested.
        
        Ensures decoder, UNet, and scheduler are all included in state_dict,
        even if frozen. All components are nested within the diffusion model.
        """
        path = Path(path)
        state_dict = self.state_dict()
        
        has_decoder = any(k.startswith("decoder.") for k in state_dict.keys())
        has_unet = any(k.startswith("unet.") for k in state_dict.keys())
        has_scheduler = any(k.startswith("scheduler.") for k in state_dict.keys())
        
        if not has_decoder:
            raise RuntimeError("Decoder not found in state_dict - checkpoint incomplete!")
        if not has_unet:
            raise RuntimeError("UNet not found in state_dict - checkpoint incomplete!")
        if not has_scheduler:
            raise RuntimeError("Scheduler not found in state_dict - checkpoint incomplete!")
        
        payload = {"state_dict": state_dict}
        if include_config:
            payload["config"] = self.to_config()
        
        payload.update(extra_state)
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
