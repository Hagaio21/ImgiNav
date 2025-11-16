import torch
import torch.nn as nn
from pathlib import Path
import yaml

from models.components.base_model import BaseModel
from models.autoencoder import Autoencoder
from models.decoder import Decoder
from models.components.unet import Unet, DualUNet, UnetWithAttention  # DualUNet for backward compatibility
from models.components.scheduler import SCHEDULER_REGISTRY


class DiffusionModel(BaseModel):
    """Minimal end-to-end diffusion model for training."""

    def _build(self):
        ae_cfg = self._init_kwargs.get("autoencoder", None)
        decoder_cfg = self._init_kwargs.get("decoder", None)
        unet_cfg = self._init_kwargs.get("unet", {})
        sched_cfg = self._init_kwargs.get("scheduler", {})

        # Build decoder: 
        # - If checkpoint path provided: load from external checkpoint (initial training)
        # - If no checkpoint path: build from config (loading from diffusion checkpoint, weights in state_dict)
        if ae_cfg:
            # Autoencoder config provided
            ae_checkpoint = ae_cfg.get("checkpoint")
            
            # Check if using VAE (for clamping bounds later)
            encoder_cfg = ae_cfg.get("encoder", {}) if isinstance(ae_cfg, dict) else {}
            self._is_vae = isinstance(encoder_cfg, dict) and encoder_cfg.get("variational", False)
            
            if ae_checkpoint:
                # Checkpoint path provided - load from external checkpoint (initial training)
                autoencoder = Autoencoder.load_checkpoint(ae_checkpoint, map_location="cpu")
                self.decoder = autoencoder.decoder
                # Also check loaded autoencoder's encoder to confirm VAE
                if hasattr(autoencoder, 'encoder') and autoencoder.encoder is not None:
                    self._is_vae = getattr(autoencoder.encoder, 'variational', False)
            else:
                # No checkpoint path - build from config (loading from diffusion checkpoint)
                # Decoder config should be in autoencoder.decoder subconfig
                decoder_subcfg = ae_cfg.get("decoder")
                if decoder_subcfg:
                    # Remove checkpoint if present (shouldn't be, but just in case)
                    decoder_subcfg = decoder_subcfg.copy()
                    decoder_subcfg.pop("checkpoint", None)
                    self.decoder = Decoder.from_config(decoder_subcfg)
                else:
                    raise ValueError("Cannot build decoder: no checkpoint path and no decoder config in autoencoder config")
            
            self.encoder = None  # Not needed for pre-embedded latents
            self.autoencoder = None
            self._has_encoder = False
            # Freeze decoder if requested
            if ae_cfg.get("frozen", False):
                self.decoder.freeze()
        elif decoder_cfg:
            # Decoder config provided
            decoder_checkpoint = decoder_cfg.get("checkpoint")
            
            # Check if using VAE (for clamping bounds later)
            # If loading from checkpoint, check the loaded autoencoder
            self._is_vae = False
            if decoder_checkpoint:
                # Checkpoint path provided - load from external checkpoint (initial training)
                # Note: decoder checkpoint should actually be an autoencoder checkpoint
                autoencoder = Autoencoder.load_checkpoint(decoder_checkpoint, map_location="cpu")
                self.decoder = autoencoder.decoder
                # Check loaded autoencoder's encoder to confirm VAE
                if hasattr(autoencoder, 'encoder') and autoencoder.encoder is not None:
                    self._is_vae = getattr(autoencoder.encoder, 'variational', False)
            else:
                # No checkpoint path - build from config (loading from diffusion checkpoint)
                decoder_cfg_copy = decoder_cfg.copy()
                decoder_cfg_copy.pop("checkpoint", None)  # Remove checkpoint key if present
                self.decoder = Decoder.from_config(decoder_cfg_copy)
            
            self.encoder = None
            self.autoencoder = None
            self._has_encoder = False
            # Freeze decoder if requested
            if decoder_cfg.get("frozen", False):
                self.decoder.freeze()
        else:
            raise ValueError("DiffusionModel requires either 'autoencoder' or 'decoder' config")

        # UNet backbone
        # Handle backward compatibility: if config specifies DualUNet, use it (for old checkpoints)
        # Also support UnetWithAttention for attention-enabled models
        unet_type = unet_cfg.get("type", "").lower()
        if unet_type in ("dualunet", "dual_unet"):
            # Old checkpoint format - use DualUNet (backward compatible alias)
            self.unet = DualUNet.from_config(unet_cfg)
        elif unet_type in ("unetwithattention", "unet_with_attention"):
            # Attention-enabled UNet
            self.unet = UnetWithAttention.from_config(unet_cfg)
        else:
            # New format - use Unet (default)
            self.unet = Unet.from_config(unet_cfg)
        # Freeze UNet if requested (or use special ControlNet freezing)
        if unet_cfg.get("frozen", False):
            self.unet.freeze()
        # Support ControlNet-style freezing via config
        if unet_cfg.get("freeze_downblocks", False):
            self.unet.freeze_downblocks()
        if unet_cfg.get("freeze_upblocks", False):
            self.unet.freeze_upblocks()
        freeze_blocks = unet_cfg.get("freeze_blocks", None)
        if freeze_blocks:
            self.unet.freeze_blocks(freeze_blocks)
        
        # EMA (Exponential Moving Average) for UNet
        # Create a copy of UNet for EMA - this will be used for sampling
        ema_decay = unet_cfg.get("ema_decay", 0.9999)  # Default decay rate
        self.use_ema = unet_cfg.get("use_ema", True)  # Can be disabled via config
        if self.use_ema:
            # Create EMA UNet by copying the structure
            import copy
            self.unet_ema = copy.deepcopy(self.unet)
            # Freeze EMA UNet parameters (they'll be updated via EMA, not gradients)
            for param in self.unet_ema.parameters():
                param.requires_grad = False
            self.ema_decay = ema_decay
        else:
            self.unet_ema = None
            self.ema_decay = None

        # noise scheduler
        sched_type = sched_cfg.get("type", "CosineScheduler")
        if sched_type not in SCHEDULER_REGISTRY:
            raise ValueError(f"Unknown scheduler: {sched_type}")
        self.scheduler = SCHEDULER_REGISTRY[sched_type].from_config(sched_cfg)
        
        # Write model statistics if save_path is available
        self._write_model_statistics()

    def _write_model_statistics(self):
        """Write model parameter statistics to Statistics.txt file."""
        try:
            # Get save path from experiment config if available
            save_path = self._init_kwargs.get("save_path", None)
            if save_path is None:
                # Try to get from experiment config in parent kwargs
                exp_cfg = self._init_kwargs.get("experiment", {})
                save_path = exp_cfg.get("save_path", None)
            
            if save_path is None:
                return  # No save path available, skip writing
            
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            stats_file = save_path / "Statistics.txt"
            
            # Count parameters
            # For diffusion, count UNet parameters (decoder is frozen)
            unet_trainable = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
            unet_total = sum(p.numel() for p in self.unet.parameters())
            unet_frozen = unet_total - unet_trainable
            
            # Also count decoder if it exists (usually frozen)
            decoder_trainable = 0
            decoder_total = 0
            if hasattr(self, 'decoder'):
                decoder_trainable = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
                decoder_total = sum(p.numel() for p in self.decoder.parameters())
            
            # Write statistics
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
            # Don't fail model building if statistics writing fails
            import warnings
            warnings.warn(f"Failed to write model statistics: {e}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x0_or_latents, t, cond=None, noise=None):
        """
        Forward diffusion training step.
        
        Args:
            x0_or_latents: Either images (if encoder available) or latents (if decoder-only)
            t: Timestep tensor
            cond: Optional conditioning
            noise: Optional noise tensor
        """
        # Encode image to latents if encoder is available, otherwise assume input is already latents
        if self._has_encoder:
            # x0 is images, encode them
            encoder_out = self.encoder(x0_or_latents)  # Returns dict
            # Extract latent z from dict
            if "latent" in encoder_out:
                latents = encoder_out["latent"]
            elif "mu" in encoder_out and "logvar" in encoder_out:
                # VAE mode: use mu as deterministic representation for diffusion
                # (diffusion will add its own noise)
                latents = encoder_out["mu"]
            else:
                raise ValueError(f"Encoder output must contain 'latent' or 'mu'/'logvar'. Got: {list(encoder_out.keys())}")
            
            # Clamp latents to [-4, 4] right after loading (diffusion only sees clamped latents)
            latents = torch.clamp(latents, -4.0, 4.0)
            
            if noise is None:
                noise = self.scheduler.randn_like(latents)
            elif noise.shape != latents.shape:
                # If noise is in image space, encode it to latent space
                if noise.shape == x0_or_latents.shape:
                    noise_out = self.encoder(noise)
                    # Extract latent from encoder output
                    if "latent" in noise_out:
                        noise = noise_out["latent"]
                    elif "mu" in noise_out:
                        noise = noise_out["mu"]
                    else:
                        noise = self.scheduler.randn_like(latents)
                else:
                    noise = self.scheduler.randn_like(latents)
        else:
            # Input is already latents (decoder-only mode)
            # Can be dict or tensor - handle both
            if isinstance(x0_or_latents, dict):
                if "latent" in x0_or_latents:
                    latents = x0_or_latents["latent"]
                elif "mu" in x0_or_latents:
                    latents = x0_or_latents["mu"]
                else:
                    raise ValueError(f"Latent dict must contain 'latent' or 'mu'. Got: {list(x0_or_latents.keys())}")
            else:
                latents = x0_or_latents  # Direct tensor
            
            # Clamp latents to [-4, 4] right after loading (diffusion only sees clamped latents)
            latents = torch.clamp(latents, -4.0, 4.0)
            
            if noise is None:
                noise = self.scheduler.randn_like(latents)

        # add noise (standard N(0,1) noise, latents should be normalized)
        # Get both noisy latents and the noise that was used
        result = self.scheduler.add_noise(latents, noise, t, return_scaled_noise=True)
        noisy_latents, noise_used = result

        # predict noise using live UNet (EMA UNet only used at sampling time)
        pred_noise = self.unet(noisy_latents, t, cond)
        

        device_obj = noisy_latents.device
        alpha_bars = self.scheduler.alpha_bars.to(device_obj)
        alpha_bar = alpha_bars[t].view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        
        # Predict x0 from noisy latents: x0 = (x_t - sqrt(1 - alpha_bar) * epsilon) / sqrt(alpha_bar)
        pred_latent = (noisy_latents - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt().clamp(min=1e-8)

        # Do NOT decode during training - decoder should only see clean latents at sampling time
        return {
            "latent": latents,  # Ground-truth clean latents (for reference/targets)
            "pred_latent": pred_latent,  # Predicted/denoised latents (what model generates)
            "noisy_latent": noisy_latents,
            "pred_noise": pred_noise,
            "noise": noise_used,  # Return the noise used (for loss computation)
        }

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample(self, batch_size=1, latent_shape=None, cond=None, num_steps=50, 
               method="ddim", eta=0.0, device=None, return_history=False, verbose=False, guidance_scale=1.0):
 
        if device is None:
            device = next(self.parameters()).device
        
        # Infer latent shape from decoder config if not provided
        if latent_shape is None:
            latent_ch = self.decoder._init_kwargs.get('latent_channels', 4)
            up_steps = self.decoder._init_kwargs.get('upsampling_steps', 4)
            spatial_res = 512 // (2 ** up_steps)
            latent_shape = (latent_ch, spatial_res, spatial_res)
        
        self.scheduler = self.scheduler.to(device)
        # Start with standard normal noise N(0,1)
        # Create a dummy tensor with the right shape, then generate noise
        dummy = torch.zeros((batch_size, *latent_shape), device=device)
        latents = self.scheduler.randn_like(dummy)
        # Clamp latents to [-4, 4] right after loading (diffusion only sees clamped latents)
        latents = torch.clamp(latents, -4.0, 4.0)
        
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
            
            # Clamp latents before passing to UNet (diffusion always sees clamped latents)
            latents = torch.clamp(latents, -4.0, 4.0)
            
            with torch.no_grad():
                # Use EMA UNet for sampling if available, otherwise use live UNet
                if self.use_ema and self.unet_ema is not None:
                    unet = self.unet_ema
                else:
                    unet = self.unet
                unet.eval()
                
                # Classifier-Free Guidance (CFG)
                if guidance_scale > 1.0 and cond is not None:
                    # Run UNet with condition (conditional prediction)
                    cond_pred = unet(latents, t_batch, cond)
                    # Run UNet without condition (unconditional prediction)
                    uncond_pred = unet(latents, t_batch, None)
                    # Combine: pred = uncond + scale * (cond - uncond)
                    pred_noise = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
                else:
                    # No CFG: use conditional or unconditional prediction directly
                    pred_noise = unet(latents, t_batch, cond)
            
            # DDIM step
            if method == "ddim":
                # Move scheduler tensors to device if needed
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
                
                # DDIM update (deterministic if eta=0)
                # Standard DDIM formula: x_{t-1} = sqrt(alpha_bar_{t-1}) * pred_x0 + sqrt(1 - alpha_bar_{t-1} - sigma_t^2) * epsilon + sigma_t * z
                if eta > 0 and i < len(timesteps) - 1:
                    # Add stochastic noise if eta > 0
                    sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t).clamp(min=1e-8) * (1 - alpha_bar_t / alpha_bar_prev.clamp(min=1e-8))).sqrt()
                    pred_dir = (1 - alpha_bar_prev - sigma**2).sqrt().clamp(min=0.0) * pred_noise
                    noise = sigma * self.scheduler.randn_like(latents)
                else:
                    # Deterministic DDIM (eta=0): x_{t-1} = sqrt(alpha_bar_{t-1}) * pred_x0 + sqrt(1 - alpha_bar_{t-1}) * epsilon
                    pred_dir = (1 - alpha_bar_prev).sqrt() * pred_noise
                    noise = 0
                
                latents = alpha_bar_prev.sqrt() * pred_x0 + pred_dir + noise
            
            else:
                # DDPM step
                # Move scheduler tensors to device if needed
                alpha_bars = self.scheduler.alpha_bars.to(device)
                alphas = self.scheduler.alphas.to(device)
                betas = self.scheduler.betas.to(device)
                
                alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
                beta_t = betas[t].view(-1, 1, 1, 1)
                
                if i < len(timesteps) - 1:
                    # Get t-1 (next timestep in reverse order)
                    t_prev = timesteps[i+1]
                    alpha_bar_prev = alpha_bars[t_prev].view(-1, 1, 1, 1)
                    alpha_t = alphas[t].view(-1, 1, 1, 1)
                    
                    # DDPM posterior mean: predict x_{t-1} from x_t and predicted noise
                    # Formula: pred_mean = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * epsilon)
                    pred_mean = (1.0 / alpha_t.sqrt()) * (latents - (beta_t / (1 - alpha_bar_t).sqrt()) * pred_noise)
                    
                    # Posterior variance: beta_tilde = (1-alpha_bar_{t-1})/(1-alpha_bar_t) * beta_t
                    # For stability, clamp to avoid numerical issues
                    posterior_variance = ((1 - alpha_bar_prev) / (1 - alpha_bar_t).clamp(min=1e-8)) * beta_t
                    posterior_variance = torch.clamp(posterior_variance, min=1e-20)
                    
                    # Sample x_{t-1} with standard normal noise
                    noise = self.scheduler.randn_like(latents)
                    latents = pred_mean + posterior_variance.sqrt() * noise
                else:
                    # Last step: predict x_0 and use it directly (no noise)
                    pred_x0 = (latents - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()
                    latents = pred_x0
            
            if return_history:
                history.append(latents.clone())
        
        # Clamp latents before decoding (decoder only sees clamped latents)
        # Use stored VAE flag from _build()
        is_vae = getattr(self, '_is_vae', False)
        
        if is_vae:
            # VAE: wider bounds due to better KL regularization
            latents = torch.clamp(latents, -4.0, 4.0)
        else:
            # AE: tighter bounds to match training distribution
            latents = torch.clamp(latents, -1.0, 1.0)
        
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
                result["rgb"] = rgb
        
        if return_history:
            result["history"] = history
        
        return result

    # ------------------------------------------------------------------
    # EMA Update
    # ------------------------------------------------------------------
    def update_ema(self):
        """
        Update EMA UNet parameters using exponential moving average.
        Should be called after each optimizer step during training.
        
        Formula: ema_param = decay * ema_param + (1 - decay) * live_param
        """
        if not self.use_ema or self.unet_ema is None:
            return
        
        with torch.no_grad():
            # Update all parameters in EMA UNet
            for ema_param, live_param in zip(self.unet_ema.parameters(), self.unet.parameters()):
                if live_param.requires_grad:  # Only update if parameter is trainable
                    ema_param.data.mul_(self.ema_decay).add_(live_param.data, alpha=1.0 - self.ema_decay)

    # ------------------------------------------------------------------
    # Config I/O
    # ------------------------------------------------------------------
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
        return cfg

    # ------------------------------------------------------------------
    # Checkpointing (inherited from BaseModel, can override if needed)
    # ------------------------------------------------------------------
    def save_checkpoint(self, path, include_config=True, **extra_state):
        """
        Save diffusion model checkpoint with all components nested.
        
        Ensures decoder, UNet, and scheduler are all included in state_dict,
        even if frozen. All components are nested within the diffusion model.
        """
        path = Path(path)
        state_dict = self.state_dict()
        
        # Verify all components are included
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
        
        # Check if decoder state exists in checkpoint (it should for diffusion checkpoints)
        state_dict = payload.get("state_dict", payload)
        has_decoder_state = any(key.startswith("decoder.") for key in state_dict.keys())
        
        # Use saved config from checkpoint (contains decoder config)
        saved_config = payload.get("config")
        
        # If user provided config, merge it but prefer decoder from saved checkpoint
        if has_decoder_state and saved_config and isinstance(saved_config, dict):
            # Decoder state is in checkpoint - use saved decoder config
            if config and isinstance(config, dict):
                # Merge user config with saved config, but keep decoder from saved config
                merged_config = config.copy()
                
                # Get decoder config from saved checkpoint
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
                    # If user config has autoencoder section, put decoder config there
                    if "autoencoder" in merged_config:
                        # Put decoder config under autoencoder (remove checkpoint path)
                        merged_config["autoencoder"] = {
                            "decoder": saved_decoder_config,
                            "frozen": merged_config["autoencoder"].get("frozen", False)
                        }
                    else:
                        # Put decoder config at top level
                        merged_config["decoder"] = saved_decoder_config
                else:
                    raise ValueError("Saved checkpoint config missing decoder config - checkpoint incomplete!")
                
                # IMPORTANT: Use scheduler from user config (not saved config) to allow changing num_steps
                # The scheduler buffers will be rebuilt to match the new num_steps
                if "scheduler" in config:
                    merged_config["scheduler"] = config["scheduler"]
                model_config = merged_config
            else:
                # No user config - use saved config as-is
                model_config = saved_config
        else:
            # No decoder state in checkpoint (shouldn't happen for diffusion checkpoints)
            # Use provided config or saved config
            model_config = config if config is not None else saved_config
        
        # Build model from config
        model = cls.from_config(model_config) if model_config else cls()

        # Load state dict (restores all component weights including decoder and scheduler with noise_scale)
        model.load_state_dict(payload["state_dict"], strict=False)
        
        # noise_scale and noise_offset are now loaded from scheduler's state_dict automatically

        if return_extra:
            # Return model and any extra state (optimizer, epoch, etc.)
            extra_state = {k: v for k, v in payload.items() 
                          if k not in ["state_dict", "config"]}
            return model, extra_state
        
        return model
