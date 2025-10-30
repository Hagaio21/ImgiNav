import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union, Tuple
import yaml
from tqdm import tqdm


# -----------------------------
#   Base Interfaces
# -----------------------------
class DiffusionBackbone(nn.Module):
    """Abstract backbone interface for diffusion noise prediction models."""
    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


# -----------------------------
#   Latent Diffusion Wrapper
# -----------------------------
class LatentDiffusion(nn.Module):


    def __init__(
        self,
        backbone: DiffusionBackbone,
        scheduler,
        autoencoder: Optional[nn.Module] = None,
        latent_shape: Optional[Tuple[int, int, int]] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.scheduler = scheduler
        self.autoencoder = autoencoder
        self.latent_shape = latent_shape


    
    def forward_step(self, x0: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """
        Single forward step for diffusion training.
        """
        device = x0.device
        B = x0.shape[0]
        t = torch.randint(0, self.scheduler.num_steps, (B,), device=device, dtype=torch.long)

        noise = torch.randn_like(x0)
        x_t = self.scheduler.add_noise(x0, noise, t)
        pred_noise = self.backbone(x_t, t, cond)
        return pred_noise, noise, t, x_t
    
    def forward(self, batch, cfg_dropout_prob: float = 0.0):
        """
        Forward pass for training that extracts data and conditions from batch.
        Args:
            batch: Dict with keys:
                - "layout" or "image" or "x": input tensor (B,C,H,W) or latent embeddings (B,C,H,W)
                - "condition" or "cond" or "embedding": optional condition tensor
            cfg_dropout_prob: Probability of dropping condition for classifier-free guidance
        Returns:
            dict with:
                - "pred_noise": predicted noise
                - "target_noise": ground-truth noise
                - "timesteps": timestep tensor
                - "x_t": noisy latent
                - "pred_x0": predicted clean latent (optional)
        """
        # Extract input data
        x = None
        for key in ["layout", "image", "x", "embedding"]:
            if key in batch:
                x = batch[key]
                break
        
        if x is None:
            raise ValueError("Batch must contain 'layout', 'image', 'x', or 'embedding' key")
        
        # Extract condition
        cond = None
        for key in ["condition", "cond", "embedding_cond"]:
            if key in batch:
                cond = batch[key]
                break
        
        # Classifier-free guidance: randomly drop condition
        if cond is not None and cfg_dropout_prob > 0.0:
            if torch.rand(1).item() < cfg_dropout_prob:
                cond = None
        
        # If using embeddings, x is already in latent space
        # Otherwise, encode to latent space
        if self.autoencoder is not None:
            if x.shape[1] != self.latent_shape[0]:
                # This is RGB, need to encode
                x = self.autoencoder.encode_latent(x, deterministic=True)
        else:
            # No autoencoder, assume x is already latent
            pass
        
        # Forward step
        pred_noise, target_noise, timesteps, x_t = self.forward_step(x, cond)
        
        # Optionally compute pred_x0 for visualization/metrics
        alpha_bar_t = self.scheduler.alpha_bars[timesteps].view(-1, 1, 1, 1)
        pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
        pred_x0 = torch.clamp(pred_x0, -3, 3)
        
        return {
            "pred_noise": pred_noise,
            "target_noise": target_noise,
            "timesteps": timesteps,
            "x_t": x_t,
            "pred_x0": pred_x0,
            "original_latent": x,
            "condition": cond,
        }

    # -------------------------
    #   Sampling
    # -------------------------
    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        image: bool = False,
        cond: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        return_latents: bool = False,
        return_full_history: bool = False,
        method: str = "ddpm",
        eta: float = 0.0, # This will now be used for DDIM
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        guidance_scale: float = 1.0,
        uncond_cond: Optional[torch.Tensor] = None,
        start_noise: Optional[torch.Tensor] = None,
        verbose: bool = True,) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:

        assert self.latent_shape is not None, "latent_shape must be set"

        self.eval()
        self.backbone.eval()
        if self.autoencoder is not None:
            self.autoencoder.eval()

        C, H, W = self.latent_shape
        device = torch.device(device)
        cond = cond.to(device) if cond is not None else None
        uncond_cond = uncond_cond.to(device) if uncond_cond is not None else None

        if start_noise is None:
            x_t = torch.randn(batch_size, C, H, W, device=device)
        else:
            x_t = start_noise.to(device)

        steps = num_steps or self.scheduler.num_steps
        
        if method.lower() == "ddim" and steps <= self.scheduler.num_steps:
            # --- CORRECTED DDIM TIMESTEP GENERATION ---
            # We need a sequence of [T-1, ..., 0] to subsample from
            original_timesteps = torch.arange(self.scheduler.num_steps - 1, -1, -1, dtype=torch.long, device=device)
            
            # Create a uniform subsampling of the original timesteps
            # Example: 1000 steps -> 50 steps: [999, 979, 959, ..., 19, 0]
            if steps < self.scheduler.num_steps:
                indices = torch.linspace(0, self.scheduler.num_steps - 1, steps, dtype=torch.long, device=device)
                timesteps = original_timesteps[indices]
            else:
                timesteps = original_timesteps
            
            # Add a -1 tensor to the end to signify the last step
            timesteps = torch.cat([timesteps, torch.tensor([-1], device=device, dtype=torch.long)])
            # Now timesteps are [t_i, t_{i-1}, ..., t_0, -1]
        else:
            # Use linear spacing for DDPM
            timesteps = torch.linspace(steps - 1, 0, steps, dtype=torch.long, device=device)
            # Add a -1 tensor to the end
            timesteps = torch.cat([timesteps, torch.tensor([-1], device=device, dtype=torch.long)])

        
        iterator = tqdm(timesteps[:-1], desc="Diffusion sampling", total=len(timesteps) - 1) if verbose else timesteps[:-1]

        noise_history, latents_history = [], []

        if verbose:
            print(f"Sampling method: {method.upper()} ({steps} steps)")

        for i, t in enumerate(iterator):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # --- Get prev timestep ---
            # t is the current timestep (e.g., t_i)
            # We need the previous timestep (e.g., t_{i-1})
            t_prev_val = timesteps[i+1].item()
            t_prev_batch = torch.full((batch_size,), t_prev_val, device=device, dtype=torch.long)
            # --------------------------

            # guidance
            if guidance_scale == 1.0 or uncond_cond is None:
                noise_pred = self.backbone(x_t, t_batch, cond)
            else:
                noise_pred_cond = self.backbone(x_t, t_batch, cond)
                noise_pred_uncond = self.backbone(x_t, t_batch, uncond_cond)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            if return_full_history:
                noise_history.append(noise_pred.clone())

            # --- Get scheduler params for t ---
            alpha_t = self.scheduler.alphas[t_batch].view(-1, 1, 1, 1)
            alpha_bar_t = self.scheduler.alpha_bars[t_batch].view(-1, 1, 1, 1)
            beta_t = self.scheduler.betas[t_batch].view(-1, 1, 1, 1)
            
            # --- REMOVED old alpha_bar_prev calculation ---

            # predict x0
            pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            
            # --- START: MODIFIED DDIM/DDPM BLOCKS ---
            if method.lower() == "ddim":
                
                if t_prev_val < 0:
                    # This is the last step (t_prev = -1)
                    x_t = pred_x0
                else:
                    # --- Get alpha_bar_prev using the CORRECT t_prev_batch ---
                    alpha_bar_prev = self.scheduler.alpha_bars[t_prev_batch].view(-1, 1, 1, 1)
                    
                    # --- Full DDIM update (with eta) ---
                    # 1. Coefficient for pred_x0
                    pred_x0_coeff = torch.sqrt(alpha_bar_prev) * pred_x0
                    
                    # 2. Calculate sigma_t
                    # sigma_t^2 = eta^2 * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
                    ratio = alpha_bar_t / alpha_bar_prev
                    variance = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - ratio)
                    
                    # Clamp variance to avoid numerical instability
                    sigma_t_sq = torch.clamp(eta**2 * variance, min=0.0)
                    sigma_t = torch.sqrt(sigma_t_sq)

                    # 3. Coefficient for "direction to x_t" (noise_pred)
                    pred_dir_coeff = torch.sqrt(1 - alpha_bar_prev - sigma_t_sq)
                    pred_dir_coeff = torch.clamp(pred_dir_coeff, min=0.0) # Ensure non-negative
                    pred_dir_xt = pred_dir_coeff * noise_pred # This is the "direction" component

                    # 4. Stochastic noise
                    noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)
                    
                    x_t = pred_x0_coeff + pred_dir_xt + sigma_t * noise
            
            else:
                # --- Corrected DDPM stochastic update ---
                
                if t_prev_val < 0:
                    # Last step
                    x_t = pred_x0
                else:
                    # --- Get alpha_bar_prev using t-1 (which is t_prev_val for DDPM) ---
                    alpha_bar_prev = self.scheduler.alpha_bars[t_prev_batch].view(-1, 1, 1, 1)
                
                    mean = (1 / torch.sqrt(alpha_t)) * (
                        x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred
                    )
                    variance = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                    noise = torch.randn_like(x_t)
                    x_t = mean + torch.sqrt(variance) * noise
            # --- END: MODIFIED DDIM/DDPM BLOCKS ---

            if return_full_history:
                latents_history.append(x_t.clone())

        if image:
            assert self.autoencoder is not None, "AutoEncoder required for image decoding"
            if verbose:
                print("Decoding...")
            rgb_out, _ = self.autoencoder.decoder(x_t)
            x_t = rgb_out

        if return_latents or return_full_history:
            history = {"latents": latents_history, "noise": noise_history}
            return x_t, history

        return x_t
        
    # -------------------------
    #   Save / Load Configs
    # -------------------------
    def to_config(self, save_path: Union[str, Path]):
        ckpt_dir = Path(save_path).parent

        config = {
            "latent": {
                "channels": self.latent_shape[0],
                "base": self.latent_shape[1],
            },
            "scheduler": {
                "type": type(self.scheduler).__name__,
                "num_steps": getattr(self.scheduler, "num_steps", None),
            },
            "autoencoder": {
                "config": str((ckpt_dir / "autoencoder_config.yaml").as_posix()),
                "checkpoint": str((ckpt_dir / "ae_latest.pt").as_posix()),
            },
            "unet": {
                "config": str((ckpt_dir / "unet_config.yaml").as_posix()),
                "checkpoint": str((ckpt_dir / "unet_latest.pt").as_posix()),
            },
        }

        with open(save_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f)
        print(f"[Config] LatentDiffusion saved → {save_path}")

    def to(self, device):
        self.backbone.to(device)
        self.scheduler.to(device)
        if self.autoencoder is not None:
            self.autoencoder.to(device)
        return self
    
    def state_dict(self, trainable_only: bool = True):

        if trainable_only:
            # Only save the trainable backbone
            return self.backbone.state_dict()
        else:
            # Save all components (for full model checkpoint)
            return {
                "backbone": self.backbone.state_dict(),
                "scheduler": self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else {},
                "autoencoder": self.autoencoder.state_dict() if self.autoencoder is not None else {},
            }

    @classmethod
    def from_config(
        cls,
        config: Union[str, Path, dict],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):

        import importlib
        
        # Backward compatibility: accept string path or dict
        if isinstance(config, (str, Path)):
            with open(config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        elif isinstance(config, dict):
            cfg = config
        else:
            raise TypeError(f"config must be str, Path, or dict, got {type(config)}")

        # Scheduler
        from .components.scheduler import LinearScheduler, CosineScheduler
        sched_cfg = cfg["scheduler"]
        sched_map = {"LinearScheduler": LinearScheduler, "CosineScheduler": CosineScheduler}
        scheduler = sched_map[sched_cfg["type"]](num_steps=sched_cfg["num_steps"]).to(device)

        # Autoencoder (optional)
        autoencoder = None
        if "autoencoder" in cfg:
            ae_cfg = cfg["autoencoder"].get("config")
            ae_ckpt = cfg["autoencoder"].get("checkpoint")
            try:
                # Try to import AutoEncoder from models.autoencoder (new) or modules.autoencoder (old)
                try:
                    from models.autoencoder import AutoEncoder
                except ImportError:
                    AutoEncoder = importlib.import_module("modules.autoencoder").AutoEncoder
                
                autoencoder = AutoEncoder.from_config(ae_cfg).to(device)
                if ae_ckpt and Path(ae_ckpt).exists():
                    ae_state = torch.load(ae_ckpt, map_location=device)
                    autoencoder.load_state_dict(ae_state.get("model", ae_state), strict=False)
            except Exception as e:
                print(f"[Warning] Autoencoder loading failed: {e}")
                pass

        # Backbone / UNet
        unet_cfg = cfg.get("unet", {}).get("config")
        unet_ckpt = cfg.get("unet", {}).get("checkpoint")
        try:
            # Try to import DualUNet from models.components.unet (new) or modules.unet (old)
            try:
                from models.components.unet import DualUNet as UNet
            except ImportError:
                UNet = importlib.import_module("modules.unet").UNet
            
            backbone = UNet.from_config(unet_cfg).to(device)
            if unet_ckpt and Path(unet_ckpt).exists():
                state = torch.load(unet_ckpt, map_location=device)
                if "state_dict" in state:
                    state = state["state_dict"]
                backbone.load_state_dict(state, strict=False)
        except Exception as e:
            raise RuntimeError(f"Backbone import failed: {e}. Check your config and module path.")

        latent = cfg["latent"]
        latent_shape = (latent["channels"], latent["base"], latent["base"])

        return cls(backbone=backbone, scheduler=scheduler,
                   autoencoder=autoencoder, latent_shape=latent_shape).to(device)
