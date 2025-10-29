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


class LatentDecoder(nn.Module):
    """Abstract interface for latent-to-image decoders."""
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# -----------------------------
#   Latent Diffusion Wrapper
# -----------------------------
class LatentDiffusion(nn.Module):
    """
    Latent diffusion wrapper for inference and sampling.

    Compatible with any backbone (UNet, Transformer, etc.)
    and any autoencoder-style decoder.
    """

    def __init__(
        self,
        backbone: DiffusionBackbone,
        scheduler,
        autoencoder: Optional[LatentDecoder] = None,
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
        eta: float = 0.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        guidance_scale: float = 1.0,
        uncond_cond: Optional[torch.Tensor] = None,
        start_noise: Optional[torch.Tensor] = None,
        verbose: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:

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
        timesteps = torch.linspace(steps - 1, 0, steps, dtype=torch.long, device=device)
        iterator = tqdm(timesteps, desc="Diffusion sampling", total=len(timesteps)) if verbose else timesteps

        noise_history, latents_history = [], []

        if verbose:
            print(f"Sampling method: {method.upper()} ({steps} steps)")

        for t in iterator:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # guidance
            if guidance_scale == 1.0 or uncond_cond is None:
                noise_pred = self.backbone(x_t, t_batch, cond)
            else:
                noise_pred_cond = self.backbone(x_t, t_batch, cond)
                noise_pred_uncond = self.backbone(x_t, t_batch, uncond_cond)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            if return_full_history:
                noise_history.append(noise_pred.clone())

            alpha_t = self.scheduler.alphas[t]
            alpha_bar_t = self.scheduler.alpha_bars[t]
            beta_t = self.scheduler.betas[t]

            if t > 0:
                alpha_bar_prev = self.scheduler.alpha_bars[t - 1]

                # predict x0
                pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
                pred_x0 = torch.clamp(pred_x0, -3, 3)

                if method.lower() == "ddim":
                    # deterministic DDIM update
                    sqrt_ab_prev = torch.sqrt(alpha_bar_prev)
                    sqrt_ab_t = torch.sqrt(alpha_bar_t)
                    x_t = sqrt_ab_prev * pred_x0 + torch.sqrt(1 - alpha_bar_prev - eta ** 2) * (
                        x_t - sqrt_ab_t * pred_x0
                    ) / torch.sqrt(1 - alpha_bar_t)
                else:
                    # stochastic DDPM update
                    coef1 = torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t)
                    coef2 = torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                    mean = coef1 * pred_x0 + coef2 * x_t
                    variance = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t
                    noise = torch.randn_like(x_t)
                    x_t = mean + torch.sqrt(variance) * noise

            else:
                # final denoising step
                x_t = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

            if return_full_history:
                latents_history.append(x_t.clone())

        if image:
            assert self.autoencoder is not None, "AutoEncoder required for image decoding"
            if verbose:
                print("Decoding...")
            x_t = self.autoencoder.decode_latent(x_t)

        if return_latents or return_full_history:
            history = {"latents": latents_history, "noise": noise_history}
            return x_t, history

        return x_t

    @torch.no_grad()
    def training_sample(self, batch_size: int = 8, cond: Optional[torch.Tensor] = None, 
                       device=None, num_steps: Optional[int] = None):
        """
        Generate samples during training for visualization.
        Uses the model's sample() method internally.
        Args:
            batch_size: Number of samples to generate
            cond: Optional condition tensor
            device: Device to generate on
            num_steps: Number of diffusion steps (uses scheduler default if None)
        Returns:
            Decoded RGB images (B, C, H, W)
        """
        if device is None:
            device = next(self.backbone.parameters()).device
        
        # Generate samples using the inference sample() method
        samples = self.sample(
            batch_size=batch_size,
            image=True,  # Decode to images
            cond=cond,
            num_steps=num_steps,
            device=device,
            verbose=False,
        )
        return samples

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

    # -------------------------
    #   Config Loader
    # -------------------------
    @classmethod
    def from_config(
        cls,
        config: Union[str, Path, dict],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Load LatentDiffusion from config.
        Args:
            config: Either a file path (str/Path) or config dict
            device: Device to load models to
        """
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
