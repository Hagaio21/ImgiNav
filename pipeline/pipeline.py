# pipeline.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from autoencoder import AutoEncoder
from diffusion import LatentDiffusion
from unet import UNet
from condition_mixer import BaseMixer


# --- metrics helpers ---
def compute_condition_correlation(pred, conds):
    result = {}
    flat_pred = pred.flatten(1)
    for name, c in zip(["pov", "graph"], conds):
        if c is None:
            continue
        flat_cond = c.flatten(1)
        sim = F.cosine_similarity(flat_pred, flat_cond, dim=1)
        result[name] = sim.mean().item()
    return result


def safe_grad_norm(model):
    norms = []
    for p in model.parameters():
        if p.grad is not None:
            norms.append(p.grad.norm() ** 2)
    if not norms:
        return 0.0
    return torch.sqrt(sum(norms)).item()


def compute_psnr(x, y):
    mse = F.mse_loss(x, y)
    return -10 * torch.log10(mse + 1e-8).item()


def compute_diversity(samples):
    flat = samples.flatten(1)
    dist = torch.cdist(flat, flat)
    return dist.mean().item()


# placeholder FID proxy
def compute_fid(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    cov1, cov2 = torch.cov(a.T), torch.cov(b.T)
    diff = (mu1 - mu2).pow(2).sum().sqrt()
    cov_diff = torch.trace(cov1 + cov2 - 2 * (cov1 @ cov2).sqrt())
    return (diff + cov_diff).item()


# --- main pipeline ---
class DiffusionPipeline(nn.Module):
    def __init__(self,
                 autoencoder: AutoEncoder,
                 unet: UNet,
                 mixer: BaseMixer,
                 scheduler,
                 embedder_manager=None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 logger=None):
        super().__init__()
        self.device = device
        self.autoencoder = autoencoder.eval().to(device)
        self.unet = unet.to(device)
        self.mixer = mixer.to(device)
        self.scheduler = scheduler
        self.embedder_manager = embedder_manager
        self.logger = logger
        self.diffusion = None
        self._build_diffusion()

    def _build_diffusion(self):
        latent_shape = (
            self.autoencoder.encoder.latent_channels,
            self.autoencoder.encoder.latent_base,
            self.autoencoder.encoder.latent_base,
        )
        self.diffusion = LatentDiffusion(
            self.unet, self.scheduler, self.autoencoder, latent_shape=latent_shape
        )

    def encode_layout(self, layout):
        with torch.no_grad():
            return self.autoencoder.encode_latent(layout.to(self.device))

    def _prepare_conditions(self, pov_raw=None, graph_raw=None, 
                           cond_pov_emb=None, cond_graph_emb=None):
        """
        Convert raw inputs to embeddings if needed.
        """
        # If embeddings already provided, use them directly:
        if cond_pov_emb is not None or cond_graph_emb is not None:
            return cond_pov_emb, cond_graph_emb
        
        pov_emb = None
        graph_emb = None
        
        if pov_raw is not None and self.embedder_manager is not None:
            # Handle single image
            if isinstance(pov_raw, Image.Image):
                pov_emb = self.embedder_manager.embed_pov(pov_raw).unsqueeze(0)
            # Handle tensor (single or batch)
            elif isinstance(pov_raw, torch.Tensor):
                if len(pov_raw.shape) == 3:  # Single [C,H,W]
                    # Convert to PIL for embedder
                    from torchvision.transforms import ToPILImage
                    pov_pil = ToPILImage()(pov_raw.cpu())
                    pov_emb = self.embedder_manager.embed_pov(pov_pil).unsqueeze(0)
                else:  # Batch [B,C,H,W]
                    pov_embs = []
                    from torchvision.transforms import ToPILImage
                    to_pil = ToPILImage()
                    for i in range(pov_raw.size(0)):
                        pov_pil = to_pil(pov_raw[i].cpu())
                        pov_embs.append(self.embedder_manager.embed_pov(pov_pil))
                    pov_emb = torch.stack(pov_embs)
            # Handle list of images
            elif isinstance(pov_raw, (list, tuple)):
                pov_emb = torch.stack([
                    self.embedder_manager.embed_pov(img) for img in pov_raw
                ])
        
        if graph_raw is not None and self.embedder_manager is not None:
            if isinstance(graph_raw, str):
                # Single path
                graph_emb = self.embedder_manager.embed_graph(graph_raw).unsqueeze(0)
            elif isinstance(graph_raw, (list, tuple)):
                # Batch of paths
                graph_emb = torch.stack([
                    self.embedder_manager.embed_graph(path) for path in graph_raw
                ])
        
        if pov_emb is not None:
            pov_emb = pov_emb.to(self.device)
        if graph_emb is not None:
            graph_emb = graph_emb.to(self.device)
        
        return pov_emb, graph_emb

    def forward(self, latents: torch.Tensor,
                cond_pov: torch.Tensor | None,
                cond_graph: torch.Tensor | None,
                timesteps: torch.Tensor):
        cond = self.mixer([cond_pov, cond_graph])
        noise_pred = self.unet(latents, timesteps, cond)
        return noise_pred

    def training_step(self, batch, loss_fn, step=None):
        """
        Training expects pre-computed embeddings for efficiency.
        """
        layout = batch["layout"]
        cond_pov_emb = batch["pov"]
        cond_graph_emb = batch["graph"]
        
        z = self.encode_layout(layout)
        t = torch.randint(0, self.scheduler.num_steps, (z.size(0),), device=self.device)
        noise = torch.randn_like(z)
        z_noisy = self.scheduler.add_noise(z, noise, t)

        cond = self.mixer([cond_pov_emb, cond_graph_emb])
        noise_pred = self.unet(z_noisy, t, cond)

        loss = loss_fn(noise_pred, noise)

        if self.logger is not None and step is not None:
            log = {"loss": loss.item(), "timestep": t.float().mean().item()}

            # correlation and cosine
            corr = compute_condition_correlation(noise_pred.detach(), [cond_pov_emb, cond_graph_emb])
            for k, v in corr.items():
                log[f"corr_{k}"] = v
            cos = F.cosine_similarity(noise_pred.flatten(1), noise.flatten(1)).mean().item()
            log["cosine_pred_true"] = cos

            # SNR
            snr = (z_noisy.var(dim=(1,2,3)) / (noise_pred - noise).var(dim=(1,2,3))).mean().item()
            log["snr"] = snr

            # grad norm
            log["grad_norm"] = safe_grad_norm(self.unet)

            self.logger.log(log, step=step)

        return loss

    def sample(self, batch_size: int, 
               pov_raw=None, graph_raw=None,           
               cond_pov_emb=None, cond_graph_emb=None, 
               image=True, step=None, noise=None,
               guidance_scale=1.0,
               num_steps=None):
        """
        Sample from the model.
        
        Args:
            batch_size: Number of samples to generate
            pov_raw: Raw POV image(s) - PIL Image, tensor, or list
            graph_raw: Raw graph path(s) - string or list of strings
            cond_pov_emb: Pre-computed POV embedding (for efficiency during training)
            cond_graph_emb: Pre-computed graph embedding (for efficiency during training)
            image: If True, decode latents to images
            step: Current training step (for logging)
            noise: Optional fixed noise for reproducibility
            guidance_scale: Classifier-free guidance scale (1.0 = no guidance)
            num_steps: Optional override for number of diffusion steps
        """
        if self.diffusion is None:
            self._build_diffusion()

        # Step 1: Embed raw inputs if provided
        cond_pov, cond_graph = self._prepare_conditions(
            pov_raw, graph_raw, cond_pov_emb, cond_graph_emb
        )
        
        # Step 2: Mix conditions into the format expected by UNet
        cond = self.mixer([cond_pov, cond_graph]) if (cond_pov is not None or cond_graph is not None) else None
        
        # Step 3: Prepare unconditional embedding for classifier-free guidance
        uncond_cond = None
        if guidance_scale != 1.0 and cond is not None:
            # Create unconditional embedding (all zeros or null conditions)
            uncond_cond = self.mixer([None, None])
        
        # Step 4: Pass to diffusion denoising loop
        samples = self.diffusion.sample(
            batch_size=batch_size, 
            image=image,
            cond=cond,
            num_steps=num_steps,
            device=self.device,
            guidance_scale=guidance_scale,
            uncond_cond=uncond_cond,
            start_noise=noise
        )

        if self.logger is not None and step is not None:
            self.logger.log({"sample_preview": samples}, step=step)

        return samples

    def to(self, device):
        """Move all submodules to target device."""
        self.device = device
        self.autoencoder.to(device)
        self.unet.to(device)
        self.mixer.to(device)
        if self.embedder_manager is not None:
            # Move embedder models if needed
            if hasattr(self.embedder_manager, 'resnet'):
                self.embedder_manager.resnet.to(device)
            if hasattr(self.embedder_manager, 'graph_encoder'):
                self.embedder_manager.graph_encoder.to(device)
            self.embedder_manager.device = device
        if self.diffusion is not None:
            self.diffusion.to(device)
        return self

    def train(self, mode=True):
        """Set train/eval mode for trainable modules only."""
        self.unet.train(mode)
        self.mixer.train(mode)
        self.autoencoder.eval()  # always frozen
        if self.embedder_manager is not None:
            # Embedders always in eval mode
            if hasattr(self.embedder_manager, 'resnet'):
                self.embedder_manager.resnet.eval()
            if hasattr(self.embedder_manager, 'graph_encoder'):
                self.embedder_manager.graph_encoder.eval()
        return self

    def eval(self):
        """Convenience alias."""
        return self.train(False)
        
    @torch.no_grad()
    def evaluate(self, val_batch, num_samples=8, step=None):
        """
        Evaluation expects pre-computed embeddings.
        """
        layout = val_batch["layout"]
        cond_pov_emb = val_batch["pov"]
        cond_graph_emb = val_batch["graph"]
        
        samples = self.sample(num_samples, cond_pov_emb=cond_pov_emb, 
                            cond_graph_emb=cond_graph_emb, image=True)
        recon = self.autoencoder(layout.to(self.device))

        metrics = {
            "psnr": compute_psnr(recon, layout),
            "diversity": compute_diversity(samples),
            "latent_fid": compute_fid(
                self.encode_layout(layout), self.encode_layout(samples)
            ),
        }

        if self.logger is not None and step is not None:
            self.logger.log(metrics, step=step)
            self.logger.log({"eval_samples": samples}, step=step)

        return metrics