# pipeline.py
import torch
import torch.nn as nn
import torch.nn.functional as F

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
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 logger=None):
        super().__init__()
        self.device = device
        self.autoencoder = autoencoder.eval().to(device)
        self.unet = unet.to(device)
        self.mixer = mixer.to(device)
        self.scheduler = scheduler
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



    def forward(self, latents: torch.Tensor,
                cond_pov: torch.Tensor | None,
                cond_graph: torch.Tensor | None,
                timesteps: torch.Tensor):
        cond = self.mixer([cond_pov, cond_graph])
        noise_pred = self.unet(latents, timesteps, cond)
        return noise_pred

    def training_step(self, batch, loss_fn, step=None):
        layout, cond_pov, cond_graph = batch
        z = self.encode_layout(layout)
        t = torch.randint(0, self.scheduler.num_steps, (z.size(0),), device=self.device)
        noise = torch.randn_like(z)
        z_noisy = self.scheduler.add_noise(z, noise, t)

        cond = self.mixer([cond_pov, cond_graph])
        noise_pred = self.unet(z_noisy, t, cond)

        loss = loss_fn(noise_pred, noise)

        if self.logger is not None and step is not None:
            log = {"loss": loss.item(), "timestep": t.float().mean().item()}

            # correlation and cosine
            corr = compute_condition_correlation(noise_pred.detach(), [cond_pov, cond_graph])
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

    def sample(self, batch_size: int, cond_pov=None, cond_graph=None, image=True, step=None):
        if self.diffusion is None:
            self._build_diffusion()

        cond = self.mixer([cond_pov, cond_graph]) if (cond_pov is not None or cond_graph is not None) else None
        samples = self.diffusion.sample(batch_size=batch_size, cond=cond, image=image)

        if self.logger is not None and step is not None:
            self.logger.log({"sample_preview": samples}, step=step)

        return samples
    
    def to(self, device):
        """Move all submodules to target device."""
        self.device = device
        self.autoencoder.to(device)
        self.unet.to(device)
        self.mixer.to(device)
        if self.diffusion is not None:
            self.diffusion.to(device)
        return self

    def train(self, mode=True):
        """Set train/eval mode for trainable modules only."""
        self.unet.train(mode)
        self.mixer.train(mode)
        self.autoencoder.eval()  # always frozen
        return self

    def eval(self):
        """Convenience alias."""
        return self.train(False)
        
    @torch.no_grad()
    def evaluate(self, val_batch, num_samples=8, step=None):
        layout, cond_pov, cond_graph = val_batch
        samples = self.sample(num_samples, cond_pov, cond_graph, image=True)
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
