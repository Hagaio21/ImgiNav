# pipeline.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from modules.autoencoder import AutoEncoder
from modules.diffusion import LatentDiffusion
from modules.unet import UNet
from modules.condition_mixer import BaseMixer
from torch.linalg import matrix_power

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


def compute_fid(a, b):
    # --- FIX: Flatten from [B, C, H, W] to [B, D_flat] ---
    a_flat = a.flatten(1)
    b_flat = b.flatten(1)
    
    # Calculate means (shape [D_flat])
    mu1 = a_flat.mean(dim=0)
    mu2 = b_flat.mean(dim=0)
    
    # Calculate covariance matrices (shape [D_flat, D_flat])
    # torch.cov expects [D, N], so we transpose [B, D] -> [D, B]
    cov1 = torch.cov(a_flat.T)
    cov2 = torch.cov(b_flat.T)
    
    # L2 norm of mean difference
    diff = (mu1 - mu2).pow(2).sum().sqrt() 
    
    try:
        # --- FIX: Use proper matrix square root ---
        # Calculate the trace term: Tr(cov1 + cov2 - 2 * (cov1 @ cov2)^(1/2))
        cov_prod_sqrt = matrix_power(cov1 @ cov2, 0.5)
        
        # The result can be complex, we need the real part of the trace
        cov_diff = torch.trace(cov1 + cov2 - 2 * cov_prod_sqrt).real
    except Exception as e:
        # Fallback if linalg fails (e.g., singular matrix)
        print(f"Warning: FID covariance calculation failed: {e}. Returning mean diff only.")
        cov_diff = torch.tensor(0.0, device=a.device)

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
        cond = self.mixer([cond_pov, cond_graph], B_hint=latents.shape[0], device_hint=latents.device)
        noise_pred = self.unet(latents, timesteps, cond)
        return noise_pred

    def sample(self, batch_size: int, 
               pov_raw=None, graph_raw=None,           
               cond_pov_emb=None, cond_graph_emb=None, 
               image=True, step=None, noise=None,
               guidance_scale=1.0,
               num_steps=None):

        if self.diffusion is None:
            self._build_diffusion()

        # Step 1: Embed raw inputs if provided
        cond_pov, cond_graph = self._prepare_conditions(
            pov_raw, graph_raw, cond_pov_emb, cond_graph_emb
        )
        
        # Step 2: Mix conditions into the format expected by UNet
        cond = self.mixer([cond_pov, cond_graph], B_hint=batch_size, device_hint=self.device)
        
        # Step 3: Prepare unconditional embedding for classifier-free guidance
        uncond_cond = None
        if guidance_scale != 1.0 and cond is not None:
            # Create unconditional embedding (all zeros or null conditions)
            uncond_cond = self.mixer([None, None], B_hint=batch_size, device_hint=self.device)
        
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
        Evaluation expects a raw batch (not pre-computed embeddings).
        """
        layout = val_batch["layout"]
        
        # --- START MODIFICATION ---
        # Get raw data from batch
        pov_raw = val_batch["pov"]
        graph_raw = val_batch["graph"]
        
        # Convert raw inputs to embeddings
        cond_pov_emb, cond_graph_emb = self._prepare_conditions(
            pov_raw, graph_raw
        )
        # --- END MODIFICATION ---
        
        # --- FIX: Resize input batch to match num_samples ---
        B_in = layout.size(0)
        
        if B_in != num_samples:
            if B_in < num_samples:
                # Repeat conditions to match num_samples
                ratio = (num_samples + B_in - 1) // B_in
                layout = layout.repeat(ratio, 1, 1, 1)[:num_samples]
                if cond_pov_emb is not None:
                    # Handle 2D embedding tensor [B, C]
                    repeat_dims = [ratio] + [1] * (cond_pov_emb.dim() - 1)
                    cond_pov_emb = cond_pov_emb.repeat(*repeat_dims)[:num_samples]
                if cond_graph_emb is not None:
                    # Handle 2D embedding tensor [B, C]
                    repeat_dims = [ratio] + [1] * (cond_graph_emb.dim() - 1)
                    cond_graph_emb = cond_graph_emb.repeat(*repeat_dims)[:num_samples]
            else: # B_in > num_samples
                # Truncate conditions
                layout = layout[:num_samples]
                if cond_pov_emb is not None:
                    cond_pov_emb = cond_pov_emb[:num_samples]
                if cond_graph_emb is not None:
                    cond_graph_emb = cond_graph_emb[:num_samples]
        # --- End Fix ---

        samples = self.sample(num_samples, cond_pov_emb=cond_pov_emb, 
                            cond_graph_emb=cond_graph_emb, image=True)
        
        # Use the (potentially resized) layout tensor for recon and FID
        recon_output = self.autoencoder(layout.to(self.device))
        recon = recon_output[0] if isinstance(recon_output, tuple) else recon_output

        metrics = {
            "psnr": compute_psnr(recon, layout.to(self.device)), # ensure layout is on device
            "diversity": compute_diversity(samples),
            "latent_fid": compute_fid(
                self.encode_layout(layout), self.encode_layout(samples)
            ),
        }

        if self.logger is not None and step is not None:
            self.logger.log(metrics, step=step)
            self.logger.log({"eval_samples": samples}, step=step)

        return metrics