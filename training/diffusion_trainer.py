# diffusion_trainer.py
import os
import torch
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image
from diffusion import LatentDiffusion
import matplotlib.pyplot as plt

class DiffusionTrainer:
    def __init__(self,
                 unet,
                 autoencoder,
                 scheduler,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 optimizer=None,
                 loss_fn=None,
                 epochs=10,
                 log_interval=100,
                 eval_interval=1000,
                 sample_interval=2000,
                 grad_clip=None,
                 logger=None,
                 ckpt_dir=None,
                 output_dir=None):
        self.unet = unet.to(device)
        self.autoencoder = autoencoder.eval().to(device)
        self.scheduler = scheduler
        self.device = device
        self.logger = logger

        self.loss_fn = loss_fn or torch.nn.MSELoss()
        self.optimizer = optimizer or AdamW(self.unet.parameters(), lr=1e-4)
        self.epochs = epochs
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.sample_interval = sample_interval
        self.grad_clip = grad_clip
        self.ckpt_dir = ckpt_dir
        self.output_dir = output_dir or "samples"
        self.global_step = 0

        os.makedirs(self.output_dir, exist_ok=True)
        if self.ckpt_dir:
            os.makedirs(self.ckpt_dir, exist_ok=True)

        # Build a diffusion object once for sampling
        latent_shape = (
            self.autoencoder.encoder.latent_channels,
            self.autoencoder.encoder.latent_base,
            self.autoencoder.encoder.latent_base,
        )
        self.diffusion = LatentDiffusion(
            self.unet, self.scheduler, self.autoencoder, latent_shape=latent_shape
        )

    def to(self, device):
        self.device = device
        self.unet.to(device)
        self.autoencoder.to(device)
        self.diffusion.to(device)
        return self

    def train(self, mode=True):
        self.unet.train(mode)
        self.autoencoder.eval()
        return self

    def fit(self, train_loader, val_loader=None):
        self.train(True)

        for epoch in range(self.epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)
            for batch in pbar:
                self.optimizer.zero_grad(set_to_none=True)

                layout = batch[0].to(self.device)
                z = self.autoencoder.encoder(layout)
                t = torch.randint(0, self.scheduler.num_steps, (z.size(0),), device=self.device)
                noise = torch.randn_like(z)
                z_noisy = self.scheduler.add_noise(z, noise, t)

                noise_pred = self.unet(z_noisy, t)
                loss = self.loss_fn(noise_pred, noise)

                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.grad_clip)
                self.optimizer.step()
                self.global_step += 1

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                if self.logger and self.global_step % self.log_interval == 0:
                    self.logger.log({"loss": loss.item(), "epoch": epoch, "step": self.global_step})

                if val_loader and self.global_step % self.eval_interval == 0:
                    self.evaluate(val_loader, step=self.global_step)
                    self.train(True)

                if self.global_step % self.sample_interval == 0:
                    self.sample_and_save(self.global_step)

            if self.ckpt_dir:
                self.save_checkpoint(f"{self.ckpt_dir}/epoch_{epoch+1}.pt")

    @torch.no_grad()
    def evaluate(self, val_loader, step=None):
        self.train(False)
        layout = next(iter(val_loader))[0].to(self.device)
        z = self.autoencoder.encoder(layout)
        t = torch.randint(0, self.scheduler.num_steps, (z.size(0),), device=self.device)
        noise = torch.randn_like(z)
        z_noisy = self.scheduler.add_noise(z, noise, t)

        noise_pred = self.unet(z_noisy, t)
        loss = self.loss_fn(noise_pred, noise).item()

        if self.logger and step is not None:
            self.logger.log({"val_loss": loss, "val_step": step}, step=step)

        return {"val_loss": loss}

    @torch.no_grad()
    def sample_and_save(self, step, batch_size=4):
        self.unet.eval()
        samples = self.diffusion.sample(batch_size=batch_size, image=True)
        save_path = os.path.join(self.output_dir, f"samples_step_{step}.png")
        save_image(samples, save_path, nrow=2, normalize=True, value_range=(0, 1))
        if self.logger:
            self.logger.log({"samples": samples, "step": step})
        print(f"[Step {step}] Saved sample grid → {save_path}")

    def save_checkpoint(self, path):
        state = {
            "unet": self.unet.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,
        }
        torch.save(state, path)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device)
        self.unet.load_state_dict(state["unet"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.global_step = state["step"]
        print(f"Checkpoint loaded from {path} at step {self.global_step}")

    @torch.no_grad()
    def log_metrics(self, loss, z_noisy, noise_pred, noise, step, autoencoder=None, dataset_div=None):
        """Compute and log lightweight metrics every N steps."""
        log = {}

        # --- core ---
        log["step"] = step
        log["loss"] = loss.item()

        # --- SNR ---
        snr = (z_noisy.var(dim=(1,2,3)) / (noise_pred - noise).var(dim=(1,2,3))).mean().item()
        log["snr"] = snr

        # --- cosine similarity ---
        cos = F.cosine_similarity(noise_pred.flatten(1), noise.flatten(1)).mean().item()
        log["cosine"] = cos

        # --- grad norm ---
        norms = [p.grad.norm()**2 for p in self.unet.parameters() if p.grad is not None]
        grad_norm = torch.sqrt(sum(norms)).item() if norms else 0.0
        log["grad_norm"] = grad_norm

        # --- diversity ---
        samples = self.diffusion.sample(batch_size=4, image=True)
        decoded = autoencoder.decoder(samples) if autoencoder else samples
        flat = decoded.flatten(1)
        div = torch.cdist(flat, flat).mean().item()
        log["diversity"] = div

        if dataset_div:
            log["div_ratio"] = div / dataset_div

        # --- store or print ---
        if not hasattr(self, "metric_log"):
            self.metric_log = []
        self.metric_log.append(log)

        msg = f"[{step}] loss={log['loss']:.4f}, snr={log['snr']:.3f}, cos={log['cosine']:.3f}, div={log['diversity']:.3f}"
        print(msg)
        return log


    @torch.no_grad()
    def create_viz_artifacts(self, autoencoder, samples, step, losses, output_dir):
        """Create plots and sample grids every M steps."""
        os.makedirs(output_dir, exist_ok=True)

        # --- loss curve ---
        if losses:
            plt.figure()
            plt.plot(losses)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"loss_curve_step_{step}.png"))
            plt.close()

        # --- diversity curve ---
        if hasattr(self, "metric_log"):
            divs = [x["diversity"] for x in self.metric_log]
            steps = [x["step"] for x in self.metric_log]
            plt.figure()
            plt.plot(steps, divs)
            plt.xlabel("Step")
            plt.ylabel("Diversity")
            plt.title("Diversity over Training")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"diversity_curve_step_{step}.png"))
            plt.close()

        # --- sample grid ---
        decoded = autoencoder.decoder(samples)
        save_image(decoded, os.path.join(output_dir, f"samples_step_{step}.png"),
                nrow=2, normalize=True, value_range=(0, 1))

