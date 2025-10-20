# diffusion_trainer.py
import os
import torch
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image
from modules.diffusion import LatentDiffusion
import matplotlib.pyplot as plt
import json


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
                 num_samples=8,
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
        self.num_samples = num_samples
        self.output_dir = output_dir or "samples"
        self.global_step = 0

        os.makedirs(self.output_dir, exist_ok=True)
        if self.ckpt_dir:
            os.makedirs(self.ckpt_dir, exist_ok=True)

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

    # ============================================================
    # Main training loop with validation logging
    # ============================================================
    def fit(self, train_loader, val_loader=None):
        self.train(True)

        self.dataset_div = self._compute_dataset_diversity(train_loader)
        print(f"Dataset diversity baseline: {self.dataset_div:.4f}")

        losses = []
        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)
            for batch in pbar:
                self.optimizer.zero_grad(set_to_none=True)

                layout = batch[0].to(self.device)
                z = self.autoencoder.encoder(layout)
                t = torch.randint(0, self.scheduler.num_steps, (z.size(0),), device=self.device, dtype=torch.long)
                noise = torch.randn_like(z)
                z_noisy = self.scheduler.add_noise(z, noise, t)

                noise_pred = self.unet(z_noisy, t)
                loss = self.loss_fn(noise_pred, noise)

                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.grad_clip)
                self.optimizer.step()
                self.global_step += 1

                losses.append(loss.item())
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                # --- lightweight metrics ---
                if self.global_step % self.log_interval == 0:
                    self.log_metrics(
                        loss=loss,
                        z_noisy=z_noisy,
                        noise_pred=noise_pred,
                        noise=noise,
                        step=self.global_step,
                        autoencoder=self.autoencoder,
                        dataset_div=self.dataset_div,
                    )

                # --- heavy visual artifacts ---
                if self.global_step % self.sample_interval == 0:
                    samples = self.diffusion.sample(batch_size=self.num_samples, image=True)
                    self.create_viz_artifacts(
                        autoencoder=self.autoencoder,
                        samples=samples,
                        step=self.global_step,
                        losses=losses,
                        output_dir=self.output_dir,
                    )

                # --- validation ---
                if val_loader and self.global_step % self.eval_interval == 0:
                    val_loss = self.compute_validation_loss(val_loader)
                    print(f"[Validation @ step {self.global_step}] loss={val_loss:.4f}")

                    # Log to JSON
                    if hasattr(self, "metric_log"):
                        self.metric_log[-1]["val_loss"] = val_loss
                        metrics_path = os.path.join(self.output_dir, "metrics.json")
                        with open(metrics_path, "w", encoding="utf-8") as f:
                            json.dump(self.metric_log, f, indent=2)

                    # Best checkpoint
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        if self.ckpt_dir:
                            best_path = os.path.join(self.ckpt_dir, "best_val.pt")
                            self.save_checkpoint(best_path)
                            print(f"[Validation] New best model saved → {best_path}")

                    self.train(True)

            if self.ckpt_dir:
                self.save_checkpoint(f"{self.ckpt_dir}/epoch_{epoch+1}.pt")

    # ============================================================
    # Validation loss computation
    # ============================================================
    @torch.no_grad()
    def compute_validation_loss(self, val_loader):
        """Compute average validation loss."""
        self.unet.eval()
        total_loss = 0.0
        count = 0

        for batch in val_loader:
            layout = batch[0].to(self.device)
            z = self.autoencoder.encoder(layout)
            t = torch.randint(0, self.scheduler.num_steps, (z.size(0),), device=self.device, dtype=torch.long)
            noise = torch.randn_like(z)
            z_noisy = self.scheduler.add_noise(z, noise, t)

            noise_pred = self.unet(z_noisy, t)
            loss = self.loss_fn(noise_pred, noise)
            total_loss += loss.item() * z.size(0)
            count += z.size(0)

        self.unet.train()
        return total_loss / max(count, 1)

    # ============================================================
    # Evaluation (one batch visualization)
    # ============================================================
    @torch.no_grad()
    def evaluate(self, val_loader, step=None):
        self.train(False)
        layout = next(iter(val_loader))[0].to(self.device)
        z = self.autoencoder.encoder(layout)
        t = torch.randint(0, self.scheduler.num_steps, (z.size(0),), device=self.device, dtype=torch.long)
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

    # ============================================================
    # Metric logging + visualization
    # ============================================================
    @torch.no_grad()
    def log_metrics(self, loss, z_noisy, noise_pred, noise, step, autoencoder=None, dataset_div=None):
        """Compute and log lightweight metrics every N steps."""
        log = {}
        log["step"] = step
        log["loss"] = loss.item()

        snr = (z_noisy.var(dim=(1, 2, 3)) / (noise_pred - noise).var(dim=(1, 2, 3))).mean().item()
        log["snr"] = snr

        cos = F.cosine_similarity(noise_pred.flatten(1), noise.flatten(1)).mean().item()
        log["cosine"] = cos

        norms = [p.grad.norm()**2 for p in self.unet.parameters() if p.grad is not None]
        grad_norm = torch.sqrt(sum(norms)).item() if norms else 0.0
        log["grad_norm"] = grad_norm

        samples = self.diffusion.sample(batch_size=4, image=True)
        decoded = autoencoder.decoder(samples) if autoencoder else samples
        flat = decoded.flatten(1)
        div = torch.cdist(flat, flat).mean().item()
        log["diversity"] = div

        if dataset_div:
            log["div_ratio"] = div / dataset_div

        if not hasattr(self, "metric_log"):
            self.metric_log = []
        self.metric_log.append(log)

        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metric_log, f, indent=2)

        msg = f"[{step}] loss={log['loss']:.4f}, snr={log['snr']:.3f}, cos={log['cosine']:.3f}, div={log['diversity']:.3f}"
        print(msg)
        return log

    @torch.no_grad()
    def create_viz_artifacts(self, autoencoder, samples, step, losses, output_dir):
        """Create plots and sample grids every M steps."""
        os.makedirs(output_dir, exist_ok=True)

        if losses:
            plt.figure()
            plt.plot(losses)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"loss_curve.png"))
            plt.close()

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
            plt.savefig(os.path.join(output_dir, f"diversity_curve.png"))
            plt.close()

        decoded = autoencoder.decoder(samples)
        save_image(decoded, os.path.join(output_dir, f"samples_step_{step}.png"),
                   nrow=2, normalize=True, value_range=(0, 1))

    # ============================================================
    # Dataset diversity computation
    # ============================================================
    @torch.no_grad()
    def _compute_dataset_diversity(self, dataloader, num_batches=5):
        diversities = []
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            x = batch[0].to(self.device)
            decoded = self.autoencoder.decoder(self.autoencoder.encoder(x))
            flat = decoded.flatten(1)
            diversities.append(torch.cdist(flat, flat).mean().item())
        return sum(diversities) / len(diversities)
