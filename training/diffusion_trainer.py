# diffusion_trainer.py
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import seaborn as sns
from modules.diffusion import LatentDiffusion

# Global Seaborn style
sns.set_theme(style="darkgrid", palette="tab20", context="talk", font_scale=0.9)


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
                 ckpt_dir=None,
                 output_dir=None,
                 use_embeddings=False,
                 experiment_name: str = "DiffusionExperiment",
                 logger=None):

        self.device = device
        self.logger = logger
        self.experiment_name = experiment_name
        self.epochs = epochs
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.sample_interval = sample_interval
        self.num_samples = num_samples
        self.grad_clip = grad_clip
        self.ckpt_dir = ckpt_dir
        self.output_dir = output_dir
        self.global_step = 0
        self.use_embeddings = use_embeddings

        os.makedirs(self.output_dir, exist_ok=True)
        self.samples_dir = os.path.join(self.output_dir, "samples")
        os.makedirs(self.samples_dir, exist_ok=True)
        if self.ckpt_dir:
            os.makedirs(self.ckpt_dir, exist_ok=True)

        # model setup
        self.unet = unet.to(device)
        self.autoencoder = autoencoder.eval().to(device)
        self.scheduler = scheduler.to(device)

        latent_shape = (
            self.autoencoder.encoder.latent_channels,
            self.autoencoder.encoder.latent_base,
            self.autoencoder.encoder.latent_base,
        )
        self.diffusion = LatentDiffusion(
            backbone=self.unet,
            scheduler=self.scheduler,
            autoencoder=self.autoencoder,
            latent_shape=latent_shape,
        ).to(device)

        self.loss_fn = loss_fn or torch.nn.MSELoss()
        self.optimizer = optimizer or AdamW(self.unet.parameters(), lr=1e-4)

        # checkpoint tracking
        self.best_loss = float("inf")
        self.latest_ckpt = os.path.join(self.ckpt_dir, "unet_latest.pt") if self.ckpt_dir else None
        self.best_ckpt = os.path.join(self.ckpt_dir, "unet_best.pt") if self.ckpt_dir else None

    # ---------------------------------------------------------
    # Training utilities
    # ---------------------------------------------------------
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

    @staticmethod
    def _get_layout_from_batch(batch, device):
        if isinstance(batch, dict):
            return batch["layout"].to(device)
        if isinstance(batch, (list, tuple)):
            return batch[0].to(device)
        if torch.is_tensor(batch):
            return batch.to(device)
        raise TypeError(f"Unexpected batch type: {type(batch)}")

    # ---------------------------------------------------------
    # Fit loop
    # ---------------------------------------------------------
    def fit(self, train_loader, val_loader=None):
        self.train(True)
        self.dataset_div = self._compute_dataset_diversity(train_loader)
        print(f"Dataset diversity baseline: {self.dataset_div:.4f}")

        with open(os.path.join(self.output_dir, "dataset_diversity.txt"), "w") as f:
            f.write(f"{self.dataset_div:.6f}\n")

        losses = []

        for epoch in range(self.epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)

            for batch in pbar:
                self.optimizer.zero_grad(set_to_none=True)

                layout = self._get_layout_from_batch(batch, self.device)

                if self.use_embeddings:  
                    z = layout
                else: 
                    z = self.autoencoder.encode_latent(layout)
                    
                pred, noise, t, z_t = self.diffusion.forward_step(z)

                loss = self.loss_fn(pred, noise)

                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.grad_clip)
                self.optimizer.step()

                self.global_step += 1
                losses.append(loss.item())
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                # --- lightweight logs ---
                if self.global_step % self.log_interval == 0:
                    self.log_metrics(loss, z_t, pred, noise, step=self.global_step)

                # --- heavy sampling ---
                if self.global_step % self.sample_interval == 0:
                    samples = self.diffusion.sample(batch_size=self.num_samples, image=True)
                    self.create_viz_artifacts(samples, self.global_step, losses)

                # --- validation ---
                if val_loader and self.global_step % self.eval_interval == 0:
                    val_loss = self.compute_validation_loss(val_loader)
                    print(f"[Validation @ step {self.global_step}] loss={val_loss:.4f}")
                    self.save_checkpoint(val_loss)
                    self._write_metrics_log()
                    self.train(True)

        # save final config
        cfg_path = os.path.join(self.ckpt_dir, "latent_diffusion.yaml")
        self.diffusion.to_config(cfg_path)
        print(f"[Config] Saved full diffusion config → {cfg_path}")

    # ---------------------------------------------------------
    # Validation and evaluation
    # ---------------------------------------------------------
    @torch.no_grad()
    def compute_validation_loss(self, val_loader):
        self.unet.eval()
        total, count = 0.0, 0
        for batch in val_loader:
            layout = self._get_layout_from_batch(batch, self.device)
            if self.use_embeddings:
                z = layout
            else:
                z = self.autoencoder.encode_latent(layout)
            pred, noise, _, _ = self.diffusion.forward_step(z)
            loss = self.loss_fn(pred, noise)
            total += loss.item() * z.size(0)
            count += z.size(0)
        self.unet.train()
        return total / max(count, 1)

    # ---------------------------------------------------------
    # Checkpoint management
    # ---------------------------------------------------------
    def save_checkpoint(self, val_loss: float):
        if not self.ckpt_dir:
            return
        state = {"state_dict": self.diffusion.backbone.state_dict()}
        torch.save(state, self.latest_ckpt)
        print(f"[Checkpoint] Latest model updated → {self.latest_ckpt}")

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(state, self.best_ckpt)
            print(f"[Checkpoint] New best model saved → {self.best_ckpt}")

    # ---------------------------------------------------------
    # Logging and visualization
    # ---------------------------------------------------------
    @torch.no_grad()
    def log_metrics(self, loss, z_noisy, noise_pred, noise, step):
        log = {
            "step": step,
            "loss": loss.item(),
            "mse": F.mse_loss(noise_pred, noise).item(),
            "snr": (z_noisy.var(dim=(1,2,3)) / (noise_pred - noise).var(dim=(1,2,3))).mean().item(),
            "cosine": F.cosine_similarity(noise_pred.flatten(1), noise.flatten(1)).mean().item(),
        }

        norms = [p.grad.norm() ** 2 for p in self.unet.parameters() if p.grad is not None]
        log["grad_norm"] = torch.sqrt(sum(norms)).item() if norms else 0.0

        samples = self.diffusion.sample(batch_size=self.num_samples, image=True)
        decoded = samples if samples.shape[1] != self.autoencoder.encoder.latent_channels else self.autoencoder.decoder(samples)
        flat = decoded.flatten(1)
        div = torch.cdist(flat, flat).mean().item()
        log["diversity"] = div
        log["div_ratio"] = div / self.dataset_div if hasattr(self, "dataset_div") else 0.0

        if not hasattr(self, "metric_log"):
            self.metric_log = []
        self.metric_log.append(log)
        self._write_metrics_log()

        print(f"[{step}] loss={log['loss']:.4f}, mse={log['mse']:.4f}, snr={log['snr']:.3f}, cos={log['cosine']:.3f}, div={log['diversity']:.3f}")

    def _write_metrics_log(self):
        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metric_log, f, indent=2)

    @torch.no_grad()
    def create_viz_artifacts(self, samples, step, losses):
        os.makedirs(self.output_dir, exist_ok=True)
        exp = self.experiment_name

        # ---- Loss curve ----
        if losses:
            plt.figure(figsize=(7, 4))
            sns.lineplot(x=range(len(losses)), y=losses, linewidth=2, color=sns.color_palette("tab20")[0])
            plt.xlabel("Step")
            plt.ylabel("MSE Loss")
            plt.title(f"{exp} — Training Loss (Noise Prediction)")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "loss_curve.png"))
            plt.close()

        # ---- MSE trend ----
        if hasattr(self, "metric_log") and any("mse" in m for m in self.metric_log):
            steps = [x["step"] for x in self.metric_log]
            mses = [x.get("mse") for x in self.metric_log if "mse" in x]
            plt.figure(figsize=(7, 4))
            sns.lineplot(x=steps, y=mses, linewidth=2, color=sns.color_palette("tab20")[2])
            plt.xlabel("Step")
            plt.ylabel("Noise MSE")
            plt.title(f"{exp} — Noise Prediction Error")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "mse_curve.png"))
            plt.close()

        # ---- Diversity curve ----
        if hasattr(self, "metric_log"):
            divs = [x["diversity"] for x in self.metric_log]
            steps = [x["step"] for x in self.metric_log]
            plt.figure(figsize=(7, 4))
            sns.lineplot(x=steps, y=divs, linewidth=2, color=sns.color_palette("tab20")[4], label="Model Diversity")
            if hasattr(self, "dataset_div"):
                plt.axhline(self.dataset_div, color=sns.color_palette("tab20")[8], linestyle="--", label="Dataset Diversity")
            plt.xlabel("Step")
            plt.ylabel("Diversity")
            plt.title(f"{exp} — Latent Diversity Over Training")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "diversity_curve.png"))
            plt.close()

        # ---- Save decoded samples ----
        decoded = samples if samples.shape[1] != self.autoencoder.encoder.latent_channels else self.autoencoder.decoder(samples)
        save_image(decoded, os.path.join(self.samples_dir, f"samples_step_{step}.png"),
                   nrow=2, normalize=True, value_range=(0, 1))

    # ---------------------------------------------------------
    # Dataset baseline
    # ---------------------------------------------------------
    @torch.no_grad()
    def _compute_dataset_diversity(self, dataloader, num_batches=5):
        divs = []
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            x = self._get_layout_from_batch(batch, self.device)
            decoded = self.autoencoder.decode_latent(x)
            flat = decoded.flatten(1)
            divs.append(torch.cdist(flat, flat).mean().item())
        return sum(divs) / len(divs) if divs else 0.0
