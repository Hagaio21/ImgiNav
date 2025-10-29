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
from pathlib import Path
from common.utils import safe_mkdir, set_seeds, extract_tensor_from_batch, MetricsLogger, save_checkpoint
from models.diffusion import LatentDiffusion
# *** ADD THIS: Import your new loss class ***
from models.losses.custom_loss import VGGPerceptualLoss


# Global Seaborn style
sns.set_theme(style="darkgrid", palette="tab20", context="talk", font_scale=0.9)


class DiffusionTrainer:
    def __init__(self,
                 unet,
                 autoencoder,
                 scheduler,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 optimizer=None,
                 
                 # *** MODIFIED: Replaced loss_fn with loss_cfg dict ***
                 loss_cfg: dict = None,
                 
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

        safe_mkdir(Path(self.output_dir))
        self.samples_dir = os.path.join(self.output_dir, "samples")
        safe_mkdir(Path(self.samples_dir))
        if self.ckpt_dir:
            safe_mkdir(Path(self.ckpt_dir))

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

        # *** MODIFIED: Loss function setup ***
        self.loss_cfg = loss_cfg or {"type": "mse"}
        self.loss_type = self.loss_cfg.get("type", "mse")
        self.mse_loss = torch.nn.MSELoss()
        self.vgg_loss = None

        if self.loss_type == "hybrid":
            print("[Trainer] Using Hybrid MSE + VGG Perceptual Loss")
            self.vgg_loss = VGGPerceptualLoss().to(self.device)
            self.lambda_mse = self.loss_cfg.get("lambda_mse", 1.0)
            self.lambda_vgg = self.loss_cfg.get("lambda_vgg", 0.1)
        else:
            print("[Trainer] Using Standard MSE Loss")
        
        # *** MODIFIED: Removed self.loss_fn ***
        self.optimizer = optimizer or AdamW(self.unet.parameters(), lr=1e-4)

        # checkpoint tracking
        self.best_loss = float("inf")
        self.latest_ckpt = os.path.join(self.ckpt_dir, "unet_latest.pt") if self.ckpt_dir else None
        self.best_ckpt = os.path.join(self.ckpt_dir, "unet_best.pt") if self.ckpt_dir else None
        
        # metrics logging
        self.metrics_logger = MetricsLogger(self.output_dir)

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
        return extract_tensor_from_batch(batch, device=device, key="layout")

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

                # *** NOTE: `layout` is the key variable ***
                # If use_embeddings=False, `layout` is the original (B,3,H,W) RGB image
                # If use_embeddings=True, `layout` is the pre-computed (B,8,H,W) latent
                layout = self._get_layout_from_batch(batch, self.device)

                if self.use_embeddings:  
                    z = layout # `z` is the latent
                else: 
                    z = self.autoencoder.encode_latent(layout) # `z` is the latent
                    
                pred, noise, t, z_t = self.diffusion.forward_step(z)

                # *** MODIFIED: Loss Calculation ***
                loss_mse = self.mse_loss(pred, noise)
                metrics = {"mse": loss_mse.item()}
                total_loss = loss_mse # Default to MSE loss

                # --- Calculate Hybrid Loss ---
                if self.vgg_loss is not None:
                    # We can only compute VGG loss if we have the original RGB image
                    if self.use_embeddings:
                        if not hasattr(self, '_vgg_warning_logged'):
                            # Log warning only once
                            print("\n[Warning] VGG loss skipped: 'use_embeddings' is true. Set to false to use VGG loss.")
                            self._vgg_warning_logged = True
                    else:
                        # This is the correct path:
                        # `layout` = original RGB image
                        # `z` = original latent
                        try:
                            # 1. Calculate predicted x0 (latent) from noise prediction
                            alpha_bar_t = self.scheduler.alpha_bars[t].view(-1, 1, 1, 1)
                            pred_x0_latent = (z_t - torch.sqrt(1 - alpha_bar_t) * pred) / torch.sqrt(alpha_bar_t)

                            # 2. Decode predicted latent to RGB (differentiable)
                            pred_rgb = self.autoencoder.decode_latent(pred_x0_latent)

                            # 3. Compute VGG loss against original RGB image (`layout`)
                            loss_vgg = self.vgg_loss(pred_rgb, layout)
                            
                            # 4. Combine losses
                            total_loss = self.lambda_mse * loss_mse + self.lambda_vgg * loss_vgg
                            metrics["vgg"] = loss_vgg.item()

                        except Exception as e:
                            if not hasattr(self, '_vgg_error_logged'):
                                print(f"\n[Warning] VGG loss failed, falling back to MSE. Error: {e}\n")
                                self._vgg_error_logged = True
                            total_loss = loss_mse # Fallback to just MSE
                
                # --- End of Loss Calculation ---

                total_loss.backward() # Use the combined loss
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.grad_clip)
                self.optimizer.step()

                self.global_step += 1
                losses.append(total_loss.item())
                pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

                # --- lightweight logs ---
                if self.global_step % self.log_interval == 0:
                    # Pass the new metrics dict
                    self.log_metrics(total_loss, z_t, pred, noise, step=self.global_step, custom_metrics=metrics)

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
            
            # *** MODIFIED: Use self.mse_loss for validation ***
            # Note: We don't run the expensive hybrid loss for validation.
            loss = self.mse_loss(pred, noise) 
            
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
        
        metadata = {"val_loss": val_loss, "step": self.global_step}
        save_checkpoint(self.diffusion.backbone.state_dict(), self.latest_ckpt, metadata)
        print(f"[Checkpoint] Latest model updated → {self.latest_ckpt}")

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            save_checkpoint(self.diffusion.backbone.state_dict(), self.best_ckpt, metadata)
            print(f"[Checkpoint] New best model saved → {self.best_ckpt}")

    # ---------------------------------------------------------
    # Logging and visualization
    # ---------------------------------------------------------
    @torch.no_grad()
    # *** MODIFIED: Added custom_metrics parameter ***
    def log_metrics(self, loss, z_noisy, noise_pred, noise, step, custom_metrics=None):
        log = {
            "step": step,
            "loss": loss.item(),
            "mse_raw": F.mse_loss(noise_pred, noise).item(), 
            "snr": (z_noisy.var(dim=(1,2,3)) / (noise_pred - noise).var(dim=(1,2,3))).mean().item(),
            "cosine": F.cosine_similarity(noise_pred.flatten(1), noise.flatten(1)).mean().item(),
        }
        
        # *** ADDED: Merge metrics from training loop (e.g., "mse", "vgg") ***
        if custom_metrics:
            log.update(custom_metrics)

        norms = [p.grad.norm() ** 2 for p in self.unet.parameters() if p.grad is not None]
        log["grad_norm"] = torch.sqrt(sum(norms)).item() if norms else 0.0

        samples = self.diffusion.sample(batch_size=self.num_samples, image=True)
        decoded = samples if samples.shape[1] != self.autoencoder.encoder.latent_channels else self.autoencoder.decoder(samples)
        flat = decoded.flatten(1)
        div = torch.cdist(flat, flat).mean().item()
        log["diversity"] = div
        log["div_ratio"] = div / self.dataset_div if hasattr(self, "dataset_div") else 0.0

        self.metrics_logger.log(log)

        # *** MODIFIED: Updated print log ***
        log_str = f"[{step}] loss={log['loss']:.4f}, mse={log.get('mse', log['mse_raw']):.4f}"
        if 'vgg' in log:
            log_str += f", vgg={log['vgg']:.4f}"
        log_str += f", snr={log['snr']:.3f}, cos={log['cosine']:.3f}, div={log['diversity']:.3f}"
        print(log_str)

    def _write_metrics_log(self):
        # Metrics are automatically saved by MetricsLogger
        pass

    @torch.no_grad()
    def create_viz_artifacts(self, samples, step, losses):
        safe_mkdir(Path(self.output_dir))
        exp = self.experiment_name

        # ---- Loss curve ----
        if losses:
            plt.figure(figsize=(7, 4))
            sns.lineplot(x=range(len(losses)), y=losses, linewidth=2, color=sns.color_palette("tab20")[0])
            plt.xlabel("Step")
            plt.ylabel("Total Loss")
            plt.title(f"{exp} — Training Loss (Total)")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "loss_curve.png"))
            plt.close()

        # ---- MSE trend ----
        metric_log = self.metrics_logger.get_metrics()
        if any("mse" in m for m in metric_log):
            steps = [x["step"] for x in metric_log if "mse" in x]
            mses = [x.get("mse") for x in metric_log if "mse" in x]
            plt.figure(figsize=(7, 4))
            sns.lineplot(x=steps, y=mses, linewidth=2, color=sns.color_palette("tab20")[2])
            plt.xlabel("Step")
            plt.ylabel("Noise MSE")
            plt.title(f"{exp} — Noise Prediction Error (MSE)")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "mse_curve.png"))
            plt.close()
        
        # ---- VGG trend ----
        if any("vgg" in m for m in metric_log):
            steps = [x["step"] for x in metric_log if "vgg" in x]
            vggs = [x.get("vgg") for x in metric_log if "vgg" in x]
            plt.figure(figsize=(7, 4))
            sns.lineplot(x=steps, y=vggs, linewidth=2, color=sns.color_palette("tab20")[3])
            plt.xlabel("Step")
            plt.ylabel("VGG Loss")
            plt.title(f"{exp} — Perceptual Loss (VGG)")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "vgg_curve.png"))
            plt.close()

        # ---- Diversity curve ----
        if metric_log:
            divs = [x["diversity"] for x in metric_log]
            steps = [x["step"] for x in metric_log]
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
            # *** MODIFIED: Handle both embeddings and RGB for diversity check ***
            if self.use_embeddings:
                # x is already a latent
                decoded = self.autoencoder.decode_latent(x)
            else:
                # x is an RGB image, must be decoded from its *own* latent
                with torch.no_grad():
                    z = self.autoencoder.encode_latent(x)
                    decoded = self.autoencoder.decode_latent(z)
            
            flat = decoded.flatten(1)
            divs.append(torch.cdist(flat, flat).mean().item())
        return sum(divs) / len(divs) if divs else 0.0