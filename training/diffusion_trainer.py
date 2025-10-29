import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from common.utils import safe_mkdir
from models.diffusion import LatentDiffusion
from models.losses.custom_loss import VGGPerceptualLoss
from training.base_trainer import BaseTrainer

sns.set_theme(style="darkgrid", palette="tab20", context="talk", font_scale=0.9)


class DiffusionTrainer(BaseTrainer):
    def __init__(self,
                 unet,
                 autoencoder,
                 scheduler,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 optimizer=None,
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

        super().__init__(
            epochs=epochs,
            log_interval=log_interval,
            sample_interval=sample_interval,
            eval_interval=eval_interval,
            output_dir=output_dir,
            ckpt_dir=ckpt_dir,
            device=device,
        )
        
        self.logger = logger
        self.experiment_name = experiment_name
        self.num_samples = num_samples
        self.grad_clip = grad_clip
        self.global_step = 0
        self.use_embeddings = use_embeddings
        self.samples_dir = os.path.join(self.output_dir, "samples")

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

        self.best_loss = float("inf")
        self.latest_ckpt = os.path.join(self.ckpt_dir, "unet_latest.pt") if self.ckpt_dir else None
        self.best_ckpt = os.path.join(self.ckpt_dir, "unet_best.pt") if self.ckpt_dir else None

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

    def _get_layout_from_batch(self, batch):
        return self._get_batch(batch, key="layout")

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
            with self._create_progress_bar(train_loader, epoch + 1, self.epochs) as pbar:
                for batch in pbar:
                    self.optimizer.zero_grad(set_to_none=True)

                    layout = self._get_layout_from_batch(batch)

                    if self.use_embeddings:  
                        z = layout
                    else: 
                        z = self.autoencoder.encode_latent(layout)
                        
                    pred, noise, t, z_t = self.diffusion.forward_step(z)

                    loss_mse = self.mse_loss(pred, noise)
                    metrics = {"mse": loss_mse.item()}
                    total_loss = loss_mse

                    if self.vgg_loss is not None:
                        if self.use_embeddings:
                            if not hasattr(self, '_vgg_warning_logged'):
                                print("\n[Warning] VGG loss skipped: 'use_embeddings' is true. Set to false to use VGG loss.")
                                self._vgg_warning_logged = True
                        else:
                            try:
                                alpha_bar_t = self.scheduler.alpha_bars[t].view(-1, 1, 1, 1)
                                pred_x0_latent = (z_t - torch.sqrt(1 - alpha_bar_t) * pred) / torch.sqrt(alpha_bar_t)
                                pred_rgb = self.autoencoder.decode_latent(pred_x0_latent)
                                loss_vgg = self.vgg_loss(pred_rgb, layout)
                                
                                total_loss = self.lambda_mse * loss_mse + self.lambda_vgg * loss_vgg
                                metrics["vgg"] = loss_vgg.item()

                            except Exception as e:
                                if not hasattr(self, '_vgg_error_logged'):
                                    print(f"\n[Warning] VGG loss failed, falling back to MSE. Error: {e}\n")
                                    self._vgg_error_logged = True
                                total_loss = loss_mse

                    total_loss.backward()
                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.grad_clip)
                    self.optimizer.step()

                    self.global_step += 1
                    losses.append(total_loss.item())
                    pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

                    if self._should_log(self.global_step):
                        self.log_metrics(total_loss, z_t, pred, noise, step=self.global_step, custom_metrics=metrics)

                    if self._should_sample(self.global_step):
                        samples = self.diffusion.sample(batch_size=self.num_samples, image=True)
                        self._save_samples(samples, self.global_step)
                        self.create_viz_artifacts(samples, self.global_step, losses)

                    if val_loader and self.global_step % self.eval_interval == 0:
                        val_loss = self.compute_validation_loss(val_loader)
                        print(f"[Validation @ step {self.global_step}] loss={val_loss:.4f}")
                        self.save_checkpoint(val_loss)
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
            layout = self._get_layout_from_batch(batch)
            if self.use_embeddings:
                z = layout
            else:
                z = self.autoencoder.encode_latent(layout)
            pred, noise, _, _ = self.diffusion.forward_step(z)
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
        self._save_checkpoint_base(self.diffusion.backbone.state_dict(), "unet_latest.pt", metadata)

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self._save_checkpoint_base(self.diffusion.backbone.state_dict(), "unet_best.pt", metadata)
            print(f"[Checkpoint] New best model saved → {os.path.join(self.ckpt_dir, 'unet_best.pt')}")

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

        log_str = f"[{step}] loss={log['loss']:.4f}, mse={log.get('mse', log['mse_raw']):.4f}"
        if 'vgg' in log:
            log_str += f", vgg={log['vgg']:.4f}"
        log_str += f", snr={log['snr']:.3f}, cos={log['cosine']:.3f}, div={log['diversity']:.3f}"
        print(log_str)

    def _save_samples(self, samples, step):
        decoded = samples if samples.shape[1] != self.autoencoder.encoder.latent_channels else self.autoencoder.decoder(samples)
        save_image(decoded, os.path.join(self.samples_dir, f"samples_step_{step}.png"),
                   nrow=2, normalize=True, value_range=(0, 1))

    @torch.no_grad()
    def create_viz_artifacts(self, samples, step, losses):
        exp = self.experiment_name

        if losses:
            import seaborn as sns
            plt.figure(figsize=(7, 4))
            sns.lineplot(x=range(len(losses)), y=losses, linewidth=2, color=sns.color_palette("tab20")[0])
            plt.xlabel("Step")
            plt.ylabel("Total Loss")
            plt.title(f"{exp} — Training Loss (Total)")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "loss_curve.png"))
            plt.close()

        self.metrics_logger.create_all_plots()
        
        if hasattr(self, "dataset_div"):
            self.metrics_logger.create_plot_with_baseline(
                "diversity",
                baseline_value=self.dataset_div,
                baseline_label="Dataset Diversity"
            )

    # ---------------------------------------------------------
    # Dataset baseline
    # ---------------------------------------------------------
    @torch.no_grad()
    def _compute_dataset_diversity(self, dataloader, num_batches=5):
        divs = []
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            x = self._get_layout_from_batch(batch)
            if self.use_embeddings:
                decoded = self.autoencoder.decode_latent(x)
            else:
                with torch.no_grad():
                    z = self.autoencoder.encode_latent(x)
                    decoded = self.autoencoder.decode_latent(z)
            
            flat = decoded.flatten(1)
            divs.append(torch.cdist(flat, flat).mean().item())
        return sum(divs) / len(divs) if divs else 0.0