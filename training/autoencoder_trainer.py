# training/autoencoder_trainer.py
import os
import sys
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import yaml
from tqdm import tqdm


class AutoEncoderTrainer:
    def __init__(
        self,
        autoencoder,
        recon_loss_fn,
        kl_weight=1e-6, 
        epochs=10,
        log_interval=10,
        sample_interval=100,
        eval_interval=1,
        lr=1e-4,
        output_dir="ae_outputs",
        ckpt_dir="ae_outputs/checkpoints",
        device=None,
    ):
        self.autoencoder = autoencoder
        self.recon_loss_fn = recon_loss_fn
        self.kl_weight = kl_weight
        self.epochs = epochs
        self.log_interval = log_interval
        self.sample_interval = sample_interval
        self.eval_interval = eval_interval
        self.output_dir = output_dir
        self.ckpt_dir = ckpt_dir
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "samples"), exist_ok=True)

        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.lr)
        self.metric_log = []

    def _get_batch(self, batch):
        """Helper to extract tensor from various batch types."""
        if isinstance(batch, dict):
            return batch["layout"].to(self.device)
        if isinstance(batch, (list, tuple)):
            return batch[0].to(self.device)
        if torch.is_tensor(batch):
            return batch.to(self.device)
        raise TypeError(f"Unexpected batch type: {type(batch)}")

    def _compute_loss(self, x, recon, mu, logvar):
        """Computes VAE loss (Reconstruction + KL Divergence)."""
        recon_loss = self.recon_loss_fn(recon, x)
        
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])
        kl_loss = torch.mean(kl_div) # Mean over batch
        
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss

    def fit(self, train_loader: DataLoader, val_loader: DataLoader = None):
        self.autoencoder.train()
        self.autoencoder.to(self.device)
        step = 0

        cfg_path = os.path.join(self.output_dir, "autoencoder_config.yaml")
        try:
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.autoencoder.to_config(), f)
            print(f"[Config] Saved: {cfg_path}", flush=True)
        except Exception as e:
            print(f"[Config] ERROR: Could not save config: {e}", flush=True)


        print(f"Training VAE for {self.epochs} epochs on {self.device} (KL weight: {self.kl_weight})", flush=True)

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0

            with tqdm(train_loader,
                    desc=f"Epoch {epoch}/{self.epochs}",
                    unit="batch",
                    file=sys.stdout,
                    ncols=100,
                    dynamic_ncols=True,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

                for batch_idx, batch in enumerate(pbar):
                    x = self._get_batch(batch)

                    recon, mu, logvar = self.autoencoder(x)
                    
                    loss, recon_loss, kl_loss = self._compute_loss(x, recon, mu, logvar)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    epoch_recon_loss += recon_loss.item()
                    epoch_kl_loss += kl_loss.item()
                    step += 1

                    if step % self.log_interval == 0:
                        log_entry = {
                            "epoch": epoch,
                            "step": step,
                            "train_loss": loss.item(),
                            "train_recon_loss": recon_loss.item(),
                            "train_kl_loss": kl_loss.item(),
                        }
                        self.metric_log.append(log_entry)
                        pbar.write(f"[Epoch {epoch}] Step {step} | Loss: {loss.item():.6f} (Recon: {recon_loss.item():.6f}, KL: {kl_loss.item():.6f})")


                    if self.sample_interval and step % self.sample_interval == 0:
                        self._save_sample(x, recon, step)

                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "kl": f"{kl_loss.item():.4f}"})

            avg_epoch_loss = epoch_loss / len(train_loader)
            avg_recon_loss = epoch_recon_loss / len(train_loader)
            avg_kl_loss = epoch_kl_loss / len(train_loader)
            
            print(f"[Epoch {epoch}] Avg Train Loss: {avg_epoch_loss:.6f} (Recon: {avg_recon_loss:.6f}, KL: {avg_kl_loss:.6f})", flush=True)

            val_loss = None
            if val_loader is not None and (self.eval_interval and epoch % self.eval_interval == 0):
                val_loss, val_recon, val_kl = self.evaluate(val_loader)
                print(f"[Epoch {epoch}] Validation Loss: {val_loss:.6f} (Recon: {val_recon:.6f}, KL: {val_kl:.6f})", flush=True)
                
                for entry in reversed(self.metric_log):
                    if entry["epoch"] == epoch:
                        entry["val_loss"] = val_loss
                        entry["val_recon_loss"] = val_recon
                        entry["val_kl_loss"] = val_kl
                        break
                else:
                    self.metric_log.append({
                        "epoch": epoch, "step": step,
                        "val_loss": val_loss, "val_recon_loss": val_recon, "val_kl_loss": val_kl
                    })


            self._save_checkpoint(epoch)
            self._update_loss_plot() 
            self._update_loss_components_plot() 
            self._save_metrics()

        print("Autoencoder training complete.", flush=True)
        
        final_path = os.path.join(self.ckpt_dir, "ae_latest.pt")
        torch.save(self.autoencoder.state_dict(), final_path)
        print(f"[Checkpoint] Saved final model: {final_path}")


    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader):
        self.autoencoder.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        n = 0

        for batch in val_loader:
            x = self._get_batch(batch)

            recon, mu, logvar = self.autoencoder(x)
            loss, recon_loss, kl_loss = self._compute_loss(x, recon, mu, logvar)
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            n += 1

        avg_val = total_loss / max(n, 1)
        avg_recon = total_recon_loss / max(n, 1)
        avg_kl = total_kl_loss / max(n, 1)
        
        self.autoencoder.train()
        return avg_val, avg_recon, avg_kl

    def _save_sample(self, x, recon, step):
        samples_dir = os.path.join(self.output_dir, "samples")
        
        top = x[:4].detach().cpu()
        bottom = recon[:4].detach().cpu()
        comparison = torch.cat([top, bottom], dim=0)

        path = os.path.join(samples_dir, f"sample_step_{step}.png")
        save_image(comparison, path, nrow=4)
        print(f"[Sample] Saved: {path}")

    def _save_checkpoint(self, epoch):
        path = os.path.join(self.ckpt_dir, f"ae_epoch_{epoch}.pt")
        torch.save(self.autoencoder.state_dict(), path)
        print(f"[Checkpoint] Saved: {path}")

    def _save_metrics(self):
        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metric_log, f, indent=2)
        print(f"[Metrics] Updated: {metrics_path}")

    def _update_loss_plot(self):
        """Plots the total combined (Train vs Val) loss."""
        train_steps = [m["step"] for m in self.metric_log if "train_loss" in m]
        train_losses = [m["train_loss"] for m in self.metric_log if "train_loss" in m]
        
        val_steps = [m["step"] for m in self.metric_log if "val_loss" in m]
        val_losses = [m["val_loss"] for m in self.metric_log if "val_loss" in m]

        if not train_losses:
            return

        plt.figure(figsize=(10, 5))
        plt.plot(train_steps, train_losses, label="Train Loss (Total)", color="blue", alpha=0.7)
        if val_losses:
            plt.plot(val_steps, val_losses, label="Val Loss (Total)", color="orange", marker='o', linestyle='--')

        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Total VAE Training Curve")
        plt.grid(True)
        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, "train_val_curve.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"[Plot] Updated: {plot_path}")

    def _update_loss_components_plot(self):
        """Plots the separate (Recon vs KL) loss components."""
        train_steps = [m["step"] for m in self.metric_log if "train_recon_loss" in m]
        train_recon = [m["train_recon_loss"] for m in self.metric_log if "train_recon_loss" in m]
        train_kl = [m["train_kl_loss"] for m in self.metric_log if "train_kl_loss" in m]

        val_steps = [m["step"] for m in self.metric_log if "val_recon_loss" in m]
        val_recon = [m["val_recon_loss"] for m in self.metric_log if "val_recon_loss" in m]
        val_kl = [m["val_kl_loss"] for m in self.metric_log if "val_kl_loss" in m]

        if not train_steps:
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax1.plot(train_steps, train_recon, label="Train Recon Loss", color="green", alpha=0.7)
        if val_steps:
            ax1.plot(val_steps, val_recon, label="Val Recon Loss", color="red", marker='o', linestyle='--')
        ax1.set_ylabel("Recon Loss")
        ax1.legend()
        ax1.set_title("Loss Components")
        ax1.grid(True)
        
        ax2.plot(train_steps, train_kl, label="Train KL Loss", color="purple", alpha=0.7)
        if val_steps:
            ax2.plot(val_steps, val_kl, label="Val KL Loss", color="magenta", marker='o', linestyle='--')
        ax2.set_ylabel("KL Loss")
        ax2.set_xlabel("Step")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "loss_components.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"[Plot] Updated: {plot_path}")