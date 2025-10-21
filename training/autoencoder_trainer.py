# training/autoencoder_trainer.py
import os
import sys
import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import yaml
from tqdm import tqdm


class AutoEncoderTrainer:
    def __init__(
        self,
        autoencoder,
        loss_fn,
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
        self.loss_fn = loss_fn
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

    def fit(self, train_loader: DataLoader, val_loader: DataLoader = None):
        self.autoencoder.train()
        self.autoencoder.to(self.device)
        step = 0

        cfg_path = os.path.join(self.output_dir, "autoencoder_config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.autoencoder.to_config(), f)
        print(f"[Config] Saved: {cfg_path}", flush=True)

        print(f"Training autoencoder for {self.epochs} epochs on {self.device}", flush=True)

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0

            with tqdm(train_loader,
                    desc=f"Epoch {epoch}/{self.epochs}",
                    unit="batch",
                    file=sys.stdout,
                    ncols=100,
                    dynamic_ncols=True,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

                for batch_idx, batch in enumerate(pbar):
                    if isinstance(batch, dict):
                        x = batch["layout"].to(self.device)
                    elif isinstance(batch, (list, tuple)):
                        x = batch[0].to(self.device)
                    else:
                        x = batch.to(self.device)

                    recon = self.autoencoder(x)
                    loss = self.loss_fn(recon, x)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    step += 1

                    if step % self.log_interval == 0:
                        pbar.write(f"[Epoch {epoch}] Step {step} | Loss: {loss.item():.6f}")
                        self.metric_log.append({
                            "epoch": epoch,
                            "step": step,
                            "train_loss": loss.item(),
                        })

                    if self.sample_interval and step % self.sample_interval == 0:
                        self._save_sample(x, recon, step)

                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"[Epoch {epoch}] Average training loss: {avg_epoch_loss:.6f}", flush=True)

            val_loss = None
            if val_loader is not None and (self.eval_interval and epoch % self.eval_interval == 0):
                val_loss = self.evaluate(val_loader, epoch)
                print(f"[Epoch {epoch}] Validation loss: {val_loss:.6f}", flush=True)

            self._save_checkpoint(epoch)
            self._update_loss_plot()
            self._save_metrics()

        print("Autoencoder training complete.", flush=True)

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, epoch: int):
        self.autoencoder.eval()
        total_loss = 0.0
        n = 0

        for batch in val_loader:
            if isinstance(batch, dict):
                x = batch["layout"].to(self.device)
            elif isinstance(batch, (list, tuple)):
                x = batch[0].to(self.device)
            else:
                x = batch.to(self.device)

            recon = self.autoencoder(x)
            loss = self.loss_fn(recon, x)
            total_loss += loss.item()
            n += 1

        avg_val = total_loss / max(n, 1)
        self.metric_log.append({"epoch": epoch, "val_loss": avg_val})
        self.autoencoder.train()
        return avg_val

    def _save_sample(self, x, recon, step):
        samples_dir = os.path.join(self.output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

        # Top: originals, bottom: reconstructions
        top = x[:4]
        bottom = recon[:4]
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
        train_steps = [m["step"] for m in self.metric_log if "train_loss" in m]
        train_losses = [m["train_loss"] for m in self.metric_log if "train_loss" in m]
        val_losses = [m["val_loss"] for m in self.metric_log if "val_loss" in m]

        if not train_losses:
            return

        plt.figure()
        plt.plot(train_steps, train_losses, label="Train Loss", color="blue")
        if val_losses:
            plt.plot(
                range(len(val_losses)),
                val_losses,
                label="Val Loss",
                color="orange",
            )

        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("AutoEncoder Training Curve")
        plt.grid(True)
        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, "train_val_curve.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"[Plot] Updated: {plot_path}")
