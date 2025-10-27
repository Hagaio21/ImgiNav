# modules/trainer.py
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
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
        loss_fn,  # Now accepts any callable loss function
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

    def _get_batch(self, batch):
        """Helper to extract tensor from various batch types."""
        if isinstance(batch, dict):
            return batch["layout"].to(self.device)
        if isinstance(batch, (list, tuple)):
            return batch[0].to(self.device)
        if torch.is_tensor(batch):
            return batch.to(self.device)
        raise TypeError(f"Unexpected batch type: {type(batch)}")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader = None):
        self.autoencoder.train()
        self.autoencoder.to(self.device)
        step = 0

        # Save config
        cfg_path = os.path.join(self.output_dir, "autoencoder_config.yaml")
        try:
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.autoencoder.to_config(), f)
            print(f"[Config] Saved: {cfg_path}", flush=True)
        except Exception as e:
            print(f"[Config] ERROR: Could not save config: {e}", flush=True)

        print(f"Training VAE for {self.epochs} epochs on {self.device}", flush=True)

        for epoch in range(1, self.epochs + 1):
            epoch_metrics = {}

            with tqdm(train_loader,
                      desc=f"Epoch {epoch}/{self.epochs}",
                      unit="batch",
                      file=sys.stdout,
                      ncols=100,
                      dynamic_ncols=True,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

                for batch_idx, batch in enumerate(pbar):
                    x = self._get_batch(batch)
                    outputs = self.autoencoder(x, deterministic=getattr(self.autoencoder, "deterministic", True))
                    loss_output = self.loss_fn(x, outputs)

                    # Call loss function - it returns (total_loss, *components, metrics_dict)
                    total_loss = loss_output[0]
                    metrics = loss_output[-1]  # Last element is always metrics dict

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

                    # Accumulate epoch metrics
                    for key, val in metrics.items():
                        epoch_metrics[key] = epoch_metrics.get(key, 0) + val

                    step += 1

                    # Logging
                    if step % self.log_interval == 0:
                        log_entry = {"epoch": epoch, "step": step}
                        for key, val in metrics.items():
                            log_entry[f"train_{key}"] = val
                        self.metric_log.append(log_entry)
                        
                        # Format log message
                        metric_str = ", ".join([f"{k}={v:.5f}" for k, v in metrics.items()])
                        pbar.write(f"[Epoch {epoch}] Step {step} | {metric_str}")

                    # Save samples
                    if self.sample_interval and step % self.sample_interval == 0:
                        self._save_sample(x, outputs["recon"], step)

                    # Update progress bar
                    pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

            # Epoch summary
            n_batches = len(train_loader)
            avg_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}
            metric_str = ", ".join([f"{k}={v:.5f}" for k, v in avg_metrics.items()])
            print(f"[Epoch {epoch}] Train Avg | {metric_str}", flush=True)

            # Validation
            if val_loader is not None and (self.eval_interval and epoch % self.eval_interval == 0):
                val_metrics = self.evaluate(val_loader)
                metric_str = ", ".join([f"{k}={v:.5f}" for k, v in val_metrics.items()])
                print(f"[Epoch {epoch}] Val Avg | {metric_str}", flush=True)

                # Add to log
                log_entry = {"epoch": epoch, "step": step}
                for key, val in val_metrics.items():
                    log_entry[f"val_{key}"] = val
                self.metric_log.append(log_entry)

            self._save_checkpoint(epoch)
            self._save_metrics()
            self._plot_losses()

        print("Autoencoder training complete.", flush=True)

        # Save final model
        final_path = os.path.join(self.ckpt_dir, "ae_latest.pt")
        torch.save(self.autoencoder.state_dict(), final_path)
        print(f"[Checkpoint] Saved final model: {final_path}")

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader):
        self.autoencoder.eval()
        total_metrics = {}
        n = 0

        for batch in val_loader:
            x = self._get_batch(batch)
            outputs = self.autoencoder(x, deterministic=getattr(self.autoencoder, "deterministic", True))
            
            # Call loss function
            loss_output = self.loss_fn(x, outputs) # <-- CHANGE
            metrics = loss_output[-1]
            
            # Accumulate
            for key, val in metrics.items():
                total_metrics[key] = total_metrics.get(key, 0) + val
            n += 1

        # Average
        avg_metrics = {k: v / max(n, 1) for k, v in total_metrics.items()}
        
        self.autoencoder.train()
        return avg_metrics

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

    def _plot_losses(self):
        """Dynamic plotting based on available metrics."""
        if not self.metric_log:
            return

        # Extract all unique metric keys
        train_keys = set()
        val_keys = set()
        for entry in self.metric_log:
            for key in entry.keys():
                if key.startswith("train_"):
                    train_keys.add(key)
                elif key.startswith("val_"):
                    val_keys.add(key)

        if not train_keys:
            return

        # Group metrics by type (excluding step/epoch)
        metric_names = sorted(set(k.replace("train_", "").replace("val_", "") 
                                 for k in train_keys | val_keys))
        
        # Create subplots
        n_plots = len(metric_names)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), 
                                 sharex=True, squeeze=False)
        axes = axes.flatten()

        for idx, metric in enumerate(metric_names):
            ax = axes[idx]
            
            # Train data
            train_key = f"train_{metric}"
            steps = [m["step"] for m in self.metric_log if train_key in m]
            values = [m[train_key] for m in self.metric_log if train_key in m]
            if steps:
                ax.plot(steps, values, label=f"Train {metric}", alpha=0.7)
            
            # Val data
            val_key = f"val_{metric}"
            val_steps = [m["step"] for m in self.metric_log if val_key in m]
            val_values = [m[val_key] for m in self.metric_log if val_key in m]
            if val_steps:
                ax.plot(val_steps, val_values, label=f"Val {metric}", 
                       marker='o', linestyle='--')
            
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True)
            ax.set_title(f"{metric.upper()}")

        axes[-1].set_xlabel("Step")
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, "loss_curves.png")
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"[Plot] Updated: {plot_path}")