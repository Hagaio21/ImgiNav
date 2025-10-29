import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from training.base_trainer import BaseTrainer
from training.utils import save_model_config


class AutoEncoderTrainer(BaseTrainer):
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
        super().__init__(
            epochs=epochs,
            log_interval=log_interval,
            sample_interval=sample_interval,
            eval_interval=eval_interval,
            output_dir=output_dir,
            ckpt_dir=ckpt_dir,
            device=device,
        )
        self.autoencoder = autoencoder
        self.loss_fn = loss_fn
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.lr)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader = None):
        self.autoencoder.train()
        self.autoencoder.to(self.device)
        step = 0

        save_model_config(self.autoencoder, self.output_dir)
        print(f"Training VAE for {self.epochs} epochs on {self.device}", flush=True)

        for epoch in range(1, self.epochs + 1):
            epoch_metrics = {}

            with self._create_progress_bar(train_loader, epoch, self.epochs) as pbar:
                for batch_idx, batch in enumerate(pbar):
                    x = self._get_batch(batch)
                    outputs = self.autoencoder(x, deterministic=getattr(self.autoencoder, "deterministic", True))
                    loss_output = self.loss_fn(x, outputs)

                    total_loss = loss_output[0]
                    metrics = loss_output[-1]

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

                    epoch_metrics = self._accumulate_metrics(metrics, epoch_metrics)
                    step += 1

                    if self._should_log(step):
                        log_entry = self._log_metrics(metrics, step, epoch, prefix="train")
                        metric_str = self._format_metric_string(metrics)
                        pbar.write(f"[Epoch {epoch}] Step {step} | {metric_str}")

                    if self._should_sample(step):
                        self._save_sample(x, outputs["recon"], step)

                    pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

            n_batches = len(train_loader)
            avg_metrics = self._average_metrics(epoch_metrics, n_batches)
            metric_str = self._format_metric_string(avg_metrics)
            print(f"[Epoch {epoch}] Train Avg | {metric_str}", flush=True)

            if val_loader is not None and self._should_validate(epoch):
                val_metrics = self.evaluate(val_loader)
                metric_str = self._format_metric_string(val_metrics)
                print(f"[Epoch {epoch}] Val Avg | {metric_str}", flush=True)
                self._log_metrics(val_metrics, step, epoch, prefix="val")

            self._save_checkpoint(epoch)
            self._save_metrics()
            self.metrics_logger.create_all_plots()

        print("Autoencoder training complete.", flush=True)
        self._save_checkpoint_base(
            self.autoencoder.state_dict(),
            "ae_latest.pt",
            metadata={"epoch": self.epochs, "final": True}
        )

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader):
        self.autoencoder.eval()
        total_metrics = {}
        n = 0

        for batch in val_loader:
            x = self._get_batch(batch)
            outputs = self.autoencoder(x, deterministic=getattr(self.autoencoder, "deterministic", True))
            loss_output = self.loss_fn(x, outputs)
            metrics = loss_output[-1]
            
            total_metrics = self._accumulate_metrics(metrics, total_metrics)
            n += 1

        avg_metrics = self._average_metrics(total_metrics, n)
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
        self._save_checkpoint_base(
            self.autoencoder.state_dict(),
            f"ae_epoch_{epoch}.pt",
            metadata={"epoch": epoch}
        )

    def _save_metrics(self):
        pass