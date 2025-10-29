import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
import inspect
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pathlib import Path
from training.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Generic trainer that works with any model that implements:
    - forward(batch) -> outputs_dict
    - training_sample(batch_size, **kwargs) -> samples
    
    And loss functions that take outputs_dict and return (loss, metrics_dict).
    """
    
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        epochs=10,
        log_interval=10,
        sample_interval=100,
        eval_interval=1,
        grad_clip=None,
        cfg_dropout_prob=0.0,
        num_training_samples=4,
        output_dir="outputs",
        ckpt_dir=None,
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
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.cfg_dropout_prob = cfg_dropout_prob
        self.num_training_samples = num_training_samples
        self.global_step = 0
        self.samples_dir = os.path.join(self.output_dir, "samples")
        
        # Move model to device
        if hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)
        else:
            for param in self.model.parameters():
                if param.device != self.device:
                    param.data = param.data.to(self.device)
    
    def fit(self, train_loader, val_loader=None):
        """Generic training loop."""
        self.model.train()
        print(f"Training for {self.epochs} epochs on {self.device}", flush=True)
        
        for epoch in range(1, self.epochs + 1):
            epoch_metrics = {}
            
            with self._create_progress_bar(train_loader, epoch, self.epochs) as pbar:
                for batch in pbar:
                    # Move batch to device
                    batch = self._move_batch_to_device(batch)
                    
                    # Forward pass (handle models with/without cfg_dropout_prob)
                    sig = inspect.signature(self.model.forward)
                    if 'cfg_dropout_prob' in sig.parameters:
                        outputs = self.model(batch, cfg_dropout_prob=self.cfg_dropout_prob)
                    else:
                        outputs = self.model(batch)
                    
                    # Compute loss
                    # Handle both legacy VAE losses (x, outputs) and new losses (outputs)
                    try:
                        # Try new-style: loss_fn(outputs)
                        loss_result = self.loss_fn(outputs)
                    except TypeError:
                        # Fall back to legacy: loss_fn(x, outputs)
                        x = outputs.get("input", outputs.get("original_latent"))
                        if x is None:
                            # Try to extract from batch if not in outputs
                            if isinstance(batch, dict):
                                x = batch.get("layout", batch.get("image", batch.get("x")))
                        if x is None:
                            raise ValueError("Could not extract input 'x' for legacy loss function")
                        loss_result = self.loss_fn(x, outputs)
                    
                    # Parse loss result
                    if isinstance(loss_result, tuple):
                        if len(loss_result) == 4:
                            total_loss, _, _, metrics = loss_result
                        elif len(loss_result) == 5:
                            total_loss, _, _, _, metrics = loss_result
                        elif len(loss_result) == 2:
                            total_loss, metrics = loss_result
                        else:
                            total_loss = loss_result[0]
                            metrics = loss_result[-1] if len(loss_result) > 1 else {}
                    else:
                        total_loss = loss_result
                        metrics = {}
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    
                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    
                    self.optimizer.step()
                    
                    # Accumulate metrics
                    epoch_metrics = self._accumulate_metrics(metrics, epoch_metrics)
                    self.global_step += 1
                    
                    # Logging
                    if self._should_log(self.global_step):
                        log_entry = self._log_metrics(metrics, self.global_step, epoch, prefix="train")
                        metric_str = self._format_metric_string(metrics)
                        pbar.write(f"[Epoch {epoch}] Step {self.global_step} | {metric_str}")
                    
                    # Sampling
                    if self._should_sample(self.global_step):
                        if hasattr(self.model, 'training_sample'):
                            samples = self.model.training_sample(
                                batch_size=self.num_training_samples,
                                device=self.device,
                            )
                            self._save_samples(samples, self.global_step)
                    
                    pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})
            
            # Epoch summary
            n_batches = len(train_loader)
            avg_metrics = self._average_metrics(epoch_metrics, n_batches)
            metric_str = self._format_metric_string(avg_metrics)
            print(f"[Epoch {epoch}] Train Avg | {metric_str}", flush=True)
            
            # Validation
            if val_loader is not None and self._should_validate(epoch):
                val_metrics = self.evaluate(val_loader)
                metric_str = self._format_metric_string(val_metrics)
                print(f"[Epoch {epoch}] Val Avg | {metric_str}", flush=True)
                self._log_metrics(val_metrics, self.global_step, epoch, prefix="val")
            
            # Checkpointing
            self._save_checkpoint(epoch)
            self.metrics_logger.create_all_plots()
        
        # Final checkpoint
        self._save_final_checkpoint()
        print("Training complete.", flush=True)
    
    @torch.no_grad()
    def evaluate(self, val_loader):
        """Generic evaluation loop."""
        self.model.eval()
        total_metrics = {}
        n_batches = 0
        
        for batch in val_loader:
            batch = self._move_batch_to_device(batch)
            # Forward pass (handle models with/without cfg_dropout_prob)
            sig = inspect.signature(self.model.forward)
            if 'cfg_dropout_prob' in sig.parameters:
                outputs = self.model(batch, cfg_dropout_prob=0.0)  # No dropout during eval
            else:
                outputs = self.model(batch)
            
            # Compute loss/metrics (same logic as training)
            try:
                loss_result = self.loss_fn(outputs)
            except TypeError:
                x = outputs.get("input", outputs.get("original_latent"))
                if x is None:
                    if isinstance(batch, dict):
                        x = batch.get("layout", batch.get("image", batch.get("x")))
                if x is None:
                    raise ValueError("Could not extract input 'x' for legacy loss function")
                loss_result = self.loss_fn(x, outputs)
            
            if isinstance(loss_result, tuple):
                if len(loss_result) >= 2:
                    metrics = loss_result[-1] if isinstance(loss_result[-1], dict) else {}
                else:
                    metrics = {}
            else:
                metrics = {}
            
            total_metrics = self._accumulate_metrics(metrics, total_metrics)
            n_batches += 1
        
        self.model.train()
        return self._average_metrics(total_metrics, n_batches)
    
    def _move_batch_to_device(self, batch):
        """Move batch dict to device."""
        if isinstance(batch, dict):
            device_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    device_batch[key] = value.to(self.device)
                elif isinstance(value, list):
                    # Handle lists of tensors
                    device_batch[key] = [v.to(self.device) if isinstance(v, torch.Tensor) else v for v in value]
                else:
                    device_batch[key] = value
            return device_batch
        else:
            # Assume it's a tensor
            return batch.to(self.device)
    
    def _save_samples(self, samples, step):
        """Save sample images."""
        if samples is None:
            return
        path = os.path.join(self.samples_dir, f"sample_step_{step}.png")
        save_image(samples, path, nrow=2, normalize=True, value_range=(0, 1))
        print(f"[Sample] Saved: {path}")
    
    def _save_checkpoint(self, epoch):
        """Save checkpoint at end of epoch."""
        state_dict = self._get_model_state_dict()
        self._save_checkpoint_base(
            state_dict,
            f"checkpoint_epoch_{epoch}.pt",
            metadata={"epoch": epoch, "step": self.global_step}
        )
    
    def _save_final_checkpoint(self):
        """Save final checkpoint."""
        state_dict = self._get_model_state_dict()
        self._save_checkpoint_base(
            state_dict,
            "checkpoint_latest.pt",
            metadata={"epoch": self.epochs, "step": self.global_step, "final": True}
        )
    
    def _get_model_state_dict(self):
        """Extract state dict from model - completely generic."""
        # Try standard PyTorch model interface first
        if hasattr(self.model, 'state_dict'):
            return self.model.state_dict()
        # Fallback: try to get state dict from any nn.Module
        elif isinstance(self.model, torch.nn.Module):
            return self.model.state_dict()
        else:
            raise ValueError(f"Model {type(self.model)} does not support state_dict()")

