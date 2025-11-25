#!/usr/bin/env python3
"""
Centralized training engine for autoencoder and diffusion models.

This module provides a unified Trainer class that extracts common training logic
from train.py and train_diffusion.py, reducing code duplication by ~70%.
"""

import torch
from pathlib import Path
from typing import Dict, Optional, Callable, Any
from tqdm import tqdm

from training.utils import (
    to_device,
    move_batch_to_device,
    create_grad_scaler,
)


class Trainer:
    """
    Centralized training engine that handles common training infrastructure:
    - AMP (Automatic Mixed Precision)
    - Gradient accumulation
    - Progress bars
    - Logging accumulation
    - Device management
    - Checkpoint saving
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
    ):
        """
        Initialize Trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            scheduler: Optional learning rate scheduler
            device: Device string ("cuda" or "cpu")
            use_amp: Whether to use Automatic Mixed Precision
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping (None = no clipping)
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = to_device(device)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Initialize gradient scaler for AMP
        self.scaler = None
        if self.use_amp:
            self.scaler = create_grad_scaler(use_amp, self.device)
    
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        epoch: int,
        step_fn: Callable[[torch.nn.Module, Dict[str, torch.Tensor], int, Callable, 'Trainer'], tuple[torch.Tensor, Dict[str, Any]]],
        collect_fn: Optional[Callable[[torch.nn.Module, Dict[str, torch.Tensor], Dict[str, Any]], None]] = None,
    ) -> tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            loss_fn: Loss function (may be used by step_fn)
            epoch: Current epoch number
            step_fn: Function that takes (model, batch, batch_idx, loss_fn, trainer) and returns (loss, logs)
                     The step_fn should compute loss but NOT call backward() or step() - Trainer handles that.
            collect_fn: Optional function to collect additional data (e.g., latents) from outputs
        
        Returns:
            (average_loss, average_logs)
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        log_dict = {}
        
        # Zero gradients at start
        self.optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = move_batch_to_device(batch, self.device)
            
            # Call step function to compute loss and logs
            # step_fn should handle model forward pass and loss computation
            # step_fn can optionally return (loss, logs, outputs) or (loss, logs)
            result = step_fn(self.model, batch, batch_idx, loss_fn, self)
            if len(result) == 3:
                loss, logs, outputs = result
                # Store outputs for collect_fn
                step_fn._last_outputs = outputs
            else:
                loss, logs = result
                outputs = getattr(step_fn, '_last_outputs', None)
            
            # Scale loss for gradient accumulation
            loss_scale = 1.0 / self.gradient_accumulation_steps
            scaled_loss = loss * loss_scale
            
            # Backward pass with AMP if enabled
            if self.use_amp:
                if self.scaler is not None:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
            else:
                scaled_loss.backward()
            
            # Collect additional data if needed (e.g., latents for statistics)
            if collect_fn is not None and outputs is not None:
                collect_fn(self.model, batch, outputs)
            
            # Step optimizer every gradient_accumulation_steps
            should_step = (batch_idx + 1) % self.gradient_accumulation_steps == 0
            
            if should_step:
                if self.use_amp and self.scaler is not None:
                    # Unscale gradients for clipping
                    if self.max_grad_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Clip gradients if needed
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update EMA if model has it
                if hasattr(self.model, 'update_ema'):
                    self.model.update_ema()
            
            # Accumulate statistics (use unscaled loss for logging)
            batch_size = self._get_batch_size(batch)
            loss_val = loss.detach().item() if isinstance(loss, torch.Tensor) else loss
            total_loss += loss_val * batch_size
            total_samples += batch_size
            
            # Update logs
            for k, v in logs.items():
                if k not in log_dict:
                    log_dict[k] = 0.0
                if isinstance(v, torch.Tensor):
                    log_dict[k] += v.detach().item() * batch_size
                else:
                    log_dict[k] += v * batch_size
            
            # Update progress bar
            pbar.set_postfix({
                "loss": loss_val,
                **{k: v / total_samples for k, v in log_dict.items()}
            })
        
        if total_samples == 0:
            raise RuntimeError("No samples processed in training epoch! Check dataloader.")
        
        avg_loss = total_loss / total_samples
        avg_logs = {k: v / total_samples for k, v in log_dict.items()}
        
        # Step scheduler once per epoch (if provided)
        if self.scheduler is not None:
            self.scheduler.step()
        
        return avg_loss, avg_logs
    
    def eval_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        step_fn: Callable[[torch.nn.Module, Dict[str, torch.Tensor], int, Callable, 'Trainer'], tuple[torch.Tensor, Dict[str, Any]]],
        collect_fn: Optional[Callable[[torch.nn.Module, Dict[str, torch.Tensor], Dict[str, Any]], None]] = None,
        limit_batches: Optional[int] = None,
    ) -> tuple[float, Dict[str, float]]:
        """
        Evaluate for one epoch.
        
        Args:
            dataloader: DataLoader for validation data
            loss_fn: Loss function (may be used by step_fn)
            step_fn: Function that takes (model, batch, batch_idx, loss_fn, trainer) and returns (loss, logs)
            collect_fn: Optional function to collect additional data (e.g., latents) from outputs
            limit_batches: Optional limit on number of batches to process
        
        Returns:
            (average_loss, average_logs)
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        log_dict = {}
        
        total_batches = len(dataloader)
        num_batches = min(limit_batches, total_batches) if limit_batches is not None else total_batches
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Eval", total=num_batches, leave=False)):
                # Break early if we've reached the limit
                if limit_batches is not None and batch_idx >= limit_batches:
                    break
                
                # Move batch to device
                batch = move_batch_to_device(batch, self.device)
                
                # Call step function to compute loss and logs
                # step_fn can optionally return (loss, logs, outputs) or (loss, logs)
                result = step_fn(self.model, batch, batch_idx, loss_fn, self)
                if len(result) == 3:
                    loss, logs, outputs = result
                    step_fn._last_outputs = outputs
                else:
                    loss, logs = result
                    outputs = getattr(step_fn, '_last_outputs', None)
                
                # Collect additional data if needed
                if collect_fn is not None and outputs is not None:
                    collect_fn(self.model, batch, outputs)
                
                # Accumulate statistics
                batch_size = self._get_batch_size(batch)
                loss_val = loss.detach().item() if isinstance(loss, torch.Tensor) else loss
                total_loss += loss_val * batch_size
                total_samples += batch_size
                
                # Update logs
                for k, v in logs.items():
                    if k not in log_dict:
                        log_dict[k] = 0.0
                    if isinstance(v, torch.Tensor):
                        log_dict[k] += v.detach().item() * batch_size
                    else:
                        log_dict[k] += v * batch_size
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_logs = {k: v / total_samples for k, v in log_dict.items()}
        
        return avg_loss, avg_logs
    
    def save_training_checkpoint(
        self,
        output_dir: Path,
        exp_name: str,
        epoch: int,
        best_val_loss: float,
        training_history: list,
        is_best: bool = False,
    ) -> None:
        """
        Save training checkpoint including model, optimizer, scheduler, and training state.
        
        Args:
            output_dir: Output directory for checkpoints
            exp_name: Experiment name
            epoch: Current epoch number
            best_val_loss: Best validation loss so far
            training_history: List of training history dictionaries
            is_best: Whether this is the best checkpoint so far
        """
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare extra state for model.save_checkpoint
        extra_state = {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "training_history": training_history,
            "optimizer_state": self.optimizer.state_dict(),
        }
        
        if self.scheduler is not None:
            extra_state["scheduler_state"] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            extra_state["scaler_state"] = self.scaler.state_dict()
        
        # Save latest checkpoint
        latest_path = checkpoint_dir / f"{exp_name}_checkpoint_latest.pt"
        self.model.save_checkpoint(latest_path, include_config=True, **extra_state)
        
        # Save best checkpoint if this is the best
        if is_best:
            best_path = checkpoint_dir / f"{exp_name}_checkpoint_best.pt"
            self.model.save_checkpoint(best_path, include_config=True, **extra_state)
    
    def load_training_checkpoint(
        self,
        checkpoint_path: Path,
        map_location: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load training checkpoint and restore optimizer, scheduler, and training state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            map_location: Device to load checkpoint on
        
        Returns:
            Dictionary containing extra state (epoch, best_val_loss, training_history, etc.)
        """
        # Load model checkpoint (model classes have load_checkpoint method)
        if hasattr(self.model, 'load_checkpoint'):
            model, extra_state = self.model.load_checkpoint(
                checkpoint_path,
                map_location=map_location,
                return_extra=True,
                config=None  # Use saved config from checkpoint
            )
            # Update our model reference
            self.model = model.to(self.device)
        else:
            # Fallback: load manually
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            self.model.load_state_dict(checkpoint["state_dict"])
            extra_state = {k: v for k, v in checkpoint.items() 
                          if k not in ["state_dict", "config"]}
        
        # Restore optimizer state
        if "optimizer_state" in extra_state:
            self.optimizer.load_state_dict(extra_state["optimizer_state"])
        
        # Restore scheduler state
        if "scheduler_state" in extra_state and self.scheduler is not None:
            self.scheduler.load_state_dict(extra_state["scheduler_state"])
        
        # Restore scaler state
        if "scaler_state" in extra_state and self.scaler is not None:
            self.scaler.load_state_dict(extra_state["scaler_state"])
        
        return extra_state
    
    def save_metrics_csv(
        self,
        metrics_path: Path,
        training_history: list,
    ) -> None:
        """
        Save training metrics to CSV file.
        
        Args:
            metrics_path: Path to CSV file
            training_history: List of dictionaries containing metrics per epoch
        """
        import pandas as pd
        from training.utils import save_metrics_csv as _save_metrics_csv
        
        _save_metrics_csv(training_history, metrics_path)
    
    def _get_batch_size(self, batch: Dict[str, torch.Tensor]) -> int:
        """Extract batch size from batch dictionary."""
        # Try common keys
        for key in ["rgb", "latent", "image", "input"]:
            if key in batch and isinstance(batch[key], torch.Tensor):
                return batch[key].shape[0]
        # Fallback: use first tensor's batch size
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return value.shape[0]
        return 1

