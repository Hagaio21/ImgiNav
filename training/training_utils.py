"""
Shared training utilities for model-specific training scripts.
"""

import os
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from contextlib import contextmanager


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_device(seed=42):
    """Setup device and set random seeds."""
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def setup_experiment_directories(config):
    """Create output and checkpoint directories from config."""
    output_dir = config.get("output_dir", "outputs")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    return output_dir, ckpt_dir


def save_experiment_config(config, output_dir):
    """Save experiment config to output directory."""
    config_path = os.path.join(output_dir, "experiment_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


@contextmanager
def create_progress_bar(dataloader, epoch, total_epochs):
    """Create progress bar for training loop."""
    pbar = tqdm(
        dataloader, 
        desc=f"Epoch {epoch}/{total_epochs}",
        leave=False,
        ncols=100
    )
    try:
        yield pbar
    finally:
        pbar.close()


class TrainingLogger:
    """Handles metrics accumulation, logging, and plotting for training."""
    
    def __init__(self, output_dir, log_interval=10, sample_interval=100, eval_interval=1):
        self.output_dir = Path(output_dir)
        self.log_interval = log_interval
        self.sample_interval = sample_interval
        self.eval_interval = eval_interval
        
        # Initialize metrics storage
        self.train_metrics = {}
        self.val_metrics = {}
        self.n_batches = 0
        self.current_epoch = 0
        
        # Initialize JSON logging
        self.metrics_log = []
        self.metrics_path = self.output_dir / "metrics.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create plots directory
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
    
    def should_log(self, step):
        """Check if we should log at this step."""
        return step % self.log_interval == 0
    
    def should_sample(self, step):
        """Check if we should generate samples at this step."""
        return step % self.sample_interval == 0
    
    def should_validate(self, epoch):
        """Check if we should validate at this epoch."""
        return epoch % self.eval_interval == 0
    
    def start_epoch(self, epoch):
        """Start a new epoch - reset metrics."""
        self.current_epoch = epoch
        self.train_metrics = {}
        self.val_metrics = {}
        self.n_batches = 0
    
    def log_batch(self, metrics, step, phase="train"):
        """Log metrics for a single batch."""
        # Accumulate metrics
        if phase == "train":
            if not self.train_metrics:
                self.train_metrics = metrics.copy()
            else:
                for key, value in metrics.items():
                    if key in self.train_metrics:
                        self.train_metrics[key] += value
                    else:
                        self.train_metrics[key] = value
            self.n_batches += 1
        else:
            if not self.val_metrics:
                self.val_metrics = metrics.copy()
            else:
                for key, value in metrics.items():
                    if key in self.val_metrics:
                        self.val_metrics[key] += value
                    else:
                        self.val_metrics[key] = value
        
        # Log to JSON file
        log_entry = {"step": step, "epoch": self.current_epoch}
        for key, value in metrics.items():
            log_entry[f"{phase}/{key}"] = value
        self.metrics_log.append(log_entry)
        from common.utils import write_json
        write_json(self.metrics_log, self.metrics_path)
    
    def get_epoch_metrics(self, phase="train"):
        """Get averaged metrics for the current epoch."""
        if phase == "train":
            if self.n_batches == 0:
                return {}
            return {key: value / self.n_batches for key, value in self.train_metrics.items()}
        else:
            if not self.val_metrics:
                return {}
            return self.val_metrics.copy()
    
    def format_metric_string(self, metrics):
        """Format metrics dictionary as string for progress bar."""
        return " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    
    def plot_metrics(self):
        """Plot all accumulated metrics."""
        import matplotlib.pyplot as plt
        
        if not self.metrics_log:
            return
        
        # Extract metrics from logged data
        train_losses = []
        val_losses = []
        epochs = []
        
        for entry in self.metrics_log:
            if "epoch" in entry:
                epochs.append(entry["epoch"])
                if "epoch/train_loss" in entry:
                    train_losses.append(entry["epoch/train_loss"])
                else:
                    train_losses.append(None)
                if "epoch/val_loss" in entry:
                    val_losses.append(entry["epoch/val_loss"])
                else:
                    val_losses.append(None)
        
        if not epochs:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot training loss
        plt.subplot(2, 2, 1)
        train_epochs = [e for e, l in zip(epochs, train_losses) if l is not None]
        train_vals = [l for l in train_losses if l is not None]
        if train_vals:
            plt.plot(train_epochs, train_vals, 'b-', label='Train Loss', linewidth=2)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Plot validation loss
        plt.subplot(2, 2, 2)
        val_epochs = [e for e, l in zip(epochs, val_losses) if l is not None]
        val_vals = [l for l in val_losses if l is not None]
        if val_vals:
            plt.plot(val_epochs, val_vals, 'r-', label='Val Loss', linewidth=2, marker='o')
        plt.title("Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Plot combined loss
        plt.subplot(2, 1, 2)
        if train_vals:
            plt.plot(train_epochs, train_vals, 'b-', label='Train Loss', linewidth=2)
        if val_vals:
            plt.plot(val_epochs, val_vals, 'r-', label='Val Loss', linewidth=2, marker='o')
        plt.title("Training Progress")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"metrics_epoch_{self.current_epoch}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def log_epoch_summary(self, epoch, train_metrics, val_metrics=None):
        """Log summary for the entire epoch."""
        print(f"Epoch {epoch} - Train: {self.format_metric_string(train_metrics)}")
        if val_metrics:
            print(f"Epoch {epoch} - Val: {self.format_metric_string(val_metrics)}")
        
        # Log to JSON file
        log_entry = {"step": epoch, "epoch": epoch}
        for key, value in train_metrics.items():
            log_entry[f"epoch/train_{key}"] = value
        if val_metrics:
            for key, value in val_metrics.items():
                log_entry[f"epoch/val_{key}"] = value
        self.metrics_log.append(log_entry)
        from common.utils import write_json
        write_json(self.metrics_log, self.metrics_path)
        
        # Generate plots
        self.plot_metrics()
    
    def get_metrics(self):
        """Get all logged metrics."""
        return self.metrics_log.copy()
    
    def clear(self):
        """Clear all metrics."""
        self.metrics_log = []
        from common.utils import write_json
        write_json(self.metrics_log, self.metrics_path)


def save_model_checkpoint(model, path, metadata=None):
    """Save model checkpoint with metadata."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {}
    }
    torch.save(checkpoint, path)