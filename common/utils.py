#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Dict, List


def safe_mkdir(path: Path, parents: bool = True, exist_ok: bool = True):
    try:
        path.mkdir(parents=parents, exist_ok=exist_ok)
    except Exception as e:
        raise RuntimeError(f"Failed to create directory {path}: {e}")


def write_json(data: Dict, path: Path, indent: int = 2):
    try:
        safe_mkdir(path.parent)
        path.write_text(json.dumps(data, indent=indent), encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to write JSON to {path}: {e}")


def create_progress_tracker(total: int, description: str = "Processing"):
    def update_progress(current: int, item_name: str = "", success: bool = True):
        status = "✓" if success else "✗"
        percentage = (current / total) * 100 if total > 0 else 0
        print(f"[{current}/{total}] ({percentage:.1f}%) {status} {description} {item_name}", flush=True)
    return update_progress


def load_config_with_profile(config_path: str = None, profile: str = None) -> Dict:
    if not config_path:
        return {}
    
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    if path.suffix.lower() in (".yml", ".yaml"):
        try:
            import yaml
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except ImportError as e:
            raise RuntimeError("YAML config requested but 'pyyaml' is not installed") from e
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    
    if not isinstance(data, dict):
        raise ValueError("Config must be a dictionary")
    
    profile_name = profile or data.get("profile")
    if profile_name and "profiles" in data:
        if profile_name not in data["profiles"]:
            raise ValueError(f"Profile '{profile_name}' not found")
        
        base_config = {k: v for k, v in data.items() if k not in ("profiles", "profile")}
        base_config.update(data["profiles"][profile_name])
        return base_config
    
    return data


def ensure_columns_exist(df, required_columns: List[str], source: str = "dataframe"):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns {missing} in {source}")


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def extract_tensor_from_batch(batch, device=None, key="layout"):
    """
    Extract tensor from various batch types.
    
    Args:
        batch: Can be dict, list/tuple, or tensor
        device: Optional device to move tensor to
        key: Key to extract from dict (default: "layout")
    
    Returns:
        torch.Tensor: Extracted tensor
    """
    import torch
    
    if isinstance(batch, dict):
        tensor = batch[key]
    elif isinstance(batch, (list, tuple)):
        tensor = batch[0]
    elif torch.is_tensor(batch):
        tensor = batch
    else:
        raise TypeError(f"Unexpected batch type: {type(batch)}")
    
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor


class MetricsLogger:
    """Shared metrics logging utility for trainers."""
    
    def __init__(self, output_dir, filename="metrics.json", auto_plot=False, plot_interval=None, train_prefix="train_", val_prefix="val_"):
        self.output_dir = Path(output_dir)
        self.filename = filename
        self.metrics_log = []
        self.metrics_path = self.output_dir / filename
        self.auto_plot = auto_plot
        self.plot_interval = plot_interval
        self.train_prefix = train_prefix
        self.val_prefix = val_prefix
        self._log_count = 0
        
        safe_mkdir(self.output_dir)
    
    def log(self, metrics_dict):
        """Log a metrics dictionary."""
        self.metrics_log.append(metrics_dict)
        self._write_to_file()
        self._log_count += 1
        
        if self.auto_plot:
            if self.plot_interval is None or self._log_count % self.plot_interval == 0:
                self.create_all_plots()
    
    def _write_to_file(self):
        """Write metrics to JSON file."""
        try:
            write_json(self.metrics_log, self.metrics_path)
        except Exception as e:
            print(f"Warning: Failed to write metrics to {self.metrics_path}: {e}")
    
    def get_metrics(self):
        """Get all logged metrics."""
        return self.metrics_log.copy()
    
    def clear(self):
        """Clear all metrics."""
        self.metrics_log = []
        self._write_to_file()
    
    def create_all_plots(self):
        """Automatically create plots for all tracked metrics."""
        if not self.metrics_log:
            return
        
        exclude_keys = {"step", "epoch"}
        train_keys = set()
        val_keys = set()
        scalar_keys = set()
        
        for entry in self.metrics_log:
            for key in entry.keys():
                if key in exclude_keys:
                    continue
                if key.startswith(self.train_prefix):
                    train_keys.add(key)
                elif key.startswith(self.val_prefix):
                    val_keys.add(key)
                elif isinstance(entry[key], (int, float)):
                    scalar_keys.add(key)
        
        if train_keys or val_keys:
            self._create_train_val_plots()
        
        for key in scalar_keys:
            if key not in exclude_keys:
                self._create_scalar_plot(key)
    
    def _create_train_val_plots(self):
        """Create plots for train/val metrics."""
        import matplotlib.pyplot as plt
        
        train_keys = set()
        val_keys = set()
        for entry in self.metrics_log:
            for key in entry.keys():
                if key.startswith(self.train_prefix):
                    train_keys.add(key)
                elif key.startswith(self.val_prefix):
                    val_keys.add(key)
        
        if not train_keys:
            return
        
        metric_names = sorted(set(
            k.replace(self.train_prefix, "").replace(self.val_prefix, "") 
            for k in train_keys | val_keys
        ))
        
        if not metric_names:
            return
        
        n_plots = len(metric_names)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), 
                                 sharex=True, squeeze=False)
        axes = axes.flatten()
        
        for idx, metric in enumerate(metric_names):
            ax = axes[idx]
            
            train_key = f"{self.train_prefix}{metric}"
            steps = [m["step"] for m in self.metrics_log if train_key in m and "step" in m]
            values = [m[train_key] for m in self.metrics_log if train_key in m and "step" in m]
            if steps:
                ax.plot(steps, values, label=f"Train {metric}", alpha=0.7)
            
            val_key = f"{self.val_prefix}{metric}"
            val_steps = [m["step"] for m in self.metrics_log if val_key in m and "step" in m]
            val_values = [m[val_key] for m in self.metrics_log if val_key in m and "step" in m]
            if val_steps:
                ax.plot(val_steps, val_values, label=f"Val {metric}", 
                       marker='o', linestyle='--')
            
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True)
            ax.set_title(f"{metric.upper()}")
        
        axes[-1].set_xlabel("Step")
        plt.tight_layout()
        
        plot_path = self.output_dir / "loss_curves.png"
        plt.savefig(plot_path)
        plt.close(fig)
    
    def _create_scalar_plot(self, metric_key):
        """Create individual plot for a scalar metric."""
        import matplotlib.pyplot as plt
        
        steps = [m["step"] for m in self.metrics_log if metric_key in m and "step" in m]
        values = [m[metric_key] for m in self.metrics_log if metric_key in m and "step" in m]
        
        if not steps or len(steps) < 2:
            return
        
        plt.figure(figsize=(7, 4))
        plt.plot(steps, values, linewidth=2)
        plt.xlabel("Step")
        plt.ylabel(metric_key)
        plt.title(f"{metric_key.replace('_', ' ').title()} Over Training")
        plt.grid(True)
        plt.tight_layout()
        
        plot_path = self.output_dir / f"{metric_key}_curve.png"
        plt.savefig(plot_path)
        plt.close()
    
    def create_plot_with_baseline(self, metric_key, baseline_value=None, baseline_label=None):
        """Create plot for a metric with optional baseline line."""
        import matplotlib.pyplot as plt
        
        steps = [m["step"] for m in self.metrics_log if metric_key in m and "step" in m]
        values = [m[metric_key] for m in self.metrics_log if metric_key in m and "step" in m]
        
        if not steps or len(steps) < 2:
            return
        
        plt.figure(figsize=(7, 4))
        plt.plot(steps, values, linewidth=2, label=metric_key.replace('_', ' ').title())
        if baseline_value is not None:
            plt.axhline(baseline_value, linestyle="--", label=baseline_label or "Baseline")
        plt.xlabel("Step")
        plt.ylabel(metric_key.replace('_', ' ').title())
        plt.title(f"{metric_key.replace('_', ' ').title()} Over Training")
        if baseline_value is not None:
            plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plot_path = self.output_dir / f"{metric_key}_curve.png"
        plt.savefig(plot_path)
        plt.close()


def save_checkpoint(state_dict, path, metadata=None):

    import torch
    
    checkpoint = {"state_dict": state_dict}
    if metadata:
        checkpoint.update(metadata)
    
    safe_mkdir(Path(path).parent)
    torch.save(checkpoint, path)