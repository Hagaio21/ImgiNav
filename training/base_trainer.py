import os
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from common.utils import safe_mkdir, extract_tensor_from_batch, MetricsLogger, save_checkpoint


class BaseTrainer:
    def __init__(self, epochs=10, log_interval=10, sample_interval=100, eval_interval=1,
                 output_dir="outputs", ckpt_dir=None, device=None):
        self.epochs = epochs
        self.log_interval = log_interval
        self.sample_interval = sample_interval
        self.eval_interval = eval_interval
        self.output_dir = output_dir
        self.ckpt_dir = ckpt_dir or os.path.join(output_dir, "checkpoints")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        safe_mkdir(Path(self.output_dir))
        safe_mkdir(Path(self.ckpt_dir))
        safe_mkdir(Path(self.output_dir) / "samples")
        
        self.metrics_logger = MetricsLogger(self.output_dir)
    
    def _get_batch(self, batch, key="layout"):
        return extract_tensor_from_batch(batch, device=self.device, key=key)
    
    def _create_progress_bar(self, dataloader, epoch, total_epochs, desc_prefix="Epoch"):
        return tqdm(
            dataloader,
            desc=f"{desc_prefix} {epoch}/{total_epochs}",
            unit="batch",
            file=sys.stdout,
            ncols=100,
            dynamic_ncols=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
    
    def _log_metrics(self, metrics, step, epoch, prefix="train"):
        log_entry = {"epoch": epoch, "step": step}
        for key, val in metrics.items():
            log_entry[f"{prefix}_{key}"] = val
        self.metrics_logger.log(log_entry)
        return log_entry
    
    def _format_metric_string(self, metrics):
        return ", ".join([f"{k}={v:.5f}" for k, v in metrics.items()])
    
    def _save_checkpoint_base(self, state_dict, filename, metadata=None):
        path = os.path.join(self.ckpt_dir, filename)
        save_checkpoint(state_dict, path, metadata=metadata)
        print(f"[Checkpoint] Saved: {path}")
        return path
    
    def _should_log(self, step):
        return step % self.log_interval == 0
    
    def _should_sample(self, step):
        return self.sample_interval and step % self.sample_interval == 0
    
    def _should_validate(self, epoch):
        return self.eval_interval and epoch % self.eval_interval == 0
    
    def _accumulate_metrics(self, metrics_dict, accumulator):
        for key, val in metrics_dict.items():
            accumulator[key] = accumulator.get(key, 0) + val
        return accumulator
    
    def _average_metrics(self, accumulator, n_batches):
        return {k: v / max(n_batches, 1) for k, v in accumulator.items()}
    

