import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse
import torch
import torchvision.transforms as T
from pathlib import Path

from common.utils import load_config_with_profile
from models.datasets import build_dataloaders, save_split_csvs
from models.builder import build_model
from training.trainer import Trainer
from training.utils import (
    build_loss_function,
    build_optimizer,
    setup_experiment_directories,
    save_experiment_config,
    setup_training_environment,
)




def main():
    parser = argparse.ArgumentParser(description="Generic training script for all model types")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config_with_profile(args.config)
    
    # Extract sections
    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    experiment_cfg = cfg.get("experiment", {})
    
    # Setup directories (handle both experiment and output_dir patterns)
    if experiment_cfg:
        # New pattern: experiment.base_path/experiment.name/output
        exp_name = experiment_cfg.get("name", "UnnamedExperiment")
        base_path = experiment_cfg.get("base_path", "./experiments")
        exp_path = os.path.join(base_path, exp_name)
        out_dir = os.path.join(exp_path, "output")
        ckpt_dir = os.path.join(exp_path, "checkpoints")
    else:
        # Old pattern: training.output_dir
        out_dir = training_cfg.get("output_dir", "outputs")
        ckpt_dir = training_cfg.get("ckpt_dir", os.path.join(out_dir, "checkpoints"))
    
    out_dir, ckpt_dir = setup_experiment_directories(out_dir, ckpt_dir)
    save_experiment_config(cfg, out_dir)
    
    # Setup environment
    seed = dataset_cfg.get("seed", 42)
    device = setup_training_environment(seed)
    
    # Build dataloaders
    transform = T.ToTensor() if not dataset_cfg.get("return_embeddings", False) else None
    dataset_cfg["pin_memory"] = True
    train_ds, val_ds, train_loader, val_loader = build_dataloaders(dataset_cfg, transform=transform)
    save_split_csvs(train_ds, val_ds, out_dir)
    
    # Build model (completely generic - no model-specific knowledge)
    model, _ = build_model(model_cfg, device)
    
    # Build loss function from config
    loss_cfg = training_cfg.get("loss", {})
    loss_fn = build_loss_function(loss_cfg)
    
    # Build optimizer from config
    optimizer = build_optimizer(model, training_cfg)
    
    # Extract training hyperparameters
    cfg_dropout_prob = training_cfg.get("cfg_dropout_prob", 0.0)
    num_training_samples = training_cfg.get("num_samples", training_cfg.get("num_training_samples", 4))
    
    # Create trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=training_cfg.get("epochs", 50),
        log_interval=training_cfg.get("log_interval", 10),
        sample_interval=training_cfg.get("sample_interval", 100),
        eval_interval=training_cfg.get("eval_interval", 1),
        grad_clip=training_cfg.get("grad_clip"),
        cfg_dropout_prob=cfg_dropout_prob,
        num_training_samples=num_training_samples,
        output_dir=out_dir,
        ckpt_dir=ckpt_dir,
        device=device,
    )
    
    # Train
    trainer.fit(train_loader, val_loader)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

