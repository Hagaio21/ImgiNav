import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse
import torch
import torchvision.transforms as T
from pathlib import Path

from common.utils import load_config_with_profile
from models.datasets import build_dataloaders, save_split_csvs
from models.autoencoder import AutoEncoder
from models.diffusion import LatentDiffusion
from training.trainer import Trainer
from training.utils import (
    build_autoencoder,
    build_scheduler,
    build_unet,
    build_loss_function,
    setup_experiment_directories,
    save_experiment_config,
    setup_training_environment,
    get_model_type,
    configure_ae_mode,
)


def build_model(model_cfg, device):
    """
    Build model from config based on model type.
    Returns the model and any additional components needed (like autoencoder for diffusion).
    """
    model_type = model_cfg.get("type", "autoencoder").lower()
    
    if model_type in ("autoencoder", "ae", "vae"):
        # AutoEncoder/VAE
        if "from_shape" in model_cfg or all(k in model_cfg for k in ["in_channels", "out_channels", "base_channels"]):
            # Build from shape parameters
            ae = AutoEncoder.from_shape(
                in_channels=model_cfg["in_channels"],
                out_channels=model_cfg["out_channels"],
                base_channels=model_cfg["base_channels"],
                latent_channels=model_cfg["latent_channels"],
                image_size=model_cfg["image_size"],
                latent_base=model_cfg["latent_base"],
                norm=model_cfg.get("norm"),
                act=model_cfg.get("act", "relu"),
                dropout=model_cfg.get("dropout", 0.0),
                num_classes=model_cfg.get("num_classes", None),
            )
            model_type_detected = model_cfg.get("type", "vae").lower()
            ae.deterministic = (model_type_detected == "ae")
        else:
            # Build from config dict/file
            ae = AutoEncoder.from_config(model_cfg.get("config", model_cfg))
        
        ae.encoder.print_summary()
        ae.decoder.print_summary()
        return ae, None
    
    elif model_type in ("diffusion", "latent_diffusion"):
        # Latent Diffusion
        if "autoencoder" not in model_cfg:
            raise ValueError("Diffusion model requires 'autoencoder' config")
        
        # Build autoencoder (frozen)
        autoencoder = build_autoencoder(model_cfg["autoencoder"])
        
        # Build diffusion components
        diff_cfg = model_cfg.get("diffusion", {})
        unet_cfg = diff_cfg.get("unet", {})
        scheduler_cfg = diff_cfg.get("scheduler", {})
        
        scheduler = build_scheduler(scheduler_cfg)
        unet = build_unet(unet_cfg)
        
        # Determine latent shape
        latent_shape = (
            autoencoder.encoder.latent_channels,
            autoencoder.encoder.latent_base,
            autoencoder.encoder.latent_base,
        )
        
        # Build LatentDiffusion
        diffusion = LatentDiffusion(
            backbone=unet,
            scheduler=scheduler,
            autoencoder=autoencoder,
            latent_shape=latent_shape,
        ).to(device)
        
        return diffusion, autoencoder
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'autoencoder', 'vae', 'ae', or 'diffusion'.")


def build_optimizer(model, training_cfg):
    """Build optimizer from training config."""
    opt_cfg = training_cfg.get("optimizer", {})
    opt_type = opt_cfg.get("type", "adam").lower()
    lr = opt_cfg.get("lr", training_cfg.get("lr", 1e-4))
    
    # Get trainable parameters
    params = model.parameters()
    if hasattr(model, "backbone") and hasattr(model, "autoencoder"):
        # For diffusion, only train backbone
        params = model.backbone.parameters()
    
    if opt_type == "adam":
        weight_decay = opt_cfg.get("weight_decay", 0.0)
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_type == "adamw":
        weight_decay = opt_cfg.get("weight_decay", 0.01)
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif opt_type == "sgd":
        momentum = opt_cfg.get("momentum", 0.9)
        weight_decay = opt_cfg.get("weight_decay", 0.0)
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")


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
    
    # Build model
    model, autoencoder = build_model(model_cfg, device)
    model = model.to(device)
    
    # Configure loss for AE mode
    loss_cfg = training_cfg.get("loss", {})
    model_type = model_cfg.get("type", "vae").lower()
    if model_type == "ae":
        if "kl_weight" not in loss_cfg:
            loss_cfg["kl_weight"] = 0.0
        elif loss_cfg.get("kl_weight", 1e-6) > 0:
            loss_cfg["kl_weight"] = 0.0
            print("[Warning] AE mode detected: setting kl_weight=0.0")
    
    # Build loss function
    loss_fn = build_loss_function(loss_cfg)
    
    # Build optimizer
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

