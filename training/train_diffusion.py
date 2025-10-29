import os, sys, yaml, random, torch, argparse, pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import torch
from pathlib import Path
from common.utils import safe_mkdir
from models.datasets import LayoutDataset, collate_skip_none, build_datasets, build_dataloaders, save_split_csvs
from models.autoencoder import AutoEncoder
from models.components.unet import DualUNet
from models.components.scheduler import LinearScheduler, CosineScheduler, QuadraticScheduler
from training.diffusion_trainer import DiffusionTrainer


# ---------------------------- Dataset ---------------------------- #
# Dataset building functions are now in models.datasets.utils


# ---------------------------- Builders ---------------------------- #

def build_autoencoder(ae_cfg):
    ae_cfg_path = ae_cfg["config"]
    ae_ckpt_path = ae_cfg["checkpoint"]
    ae = AutoEncoder.from_config(ae_cfg_path)
    state = torch.load(ae_ckpt_path, map_location="cpu")
    ae.load_state_dict(state.get("model", state))
    ae.eval()
    return ae


def build_scheduler(sched_cfg):
    sched_type = sched_cfg.get("type", "cosine").lower()
    num_steps = sched_cfg.get("num_steps", 1000)
    if sched_type == "cosine":
        return CosineScheduler(num_steps=num_steps)
    if sched_type == "linear":
        return LinearScheduler(num_steps=num_steps)
    if sched_type == "quadratic":  # <-- ADD THIS LINE
        return QuadraticScheduler(num_steps=num_steps)  # <-- AND ADD THIS LINE
    raise ValueError(f"Unknown scheduler type: {sched_type}")


def build_unet(unet_cfg):
    return DualUNet.from_config(unet_cfg)


# ---------------------------- Main ---------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Run Latent Diffusion Experiment")
    parser.add_argument("--config", required=True, help="Path to diffusion experiment config YAML")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # --- Parse experiment ---
    exp_cfg = cfg.get("experiment", {})
    exp_name = exp_cfg.get("name", "UnnamedDiffusion")
    base_path = exp_cfg.get("base_path", "./experiments")
    exp_path = os.path.join(base_path, exp_name)

    out_dir = os.path.join(exp_path, "output")
    ckpt_dir = os.path.join(exp_path, "checkpoints")
    safe_mkdir(Path(out_dir))
    safe_mkdir(Path(ckpt_dir))
    
    # --- Load configs ---
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    
    loss_cfg = training_cfg.get("loss", {"type": "mse"})


    autoencoder = build_autoencoder(model_cfg["autoencoder"])
    diff_cfg = model_cfg["diffusion"]

    unet_cfg = diff_cfg["unet"]
    scheduler_cfg = diff_cfg["scheduler"]

    scheduler = build_scheduler(scheduler_cfg)
    unet = build_unet(unet_cfg)

    # --- Data ---
    train_ds, val_ds, train_loader, val_loader = build_dataloaders(dataset_cfg)
    save_split_csvs(train_ds, val_ds, out_dir)

    # --- Trainer ---
    trainer = DiffusionTrainer(
        unet=unet,
        autoencoder=autoencoder,
        scheduler=scheduler,
        epochs=training_cfg.get("epochs", 50),
        log_interval=training_cfg.get("log_interval", 100),
        eval_interval=training_cfg.get("eval_interval", 1000),
        sample_interval=training_cfg.get("sample_interval", 2000),
        num_samples=training_cfg.get("num_samples", 8),
        grad_clip=training_cfg.get("grad_clip"),
        use_embeddings=dataset_cfg["return_embeddings"],
        loss_cfg=loss_cfg,
        ckpt_dir=ckpt_dir,
        output_dir=out_dir,
        experiment_name=exp_name,
    )

    if loss_cfg.get("type") == "hybrid" and dataset_cfg["return_embeddings"]:
        print("\n" + "="*60)
        print("!! WARNING: 'hybrid' loss is set but 'return_embeddings' is true.")
        print("   Perceptual (VGG) loss requires original RGB images to work.")
        print("   VGG LOSS WILL BE SKIPPED. Training will only use MSE loss.")
        print("   To fix, set 'return_embeddings: false' in your config file.")
        print("="*60 + "\n")
        
    trainer.fit(train_loader, val_loader)

    # --- Save unified experiment YAML ---
    exp_save_path = os.path.join(out_dir, "experiment_config.yaml")
    with open(exp_save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    print(f"[Config] Saved experiment config to {exp_save_path}")


if __name__ == "__main__":
    main()
