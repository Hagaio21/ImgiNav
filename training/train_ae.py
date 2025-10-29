# train_ae.py
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import yaml
import random
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T

# Add project root to path
from models.datasets import LayoutDataset, collate_skip_none, build_datasets, build_dataloaders, save_split_csvs
from models.autoencoder import AutoEncoder
from training.autoencoder_trainer import AutoEncoderTrainer
from models.losses.custom_loss import StandardVAELoss, SegmentationVAELoss
import sys
from pathlib import Path
from common.taxonomy import load_valid_colors
from common.utils import safe_mkdir


def build_loss_function(loss_cfg):
    """
    Factory function to build loss function from config.

    Expected loss_cfg format:
    {
        "type": "standard" | "segmentation",
        "kl_weight": float,
        # For segmentation:
        "lambda_seg": float,
        "lambda_mse": float,
        "taxonomy_path": str,     # path to taxonomy.json
        "include_background": bool
    }
    """
    loss_type = loss_cfg.get("type", "standard").lower()
    kl_weight = loss_cfg.get("kl_weight", 1e-6)

    if loss_type == "standard":
        print(f"[Loss] Using StandardVAELoss (kl_weight={kl_weight})")
        return StandardVAELoss(kl_weight=kl_weight)

    elif loss_type == "segmentation":
        lambda_seg = loss_cfg.get("lambda_seg", 1.0)
        lambda_mse = loss_cfg.get("lambda_mse", 1.0)
        taxonomy_path = loss_cfg.get("taxonomy_path")
        include_bg = loss_cfg.get("include_background", True)

        if not taxonomy_path:
            raise ValueError("[Loss] 'taxonomy_path' must be provided for segmentation loss")

        # Load filtered id→color mapping
        id_to_color, valid_ids = load_valid_colors(taxonomy_path, include_background=include_bg)

        print(f"[Loss] Using SegmentationVAELoss (kl_weight={kl_weight}, "
              f"lambda_seg={lambda_seg}, lambda_mse={lambda_mse})")
        print(f"[Loss] Loaded {len(valid_ids)} valid class IDs: {valid_ids}")

        return SegmentationVAELoss(
            id_to_color=id_to_color,
            kl_weight=kl_weight,
            lambda_seg=lambda_seg,
            lambda_mse=lambda_mse,
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose 'standard' or 'segmentation'.")

def main():
    parser = argparse.ArgumentParser(description="Run AutoEncoder or VAE experiment")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]

    # --- Model type ---
    model_type = model_cfg.get("type", "vae").lower()
    is_ae = model_type == "ae"

    # if AE, disable KL
    if is_ae:
        training_cfg["loss"]["kl_weight"] = 0.0

    # --- Experiment setup ---
    out_dir = training_cfg.get("output_dir", "ae_outputs")
    ckpt_dir = training_cfg.get("ckpt_dir", os.path.join(out_dir, "checkpoints"))
    safe_mkdir(Path(out_dir))
    safe_mkdir(Path(ckpt_dir))

    # Save experiment config
    model_cfg_path = os.path.join(out_dir, "experiment_config.yaml")
    with open(model_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    print(f"[Config] Saved experiment config to {model_cfg_path}")

    # --- Dataset setup ---
    seed = dataset_cfg.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    transform = T.ToTensor()

    train_ds, val_ds = build_datasets(dataset_cfg, transform=transform)

    batch_size = dataset_cfg.get("batch_size", 32)
    num_workers = dataset_cfg.get("num_workers", 4)
    shuffle = dataset_cfg.get("shuffle", True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_skip_none
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_skip_none
    )

    save_split_csvs(train_ds, val_ds, out_dir)

    # --- Model ---
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

    ae.encoder.print_summary()
    ae.decoder.print_summary()

    # assign deterministic flag based on model type
    ae.deterministic = is_ae

    # --- Build loss ---
    loss_cfg = training_cfg.get("loss", {"type": "standard", "kl_weight": 1e-6})
    if is_ae:
        loss_cfg["kl_weight"] = 0.0

    loss_fn = build_loss_function(loss_cfg)

    # --- Trainer ---
    trainer = AutoEncoderTrainer(
        autoencoder=ae,
        loss_fn=loss_fn,
        epochs=training_cfg.get("epochs", 50),
        log_interval=training_cfg.get("log_interval", 10),
        sample_interval=training_cfg.get("sample_interval", 100),
        eval_interval=training_cfg.get("eval_interval", 1),
        lr=training_cfg.get("lr", 1e-4),
        output_dir=out_dir,
        ckpt_dir=ckpt_dir,
    )

    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()