from __future__ import annotations
import sys
import os
import yaml
import random
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from modules.datasets import LayoutDataset, collate_skip_none
from modules.autoencoder import AutoEncoder
from training.autoencoder_trainer import AutoEncoderTrainer


def build_datasets(manifest_path, split_ratio, seed, transform, dataset_cfg):
    dataset = LayoutDataset(
        manifest_path,
        transform=transform,
        mode="all",
        one_hot=dataset_cfg.get("one_hot", False),
        taxonomy_path=dataset_cfg.get("taxonomy_path"),
    )

    n_total = len(dataset)
    n_train = int(n_total * split_ratio)
    n_val = n_total - n_train

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    return train_ds, val_ds


def save_split_csvs(train_ds, val_ds, output_dir):
    train_paths = [train_ds.dataset.entries[i]["layout_path"] for i in train_ds.indices]
    val_paths = [val_ds.dataset.entries[i]["layout_path"] for i in val_ds.indices]

    train_df = pd.DataFrame({"layout_path": train_paths})
    val_df = pd.DataFrame({"layout_path": val_paths})

    train_df.to_csv(os.path.join(output_dir, "trained_on.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "evaluated_on.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Run AutoEncoder experiment")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]

    # --- Experiment setup ---
    out_dir = training_cfg.get("output_dir", "ae_outputs")
    ckpt_dir = training_cfg.get("ckpt_dir", os.path.join(out_dir, "checkpoints"))
    os.makedirs(ckpt_dir, exist_ok=True)

    model_cfg_path = os.path.join(out_dir, "experiment_config.yaml")
    with open(model_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    print(f"[Config] Saved experiment config to {model_cfg_path}")

    # --- Dataset config ---
    manifest = dataset_cfg["manifest"]
    batch_size = dataset_cfg.get("batch_size", 32)
    split_ratio = dataset_cfg.get("split_ratio", 0.9)
    num_workers = dataset_cfg.get("num_workers", 4)
    shuffle = dataset_cfg.get("shuffle", True)
    seed = dataset_cfg.get("seed", 42)

    random.seed(seed)
    torch.manual_seed(seed)
    transform = T.ToTensor()

    train_ds, val_ds = build_datasets(
    manifest_path=dataset_cfg["manifest"],
    split_ratio=dataset_cfg.get("split_ratio", 0.9),
    seed=dataset_cfg.get("seed", 42),
    transform=transform,
    dataset_cfg=dataset_cfg,
    )


    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_skip_none
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_skip_none
    )

    os.makedirs(out_dir, exist_ok=True)
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
    )

    ae.encoder.print_summary()
    ae.decoder.print_summary()

    # --- Choose loss function ---
    loss_type = training_cfg.get("loss", "mse").lower()
    if loss_type == "crossentropy":
        recon_loss_fn = nn.CrossEntropyLoss()
    elif loss_type == "mse":
        recon_loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    kl_weight = training_cfg.get("kl_weight", 1e-6)
    print(f"[Trainer] Using KL weight: {kl_weight}, loss={loss_type}")

    trainer = AutoEncoderTrainer(
        autoencoder=ae,
        recon_loss_fn=recon_loss_fn,
        kl_weight=kl_weight,
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
