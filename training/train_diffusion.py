# train_diffusion.py
from __future__ import annotations
import sys, os, yaml, random, torch, pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from torch.utils.data import DataLoader, random_split
from modules.datasets import LayoutDataset, collate_skip_none
from modules.autoencoder import AutoEncoder
from modules.unet import UNet
from modules.scheduler import CosineScheduler, LinearScheduler
from training.diffusion_trainer import DiffusionTrainer


# ---------------------------- Dataset utils ---------------------------- #

def build_datasets(manifest_path, split_ratio, seed, transform):
    dataset = LayoutDataset(manifest_path, transform=transform, mode="all")
    n_total = len(dataset)
    n_train = int(n_total * split_ratio)
    n_val = n_total - n_train
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=generator)


def save_split_csvs(train_ds, val_ds, output_dir):
    train_paths = [train_ds.dataset.entries[i]["layout_path"] for i in train_ds.indices]
    val_paths = [val_ds.dataset.entries[i]["layout_path"] for i in val_ds.indices]
    pd.DataFrame({"layout_path": train_paths}).to_csv(os.path.join(output_dir, "trained_on.csv"), index=False)
    pd.DataFrame({"layout_path": val_paths}).to_csv(os.path.join(output_dir, "evaluated_on.csv"), index=False)


def build_dataloaders(dataset_cfg):
    import torchvision.transforms as T
    transform = T.ToTensor()

    manifest = dataset_cfg["manifest"]
    split_ratio = dataset_cfg.get("split_ratio", 0.9)
    seed = dataset_cfg.get("seed", 42)
    batch_size = dataset_cfg.get("batch_size", 16)
    num_workers = dataset_cfg.get("num_workers", 4)
    shuffle = dataset_cfg.get("shuffle", True)

    random.seed(seed)
    torch.manual_seed(seed)

    train_ds, val_ds = build_datasets(manifest, split_ratio, seed, transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, collate_fn=collate_skip_none)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_skip_none)
    return train_ds, val_ds, train_loader, val_loader


# ---------------------------- Model builders ---------------------------- #

def build_autoencoder(encoder_cfg):
    ae_cfg_path = encoder_cfg["autoencoder_config"]
    ae_ckpt_path = encoder_cfg["autoencoder_ckpt"]
    autoencoder = AutoEncoder.from_config(ae_cfg_path)
    autoencoder.load_state_dict(torch.load(ae_ckpt_path, map_location="cpu"))
    autoencoder.eval()
    return autoencoder, ae_cfg_path


def build_unet(unet_cfg):
    unet = UNet.from_shape(
        in_channels=unet_cfg["latent_channels"],
        out_channels=unet_cfg["latent_channels"],
        base_channels=unet_cfg["base_channels"],
        depth=unet_cfg.get("depth", 3),
        norm=unet_cfg.get("norm", "batch"),
        act=unet_cfg.get("act", "relu"),
    )
    unet.print_summary()
    return unet


def build_scheduler(diff_cfg):
    sched_type = diff_cfg.get("scheduler", "cosine").lower()
    num_steps = diff_cfg.get("num_steps", 1000)
    if sched_type == "cosine":
        return CosineScheduler(num_steps=num_steps)
    if sched_type == "linear":
        return LinearScheduler(num_steps=num_steps)
    raise ValueError(f"Unknown scheduler type: {sched_type}")


def save_model_config(ae_cfg_path, unet_cfg, diff_cfg, model_cfg_path):
    nested_cfg = {
        "autoencoder": AutoEncoder.from_config(ae_cfg_path).to_config(),
        "unet": UNet.from_shape(
            in_channels=unet_cfg["latent_channels"],
            out_channels=unet_cfg["latent_channels"],
            base_channels=unet_cfg["base_channels"],
            depth=unet_cfg.get("depth", 3),
            norm=unet_cfg.get("norm", "batch"),
            act=unet_cfg.get("act", "relu"),
        ).to_config()["unet"],
        "scheduler": diff_cfg.get("scheduler", "cosine"),
        "num_steps": diff_cfg.get("num_steps", 1000),
    }
    with open(model_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(nested_cfg, f)
    print(f"[Config] Saved nested model architecture config → {model_cfg_path}")


# ---------------------------- Main routine ---------------------------- #

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Diffusion experiment")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]

    encoder_cfg = model_cfg["encoder"]
    unet_cfg = model_cfg["unet"]
    diff_cfg = model_cfg["diffusion"]

    out_dir = training_cfg.get("output_dir", "diffusion_outputs")
    ckpt_dir = training_cfg.get("ckpt_dir", os.path.join(out_dir, "checkpoints"))
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    model_cfg_path = os.path.join(ckpt_dir, "model_config.yaml")

    train_ds, val_ds, train_loader, val_loader = build_dataloaders(dataset_cfg)
    save_split_csvs(train_ds, val_ds, out_dir)

    autoencoder, ae_cfg_path = build_autoencoder(encoder_cfg)
    unet = build_unet(unet_cfg)
    scheduler = build_scheduler(diff_cfg)
    save_model_config(ae_cfg_path, unet_cfg, diff_cfg, model_cfg_path)

    trainer = DiffusionTrainer(
        unet=unet,
        autoencoder=autoencoder,
        scheduler=scheduler,
        epochs=training_cfg.get("epochs", 10),
        log_interval=training_cfg.get("log_interval", 100),
        eval_interval=training_cfg.get("eval_interval", 1000),
        sample_interval=training_cfg.get("sample_interval", 2000),
        checkpoint_interval=training_cfg.get("checkpoint_interval",10),
        num_samples=training_cfg.get("num_samples", 8),
        grad_clip=training_cfg.get("grad_clip"),
        ckpt_dir=ckpt_dir,
        output_dir=out_dir,
    )

    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
