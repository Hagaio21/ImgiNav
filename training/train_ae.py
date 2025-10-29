# train_ae.py
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import yaml
import argparse
import torchvision.transforms as T

from models.datasets import build_dataloaders, save_split_csvs
from models.autoencoder import AutoEncoder
from training.autoencoder_trainer import AutoEncoderTrainer
from training.utils import (
    build_loss_function,
    setup_experiment_directories,
    save_experiment_config,
    get_model_type,
    configure_ae_mode,
    setup_training_environment
)


def main():
    parser = argparse.ArgumentParser(description="Run AutoEncoder or VAE experiment")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]

    model_type, is_ae = get_model_type(model_cfg)
    training_cfg = configure_ae_mode(training_cfg, is_ae)

    out_dir = training_cfg.get("output_dir", "ae_outputs")
    ckpt_dir = training_cfg.get("ckpt_dir", os.path.join(out_dir, "checkpoints"))
    out_dir, ckpt_dir = setup_experiment_directories(out_dir, ckpt_dir)
    save_experiment_config(cfg, out_dir)

    seed = dataset_cfg.get("seed", 42)
    setup_training_environment(seed)
    transform = T.ToTensor()

    # Set pin_memory for better performance
    dataset_cfg["pin_memory"] = True
    
    # Use build_dataloaders utility instead of manual creation
    train_ds, val_ds, train_loader, val_loader = build_dataloaders(dataset_cfg, transform=transform)

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

    ae.deterministic = is_ae

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