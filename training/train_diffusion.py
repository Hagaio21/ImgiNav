import os, sys, yaml, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.datasets import build_dataloaders, save_split_csvs
from training.diffusion_trainer import DiffusionTrainer
from training.utils import (
    build_autoencoder,
    build_scheduler,
    build_unet,
    setup_experiment_directories,
    save_experiment_config,
    setup_training_environment
)

def main():
    parser = argparse.ArgumentParser(description="Run Latent Diffusion Experiment")
    parser.add_argument("--config", required=True, help="Path to diffusion experiment config YAML")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg.get("experiment", {})
    exp_name = exp_cfg.get("name", "UnnamedDiffusion")
    base_path = exp_cfg.get("base_path", "./experiments")
    exp_path = os.path.join(base_path, exp_name)

    out_dir = os.path.join(exp_path, "output")
    ckpt_dir = os.path.join(exp_path, "checkpoints")
    out_dir, ckpt_dir = setup_experiment_directories(out_dir, ckpt_dir)
    
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

    save_experiment_config(cfg, out_dir)


if __name__ == "__main__":
    main()
