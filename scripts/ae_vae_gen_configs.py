import os
import yaml

# local folder to save the configs (adjust freely)
save_dir = "config/architecture/autoencoders"
os.makedirs(save_dir, exist_ok=True)

# HPC paths remain unchanged inside configs
taxonomy_path = "/work3/s233249/ImgiNav/ImgiNav/config/taxonomy.json"
manifest_path = "/work3/s233249/ImgiNav/datasets/layouts.csv"
exp_base = "/work3/s233249/ImgiNav/experiments/AEVAE_sweep"

shared_dataset = {
    "batch_size": 64,
    "manifest": manifest_path,
    "num_workers": 4,
    "seed": 42,
    "shuffle": True,
    "split_ratio": 0.9,
    "taxonomy_path": taxonomy_path,
}

shared_training = {
    "epochs": 50,
    "eval_interval": 1,
    "log_interval": 10,
    "lr": 1e-4,
    "sample_interval": 200,
}

def make_config(name, model_type, latent_channels, latent_base,
                dropout=0.0, kl_weight=0.0, use_seg=False):
    loss_type = "segmentation" if use_seg else "standard"
    output_dir = f"{exp_base}/{name}/output"
    ckpt_dir = f"{exp_base}/{name}/checkpoints"

    cfg = {
        "dataset": shared_dataset,
        "model": {
            "type": model_type,
            "in_channels": 3,
            "out_channels": 3,
            "base_channels": 64,
            "latent_channels": latent_channels,
            "latent_base": latent_base,
            "image_size": 512,
            "act": "relu",
            "norm": "batch",
            "dropout": dropout,
            "num_classes": 13,
            "use_sigmoid": True,
        },
        "training": {
            **shared_training,
            "output_dir": output_dir,
            "ckpt_dir": ckpt_dir,
            "loss": {
                "type": loss_type,
                "kl_weight": kl_weight,
                "lambda_mse": 1.0,
            },
        },
    }

    if use_seg:
        cfg["training"]["loss"].update({
            "lambda_seg": 1.0,
            "taxonomy_path": taxonomy_path,
            "include_background": True,
        })

    config_path = os.path.join(save_dir, f"{name}.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[Saved] {config_path}")

# --- Experiments ---
experiments = [
    ("AE_small_latent", "ae", 4, 16, 0.0, 0.0, False),
    ("AE_large_latent_seg", "ae", 8, 32, 0.0, 0.0, True),
    ("AE_dropout", "ae", 4, 32, 0.2, 0.0, False),
    ("VAE_small_KL_seg", "vae", 4, 32, 0.0, 1e-6, True),
    ("VAE_med_KL", "vae", 4, 32, 0.0, 1e-4, False),
    ("VAE_large_KL_seg", "vae", 4, 32, 0.0, 1e-3, True),
]

for args in experiments:
    make_config(*args)
