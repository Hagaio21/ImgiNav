import os, yaml

# ---------------------------------------------------------------------
# Base configuration template
# ---------------------------------------------------------------------
def make_config(exp_name, base_channels, scheduler_type, base_exp_path):
    exp_dir = os.path.join(base_exp_path, exp_name)
    return {
        "experiment": {
            "name": exp_name,
            "base_path": base_exp_path,
        },
        "dataset": {
            "manifest": "/work3/s233249/ImgiNav/datasets/layouts.csv",
            "split_ratio": 0.9,
            "seed": 42,
            "batch_size": 32,
            "num_workers": 8,
            "shuffle": True,
            "taxonomy_path": "/work3/s233249/ImgiNav/ImgiNav/config/taxonomy.json",
        },
        "model": {
            "autoencoder": {
                "config": "/work3/s233249/ImgiNav/experiments/AEVAE_sweep/AE_large_latent_seg/output/autoencoder_config.yaml",
                "checkpoint": "/work3/s233249/ImgiNav/experiments/AEVAE_sweep/AE_large_latent_seg/checkpoints/ae_latest.pt",
            },
            "diffusion": {
                "scheduler": {
                    "type": scheduler_type.lower(),
                    "num_steps": 1000,
                },
                "unet": {
                    "type": "dualunet",
                    "in_channels": 4,
                    "out_channels": 4,
                    "base_channels": base_channels,
                    "depth": 4,
                    "num_res_blocks": 2,     # required by DualUNet
                    "time_dim": 128,         # required by DualUNet
                    "cond_channels": 0,
                    "act": "relu",
                    "norm": "batch",
                    "dropout": 0.1,
                    "fusion_mode": "none",   # renamed key
                    "cond_mult": 1.0,
                },
            },
        },
        "training": {
            "epochs": 50,
            "log_interval": 100,
            "eval_interval": 1000,
            "sample_interval": 2000,
            "grad_clip": 1.0,
            "num_samples": 8,
            "output_dir": os.path.join(exp_dir, "output"),
            "ckpt_dir": os.path.join(exp_dir, "checkpoints"),
        },
    }


# ---------------------------------------------------------------------
# Experiments definition
# ---------------------------------------------------------------------
EXPERIMENTS = [
    ("E1_Linear_64", 64, "linear"),
    ("E2_Cosine_64", 64, "cosine"),
    ("E3_Quadratic_64", 64, "quadratic"),
    ("E4_Linear_128", 128, "linear"),
    ("E5_Cosine_128", 128, "cosine"),
    ("E6_Quadratic_128", 128, "quadratic"),
]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    base_exp_path = "/work3/s233249/ImgiNav/experiments/Diffusion_Uncond"
    config_output_path = (
        r"C:\Users\Hagai.LAPTOP-QAG9263N\Desktop\Thesis\repositories\ImgiNav\config\architecture\diffusion"
    )

    os.makedirs(base_exp_path, exist_ok=True)
    os.makedirs(config_output_path, exist_ok=True)

    for name, base_ch, sched in EXPERIMENTS:
        exp_dir = os.path.join(base_exp_path, name)
        os.makedirs(os.path.join(exp_dir, "output"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)

        cfg = make_config(name, base_ch, sched, base_exp_path)

        cfg_path = os.path.join(config_output_path, f"{name}.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print(f"[✓] Saved config: {cfg_path}")

    print(f"\nAll diffusion configs written to {config_output_path}")


if __name__ == "__main__":
    main()
