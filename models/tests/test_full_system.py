import torch
from pathlib import Path
import tempfile
import yaml
from models.datasets.datasets import ManifestDataset
from models.diffusion import DiffusionModel
from models.losses.base_loss import LOSS_REGISTRY


def make_dummy_dataset(tmp_path: Path):
    """Create a minimal manifest with a few synthetic images."""
    from PIL import Image
    import pandas as pd

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(3):
        img = Image.new("RGB", (16, 16), color=(i*60, 0, 0))
        img.save(img_dir / f"img_{i}.png")

    df = pd.DataFrame({
        "layout_path": [str(img_dir / f"img_{i}.png") for i in range(3)],
        "room_id": [0, 1, 2],
        "type": ["kitchen", "bathroom", "bedroom"],
        "is_empty": [False, False, False]
    })
    manifest = tmp_path / "manifest.csv"
    df.to_csv(manifest, index=False)
    return manifest


def make_diffusion_config(tmp_path: Path):
    """Return a minimal YAML config for a working diffusion pipeline."""
    cfg = {
        "type": "DiffusionModel",
        "autoencoder": {
            "type": "Autoencoder",
            "encoder": {
                "type": "Encoder",
                "in_channels": 3,
                "latent_channels": 4,
                "base_channels": 16,
                "downsampling_steps": 2,
                "activation": "SiLU",
                "norm_groups": 4
            },
            "decoder": {
                "type": "Decoder",
                "latent_channels": 4,
                "base_channels": 16,
                "upsampling_steps": 2,
                "activation": "SiLU",
                "norm_groups": 4,
                "heads": [
                    {"type": "RGBHead", "name": "rgb", "out_channels": 3, "final_activation": "tanh"}
                ]
            }
        },
        "unet": {
            "type": "DualUNet",
            "in_channels": 4,
            "out_channels": 4,
            "base_channels": 16,
            "depth": 2,
            "num_res_blocks": 1,
            "time_dim": 32,
            "fusion_mode": "none"
        },
        "scheduler": {
            "type": "CosineScheduler",
            "num_steps": 50
        }
    }
    path = tmp_path / "diffusion_model.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def test_full_system(tmp_path):
    """Integration test: dataset + model + loss + optimizer end-to-end."""
    manifest = make_dummy_dataset(tmp_path)

    ds_cfg = {
        "manifest": str(manifest),
        "outputs": {"rgb": "layout_path", "label": "room_id"},
        "filters": {"is_empty": [False]},
    }
    dataset = ManifestDataset(**ds_cfg)
    loader = dataset.make_dataloader(batch_size=2)

    model_cfg_path = make_diffusion_config(tmp_path)
    model = DiffusionModel.load_config(model_cfg_path)
    model.train()

    loss_cfg = {
        "type": "CompositeLoss",
        "losses": [
            {"type": "L1Loss", "key": "rgb", "target": "rgb", "weight": 1.0},
            {"type": "MSELoss", "key": "pred_noise", "target": "noise", "weight": 0.5},
        ]
    }
    loss_fn = LOSS_REGISTRY["CompositeLoss"].from_config(loss_cfg)

    optimizer = torch.optim.AdamW(model.parameter_groups(), lr=1e-4)

    batch = next(iter(loader))
    x0 = batch["rgb"]
    noise = torch.randn_like(x0)
    t = torch.randint(0, model.scheduler.num_steps, (x0.size(0),))

    outputs = model(x0, t, noise=noise)
    # Use encoded noise from outputs (in latent space) to match pred_noise shape
    targets = {"rgb": x0, "noise": outputs["noise"]}

    loss, logs = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()

    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert "rgb" in outputs
    assert "pred_noise" in outputs
    assert len(logs) >= 1
