import torch
from pathlib import Path
import json
from models.datasets.datasets import ManifestDataset
from models.diffusion import DiffusionModel
from models.losses.base_loss import LOSS_REGISTRY


def test_real_multihead_diffusion(tmp_path):
    """
    Test training with the real manifest and multi-head decoder.
    Uses:
      - pov_image for RGB
      - layout_image for segmentation
      - room_id for classification
    """
    # Path to your manifest and taxonomy
    manifest = Path(r"C:\Users\Hagai.LAPTOP-QAG9263N\Desktop\Thesis\repositories\ImgiNav\test_dataset\manifests\manifest.csv")
    taxonomy_path = Path(r"C:\Users\Hagai.LAPTOP-QAG9263N\Desktop\Thesis\repositories\ImgiNav\config\taxonomy.json")
    assert manifest.exists(), f"Manifest not found: {manifest}"
    assert taxonomy_path.exists(), f"taxonomy.json not found: {taxonomy_path}"

    # Load taxonomy to count segmentation classes
    with open(taxonomy_path, "r") as f:
        taxonomy = json.load(f)
    num_classes = len(taxonomy)

    # Dataset configuration
    ds_cfg = {
        "manifest": str(manifest),
        "outputs": {
            "rgb": "pov_image",
            "segmentation": "layout_image",
            "label": "room_id",
        },
        "filters": {"is_empty": [False]},
    }
    dataset = ManifestDataset(**ds_cfg)
    dataloader = dataset.make_dataloader(batch_size=1, shuffle=True, num_workers=0)

    # Model configuration
    model_cfg = {
        "autoencoder": {
            "encoder": {
                "in_channels": 3,
                "latent_channels": 4,
                "base_channels": 16,
                "downsampling_steps": 2,
            },
            "decoder": {
                "latent_channels": 4,
                "base_channels": 16,
                "upsampling_steps": 2,
                "heads": [
                    {"type": "RGBHead", "name": "rgb", "out_channels": 3, "final_activation": "tanh"},
                    {"type": "SegmentationHead", "name": "segmentation", "num_classes": num_classes},
                    {"type": "ClassificationHead", "name": "classification", "num_classes": 20},  # or real room classes
                ],
            },
        },
        "unet": {
            "in_channels": 4,
            "out_channels": 4,
            "base_channels": 16,
            "depth": 2,
            "num_res_blocks": 1,
            "time_dim": 32,
        },
        "scheduler": {"type": "CosineScheduler", "num_steps": 50},
    }
    model = DiffusionModel.from_config(model_cfg)
    model.train()

    # Loss configuration
    loss_cfg = {
        "type": "CompositeLoss",
        "losses": [
            {"type": "L1Loss", "key": "rgb", "target": "rgb", "weight": 1.0},
            {"type": "CrossEntropyLoss", "key": "segmentation", "target": "segmentation", "weight": 0.5},
            {"type": "CrossEntropyLoss", "key": "classification", "target": "label", "weight": 1.0},
        ],
    }
    loss_fn = LOSS_REGISTRY["CompositeLoss"].from_config(loss_cfg)
    optimizer = torch.optim.AdamW(model.parameter_groups(), lr=1e-4)

    # Mini training iteration
    batch = next(iter(dataloader))
    x0 = batch["rgb"]
    noise = torch.randn_like(x0)
    t = torch.randint(0, model.scheduler.num_steps, (x0.size(0),))

    outputs = model(x0, t, noise=noise)
    targets = {
        "rgb": batch["rgb"],
        "segmentation": batch["segmentation"],
        "label": batch["label"],
        "noise": noise,
    }

    loss, logs = loss_fn(outputs, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")
    print("Logs:", logs)

    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert "rgb" in outputs and "segmentation" in outputs and "classification" in outputs
