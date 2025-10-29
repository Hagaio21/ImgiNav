import os
import yaml
import torch
from pathlib import Path
from common.utils import safe_mkdir, write_json
from common.taxonomy import load_valid_colors
from models.autoencoder import AutoEncoder
from models.components.unet import DualUNet
from models.components.scheduler import LinearScheduler, CosineScheduler, QuadraticScheduler
from models.losses.custom_loss import StandardVAELoss, SegmentationVAELoss


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
    if sched_type == "quadratic":
        return QuadraticScheduler(num_steps=num_steps)
    raise ValueError(f"Unknown scheduler type: {sched_type}")


def build_unet(unet_cfg):
    return DualUNet.from_config(unet_cfg)


def build_loss_function(loss_cfg):
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


def setup_experiment_directories(output_dir, ckpt_dir=None):
    if ckpt_dir is None:
        ckpt_dir = os.path.join(output_dir, "checkpoints")
    safe_mkdir(Path(output_dir))
    safe_mkdir(Path(ckpt_dir))
    return output_dir, ckpt_dir


def save_experiment_config(config, output_dir, filename="experiment_config.yaml"):
    config_path = os.path.join(output_dir, filename)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    print(f"[Config] Saved experiment config to {config_path}")
    return config_path


def load_experiment_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model_type(model_cfg):
    model_type = model_cfg.get("type", "vae").lower()
    is_ae = model_type == "ae"
    return model_type, is_ae


def configure_ae_mode(training_cfg, is_ae):
    if is_ae:
        training_cfg["loss"]["kl_weight"] = 0.0
    return training_cfg


def validate_embedding_shape(z, expected_dims=3):
    if z.dim() == 4 and z.shape[0] == 1:
        z = z.squeeze(0)
    
    if z.dim() != expected_dims:
        raise ValueError(f"Expected {expected_dims}D tensor, got {z.dim()}D with shape {z.shape}")
    
    return z


def setup_training_environment(seed=42, device=None):
    from common.utils import set_seeds
    set_seeds(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def save_model_config(model, output_dir, filename="autoencoder_config.yaml"):
    try:
        config_path = os.path.join(output_dir, filename)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(model.to_config(), f)
        print(f"[Config] Saved: {config_path}", flush=True)
    except Exception as e:
        print(f"[Config] ERROR: Could not save config: {e}", flush=True)

