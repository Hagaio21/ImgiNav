import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer
from unified_dataset import UnifiedLayoutDataset
from autoencoder import AutoEncoder
from unet import UNet
from condition_mixer import LinearConcatMixer, NonLinearConcatMixer
from pipeline import DiffusionPipeline
from pipeline_trainer import PipelineTrainer
from modules.scheduler import LinearScheduler, CosineScheduler
from stage8_create_graph_embeddings import graph2text
import torch.nn as nn


def load_scheduler(name: str, num_steps: int):
    name = name.lower()
    if name == "cosine":
        return CosineScheduler(num_steps=num_steps)
    if name == "linear":
        return LinearScheduler(num_steps=num_steps)
    raise ValueError(f"Unknown scheduler type: {name}")


def select_mixer(name: str, out_channels: int, latent_base: int,
                 pov_dim: int, graph_dim: int, hidden_dim_mlp: int | None = None):
    size = (latent_base, latent_base)
    name = name.lower()
    if name == "mlp":
        return NonLinearConcatMixer(out_channels, size, pov_dim, graph_dim,
                                    hidden_dim_mlp=hidden_dim_mlp)
    return LinearConcatMixer(out_channels, size, pov_dim, graph_dim)


def build_dataloader(cfg):
    ds = UnifiedLayoutDataset(
        room_manifest=cfg["room_manifest"],
        scene_manifest=cfg["scene_manifest"],
        data_mode=cfg["data_mode"],
        pov_type=cfg["pov_type"]
    )
    return DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=True,
        drop_last=True
    )


def build_pipeline(cfg, device, embedder_manager):  # ✓ Accept embedder
    model_cfg = cfg["model"]
    ae_cfg = model_cfg["autoencoder"]
    autoencoder = AutoEncoder.from_config(ae_cfg["config"]).to(device)
    if os.path.exists(ae_cfg["ckpt"]):
        ae_state = torch.load(ae_cfg["ckpt"], map_location=device)
        autoencoder.load_state_dict(ae_state["model"] if "model" in ae_state else ae_state)
    autoencoder.eval()

    diff_cfg = model_cfg["diffusion"]
    scheduler = load_scheduler(diff_cfg["scheduler"], diff_cfg["num_steps"])

    unet = UNet.from_config(diff_cfg["unet_config"],
                            latent_channels=diff_cfg["latent_channels"],
                            latent_base=model_cfg.get("latent_base", 32)).to(device)

    pov_dim = 512
    graph_dim = 384
    hidden_dim_mlp = model_cfg.get("mixer_hidden_dim")
    mixer = select_mixer(model_cfg["mixer"],
                        out_channels=diff_cfg["latent_channels"],
                        latent_base=model_cfg.get("latent_base", 32),
                        pov_dim=pov_dim,
                        graph_dim=graph_dim,
                        hidden_dim_mlp=hidden_dim_mlp).to(device)

    pipeline = DiffusionPipeline(
        autoencoder=autoencoder,
        unet=unet,
        mixer=mixer,
        scheduler=scheduler,
        embedder_manager=embedder_manager,  # ✓ Pass embedder
        device=device
    )
    return pipeline

class EmbedderManager:
    def __init__(self, pov_name: str, graph_name: str, device):
        self.device = device
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()
        self.resnet.eval().to(device)
        self.graph_encoder = SentenceTransformer(graph_name).to(device)
        self.graph_encoder.eval()
        self.cache = {}
        self.pov_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    @torch.no_grad()
    def embed_pov(self, img):
        img = self.pov_transform(img).unsqueeze(0).to(self.device)
        return self.resnet(img).squeeze(0)

    @torch.no_grad()
    def embed_graph(self, graph_path):
        if graph_path in self.cache:
            text = self.cache[graph_path]
        else:
            text = graph2text(graph_path)
            self.cache[graph_path] = text
        emb = self.graph_encoder.encode([text], convert_to_tensor=True, device=self.device)
        return emb.squeeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["dataset"].get("seed", 42))

    train_loader = build_dataloader(cfg["dataset"])
    pipeline = build_pipeline(cfg, device, embed)
    embed = EmbedderManager(cfg["model"]["embedders"]["pov"],
                            cfg["model"]["embedders"]["graph"],
                            device)

    trainer_cfg = cfg["training"]
    trainer = PipelineTrainer(
    pipeline=pipeline,
    optimizer=None,
    loss_fn=None,
    epochs=trainer_cfg["epochs"],
    lr=trainer_cfg["lr"],
    weight_decay=trainer_cfg.get("weight_decay", 0.0),
    grad_clip=trainer_cfg.get("grad_clip"),
    log_interval=trainer_cfg.get("log_interval", 100),
    eval_interval=trainer_cfg.get("eval_interval", 1000),
    sample_interval=trainer_cfg.get("sample_interval", 2000),
    ckpt_dir=trainer_cfg["ckpt_dir"],
    output_dir=trainer_cfg["output_dir"],
    mixed_precision=trainer_cfg.get("mixed_precision", False),
    ema_decay=trainer_cfg.get("ema_decay"),
    cond_dropout_pov=trainer_cfg.get("cond_dropout_pov", 0.0),
    cond_dropout_graph=trainer_cfg.get("cond_dropout_graph", 0.0),
    cond_dropout_both=trainer_cfg.get("cond_dropout_both", 0.0)
    )

    # ---- new line: modality control ----
    trainer.use_modalities = trainer_cfg.get("use_modalities", "both")

    # ---- run training ----
    trainer.fit(train_loader)


    save_dir = trainer_cfg["ckpt_dir"]
    os.makedirs(save_dir, exist_ok=True)
    pipeline_config = {
        "autoencoder": {
            "config": cfg["model"]["autoencoder"]["config"],
            "checkpoint": os.path.join(save_dir, "ae_latest.pt")
        },
        "unet": {
            "config": cfg["model"]["diffusion"]["unet_config"],
            "checkpoint": os.path.join(save_dir, "unet_latest.pt")
        },
        "mixer": {
            "type": cfg["model"]["mixer"],
            "checkpoint": os.path.join(save_dir, "mixer_latest.pt")
        },
        "scheduler": {
            "type": cfg["model"]["diffusion"]["scheduler"],
            "num_steps": cfg["model"]["diffusion"]["num_steps"]
        },
        "embedders": cfg["model"]["embedders"],
        "latent_channels": cfg["model"]["diffusion"]["latent_channels"],
        "latent_base": cfg["model"].get("latent_base", 32)
    }

    with open(os.path.join(save_dir, "pipeline_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(pipeline_config, f)

    print(f"[Done] Saved pipeline config and weights to {save_dir}")


if __name__ == "__main__":
    main()
