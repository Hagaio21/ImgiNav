import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer
from modules.unified_dataset import UnifiedLayoutDataset, collate_fn
from modules.autoencoder import AutoEncoder
from modules.unet import UNet
from modules.condition_mixer import LinearConcatMixer, NonLinearConcatMixer
from pipeline.pipeline import DiffusionPipeline
from training.pipeline_trainer import PipelineTrainer
from modules.scheduler import LinearScheduler, CosineScheduler
from utils.utlis import load_taxonomy
import torch.nn as nn
from pathlib import Path

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


def build_dataloader(cfg, use_embeddings=False, shuffle=True):
    """Build dataloader with new unified manifest structure."""
    ds = UnifiedLayoutDataset(
        manifest_path=cfg["manifest_path"],
        use_embeddings=use_embeddings,
        sample_type=cfg.get("sample_type", "both"),  # 'room', 'scene', or 'both'
    )
    return DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=shuffle,
        drop_last=True,
        collate_fn=collate_fn
    )


def build_pipeline(cfg, device, embedder_manager):
    model_cfg = cfg["model"]
    ae_cfg = model_cfg["autoencoder"]
    autoencoder = AutoEncoder.from_config(ae_cfg["config"]).to(device)
    if os.path.exists(ae_cfg["ckpt"]):
        ae_state = torch.load(ae_cfg["ckpt"], map_location=device)
        autoencoder.load_state_dict(ae_state["model"] if "model" in ae_state else ae_state)
    autoencoder.eval()

    diff_cfg = model_cfg["diffusion"]
    scheduler = load_scheduler(diff_cfg["scheduler"], diff_cfg["num_steps"])
    scheduler.to(device)

    # Detect true latent channels
    try:
        dummy_input_shape = (1, autoencoder.encoder.in_channels, 
                             autoencoder.encoder.image_size, autoencoder.encoder.image_size)
        dummy_input = torch.randn(dummy_input_shape, device=device)
        
        with torch.no_grad():
            dummy_latent = autoencoder.encode_latent(dummy_input)
        
        true_latent_channels = dummy_latent.shape[1]
        
        print(f"[Info] AutoEncoder latent channels *detected via test*: {true_latent_channels}")
        if true_latent_channels != diff_cfg["latent_channels"]:
             print(f"[Warning] Config mismatch: experiment_config.yaml says latent_channels={diff_cfg['latent_channels']}, "
                   f"but loaded AE *outputs* {true_latent_channels}. Using {true_latent_channels}.")
    
    except Exception as e:
        print(f"[Warning] Failed to detect latent channels via test: {e}. "
              f"Falling back to config value: {diff_cfg['latent_channels']}")
        true_latent_channels = diff_cfg["latent_channels"]

    unet = UNet.from_config(diff_cfg["unet_config"],
                            latent_channels=true_latent_channels,
                            latent_base=model_cfg.get("latent_base", 32)).to(device)

    pov_dim = 512
    graph_dim = 384
    hidden_dim_mlp = model_cfg.get("mixer_hidden_dim")
    mixer = select_mixer(model_cfg["mixer"],
                        out_channels=true_latent_channels,
                        latent_base=model_cfg.get("latent_base", 32),
                        pov_dim=pov_dim,
                        graph_dim=graph_dim,
                        hidden_dim_mlp=hidden_dim_mlp).to(device)

    pipeline = DiffusionPipeline(
        autoencoder=autoencoder,
        unet=unet,
        mixer=mixer,
        scheduler=scheduler,
        embedder_manager=embedder_manager,
        device=device
    )
    return pipeline


class EmbedderManager:
    """
    Manages embedding models for POV images and graph text.
    Handles both pre-loaded embeddings and on-the-fly embedding generation.
    """
    def __init__(self, pov_name: str, graph_name: str, taxonomy_path: Path, device):
        self.device = device
        
        # POV embedder (ResNet)
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()
        self.resnet.eval().to(device)
        
        # Graph text embedder (SentenceTransformer)
        self.graph_encoder = SentenceTransformer(graph_name).to(device)
        self.graph_encoder.eval()
        
        # Taxonomy for graph processing (not used with text files)
        self.taxonomy = load_taxonomy(taxonomy_path)
        
        # Text cache for graph paths
        self.cache = {}
        
        # POV preprocessing
        self.pov_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def embed_pov(self, img):
        """Embed a single POV image (PIL Image)."""
        img_tensor = self.pov_transform(img).unsqueeze(0).to(self.device)
        return self.resnet(img_tensor).squeeze(0)

    @torch.no_grad()
    def embed_graph(self, graph_data):
        """
        Embed graph data. Handles:
        - str: Path to .txt file (cached)
        - Already loaded text string
        """
        if isinstance(graph_data, str):
            # Check if it's cached text
            if graph_data in self.cache:
                text = self.cache[graph_data]
            else:
                # Load from file if it looks like a path
                if os.path.exists(graph_data) and graph_data.endswith('.txt'):
                    with open(graph_data, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    self.cache[graph_data] = text
                else:
                    # Treat as raw text
                    text = graph_data
        else:
            raise TypeError(f"embed_graph expected str, but got {type(graph_data)}")

        # Encode text to embedding
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
    
    taxonomy_path = Path(cfg["dataset"]["taxonomy_path"])
    
    embed = EmbedderManager(
        cfg["model"]["embedders"]["pov"],
        cfg["model"]["embedders"]["graph"],
        taxonomy_path,
        device
    )
    
    # Build dataloaders
    train_loader = build_dataloader(cfg["dataset"], use_embeddings=False, shuffle=True)
    val_loader = build_dataloader(cfg["dataset"], use_embeddings=False, shuffle=False)
    sample_loader = build_dataloader(cfg["dataset"], use_embeddings=False, shuffle=False)
    
    # Build pipeline
    pipeline = build_pipeline(cfg, device, embed)
    
    trainer_cfg = cfg["training"]
    trainer = PipelineTrainer(
        pipeline=pipeline,
        sample_loader=sample_loader,
        optimizer=None,
        loss_fn=None,
        epochs=trainer_cfg["epochs"],
        lr=trainer_cfg["lr"],
        weight_decay=trainer_cfg.get("weight_decay", 0.0),
        grad_clip=trainer_cfg.get("grad_clip"),
        log_interval=trainer_cfg.get("log_interval", 100),
        eval_interval=trainer_cfg.get("eval_interval", 1000),
        sample_interval=trainer_cfg.get("sample_interval", 2000),
        eval_num_samples=trainer_cfg.get("eval_sample_num", 8),
        ckpt_dir=trainer_cfg["ckpt_dir"],
        output_dir=trainer_cfg["output_dir"],
        mixed_precision=trainer_cfg.get("mixed_precision", False),
        ema_decay=trainer_cfg.get("ema_decay"),
        cond_dropout_pov=trainer_cfg.get("cond_dropout_pov", 0.0),
        cond_dropout_graph=trainer_cfg.get("cond_dropout_graph", 0.0),
        cond_dropout_both=trainer_cfg.get("cond_dropout_both", 0.0),
        taxonomy=taxonomy_path,
        use_modalities=trainer_cfg.get("use_modalities", "both")
    )

    # Run training
    trainer.fit(train_loader, val_loader=val_loader)

    # Save final weights and config
    save_dir = trainer_cfg["ckpt_dir"]
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(pipeline.autoencoder.state_dict(), 
               os.path.join(save_dir, "ae_latest.pt"))
    torch.save(pipeline.unet.state_dict(), 
               os.path.join(save_dir, "unet_latest.pt"))
    torch.save(pipeline.mixer.state_dict(), 
               os.path.join(save_dir, "mixer_latest.pt"))
    
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
        "latent_channels": pipeline.unet.in_channels,
        "latent_base": cfg["model"].get("latent_base", 32)
    }

    with open(os.path.join(save_dir, "pipeline_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(pipeline_config, f)

    print(f"[Done] Saved pipeline config and weights to {save_dir}")


if __name__ == "__main__":
    main()