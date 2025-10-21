import os
import sys
import tempfile
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.autoencoder import AutoEncoder
from modules.unet import UNet
from modules.condition_mixer import LinearConcatMixer
from modules.scheduler import CosineScheduler
from pipeline.pipeline import DiffusionPipeline
from training.pipeline_trainer import PipelineTrainer


def collate_fn(batch):
    """Collate function for fake dataset."""
    layouts = torch.stack([item["layout"] for item in batch])
    
    # Check if we have embeddings or raw data
    if isinstance(batch[0]["pov"], torch.Tensor):
        # Embeddings
        pov = torch.stack([item["pov"] for item in batch])
        graph = torch.stack([item["graph"] for item in batch])
    else:
        # Raw data
        pov = [item["pov"] for item in batch]
        graph = [item["graph"] for item in batch]
    
    return {
        "layout": layouts,
        "pov": pov,
        "graph": graph
    }


class FakeEmbedderManager:
    """Fake embedder that mimics EmbedderManager behavior."""
    
    def __init__(self, device):
        self.device = device
        # Simple fake models
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Identity()
        self.resnet.eval().to(device)
        
        # For graph embedding, just use a simple linear layer
        self.graph_encoder = nn.Linear(100, 384).eval().to(device)
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
            return self.cache[graph_path]
        
        # Generate a fake embedding
        fake_input = torch.randn(100).to(self.device)
        emb = self.graph_encoder(fake_input)
        self.cache[graph_path] = emb
        return emb


class FakeLayoutDataset(Dataset):
    """Fake dataset that mimics UnifiedLayoutDataset structure."""
    
    def __init__(self, num_samples=16, use_embeddings=True, embedder_manager=None):
        self.num_samples = num_samples
        self.use_embeddings = use_embeddings
        self.embedder_manager = embedder_manager
        
        # Generate fake data
        self.layouts = []
        self.pov_images = []
        self.graph_paths = []
        
        for i in range(num_samples):
            # Fake layout image (3, 256, 256)
            layout = torch.rand(3, 256, 256)
            self.layouts.append(layout)
            
            # Fake POV image
            pov_img = Image.fromarray(
                (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
            )
            self.pov_images.append(pov_img)
            
            # Fake graph path (just an identifier)
            self.graph_paths.append(f"fake_graph_{i}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        layout = self.layouts[idx]
        pov_img = self.pov_images[idx]
        graph_path = self.graph_paths[idx]
        
        if self.use_embeddings and self.embedder_manager is not None:
            # Return pre-computed embeddings
            pov_emb = self.embedder_manager.embed_pov(pov_img)
            graph_emb = self.embedder_manager.embed_graph(graph_path)
            
            return {
                "layout": layout,
                "pov": pov_emb,
                "graph": graph_emb
            }
        else:
            # Return raw data
            return {
                "layout": layout,
                "pov": pov_img,
                "graph": graph_path
            }


def create_fake_config(temp_dir):
    """Create a minimal fake config for testing."""
    config = {
        "dataset": {
            "room_manifest": "fake_room_manifest.json",
            "scene_manifest": "fake_scene_manifest.json",
            "data_mode": "room",
            "pov_type": "image",
            "batch_size": 4,
            "num_workers": 0,
            "seed": 42,
            "taxonomy_path": "C:/Users/Hagai.LAPTOP-QAG9263N/Desktop/Thesis/repositories/ImagiNav/config/taxonomy.json"
        },
        "model": {
            "autoencoder": {
                "config": {
                    "in_channels": 3,
                    "out_channels": 3,
                    "base_channels": 32,
                    "latent_channels": 4,
                    "image_size": 256,
                    "latent_base": 32,
                    "norm": "batch",
                    "act": "relu",
                    "dropout": 0.0,
                    "kernel_size": 3,
                    "stride": 2,
                    "padding": 1
                },
                "ckpt": ""  # Will train from scratch
            },
            "diffusion": {
                "scheduler": "cosine",
                "num_steps": 100,
                "latent_channels": 4,
                "unet_config": {
                    "in_channels": 4,
                    "out_channels": 4,
                    "cond_channels": 4,
                    "base_channels": 64,
                    "depth": 3,
                    "num_res_blocks": 1,
                    "time_dim": 128,
                    "norm": "batch",
                    "act": "relu"
                }
            },
            "mixer": "linear",
            "mixer_hidden_dim": None,
            "latent_base": 32,
            "embedders": {
                "pov": "resnet18",
                "graph": "sentence-transformers/all-MiniLM-L6-v2"
            }
        },
        "training": {
            "epochs": 2,
            "lr": 1e-4,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "log_interval": 2,
            "eval_interval": 4,
            "sample_interval": 8,
            "eval_sample_num": 8,
            "ckpt_dir": os.path.join(temp_dir, "checkpoints"),
            "output_dir": os.path.join(temp_dir, "output"),
            "mixed_precision": False,
            "ema_decay": None,
            "cond_dropout_pov": 0.1,
            "cond_dropout_graph": 0.1,
            "cond_dropout_both": 0.05,
            "use_modalities": "both"
        }
    }
    return config


def test_training():
    """Test the training pipeline with fake data."""
    print("[Test] Starting training pipeline test...")
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"[Test] Using temp directory: {temp_dir}")
        
        # Create fake config
        config = create_fake_config(temp_dir)
        config_path = os.path.join(temp_dir, "test_config.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        print(f"[Test] Created config at: {config_path}")
        
        # Setup device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Test] Using device: {device}")
        torch.manual_seed(42)
        
        # Create fake embedder
        embedder = FakeEmbedderManager(device)
        print("[Test] Created fake embedder manager")
        
        # Create fake datasets
        train_dataset = FakeLayoutDataset(
            num_samples=16, 
            use_embeddings=True, 
            embedder_manager=embedder
        )
        val_dataset = FakeLayoutDataset(
            num_samples=8, 
            use_embeddings=True, 
            embedder_manager=embedder
        )
        sample_dataset = FakeLayoutDataset(
            num_samples=4, 
            use_embeddings=False, 
            embedder_manager=embedder
        )
        print(f"[Test] Created fake datasets (train={len(train_dataset)}, val={len(val_dataset)}, sample={len(sample_dataset)})")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["dataset"]["batch_size"],
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["dataset"]["batch_size"],
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn
        )
        sample_loader = DataLoader(
            sample_dataset,
            batch_size=config["dataset"]["batch_size"],
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn
        )
        print("[Test] Created dataloaders")
        
        # Build autoencoder
        ae_cfg = config["model"]["autoencoder"]["config"]
        autoencoder = AutoEncoder.from_shape(**ae_cfg).to(device)
        autoencoder.eval()
        print("[Test] Created autoencoder")
        
        # Build scheduler
        diff_cfg = config["model"]["diffusion"]
        scheduler = CosineScheduler(num_steps=diff_cfg["num_steps"])
        print("[Test] Created scheduler")
        
        # Build UNet
        unet = UNet.from_config(
            diff_cfg["unet_config"],
            latent_channels=diff_cfg["latent_channels"],
            latent_base=config["model"]["latent_base"]
        ).to(device)
        print("[Test] Created UNet")
        
        # Build mixer
        mixer = LinearConcatMixer(
                out_channels=diff_cfg["latent_channels"],  # 4
                target_size=(config["model"]["latent_base"], config["model"]["latent_base"]),  # (32, 32)
                pov_channels=512,
                graph_channels=384).to(device)
        
        print("[Test] Created mixer")
        print(f"[INFO]: mixer target size ->  {mixer.target_size}")
        # Build pipeline
        pipeline = DiffusionPipeline(
            autoencoder=autoencoder,
            unet=unet,
            mixer=mixer,
            scheduler=scheduler,
            embedder_manager=embedder,
            device=device
        )
        print("[Test] Created pipeline")
        dataset_cfg = config["dataset"]
        taxonomy_path = dataset_cfg["taxonomy_path"]
        # Build trainer
        trainer_cfg = config["training"]
        trainer = PipelineTrainer(
            pipeline=pipeline,
            sample_loader=sample_loader,
            optimizer=None,
            loss_fn=None,
            epochs=trainer_cfg["epochs"],
            lr=trainer_cfg["lr"],
            weight_decay=trainer_cfg.get("weight_decay", 0.0),
            grad_clip=trainer_cfg.get("grad_clip"),
            log_interval=trainer_cfg.get("log_interval", 2),
            eval_interval=trainer_cfg.get("eval_interval", 4),
            sample_interval=trainer_cfg.get("sample_interval", 8),
            ckpt_dir=trainer_cfg["ckpt_dir"],
            output_dir=trainer_cfg["output_dir"],
            mixed_precision=trainer_cfg.get("mixed_precision", False),
            ema_decay=trainer_cfg.get("ema_decay"),
            cond_dropout_pov=trainer_cfg.get("cond_dropout_pov", 0.0),
            cond_dropout_graph=trainer_cfg.get("cond_dropout_graph", 0.0),
            cond_dropout_both=trainer_cfg.get("cond_dropout_both", 0.0),
            use_modalities=trainer_cfg.get("use_modalities", "both"),
            taxonomy=taxonomy_path,
        )
        print("[Test] Created trainer")
        
        # Run training
        print("[Test] Starting training loop...")
        trainer.fit(train_loader, val_loader=val_loader)
        print("[Test] Training completed!")
        
        # Check outputs
        print("\n[Test] Checking generated outputs...")
        plots_dir = os.path.join(trainer_cfg["output_dir"], "plots")
        if os.path.exists(plots_dir):
            print(f"[Test] ✓ Plots directory exists: {plots_dir}")
            plot_files = os.listdir(plots_dir)
            print(f"[Test] ✓ Generated plots: {plot_files}")
        else:
            print("[Test] ✗ Plots directory not found!")
        
        ckpt_dir = trainer_cfg["ckpt_dir"]
        if os.path.exists(ckpt_dir):
            print(f"[Test] ✓ Checkpoint directory exists: {ckpt_dir}")
            ckpt_files = os.listdir(ckpt_dir)
            print(f"[Test] ✓ Saved checkpoints: {ckpt_files}")
        else:
            print("[Test] ✗ Checkpoint directory not found!")
        
        samples_dir = os.path.join(trainer_cfg["output_dir"], "samples")
        if os.path.exists(samples_dir):
            print(f"[Test] ✓ Samples directory exists: {samples_dir}")
        else:
            print("[Test] ✗ Samples directory not found!")
        
        print("\n[Test] ✓ All tests passed!")
        print(f"[Test] Note: Temporary files will be cleaned up automatically")


if __name__ == "__main__":
    test_training()