"""
Test script to verify training initialization works with migrated checkpoints.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from training.utils import load_config, get_device, to_device, build_dataset, split_dataset
from models.diffusion import DiffusionModel


def test_training_init(config_path: str):
    """Test that training can initialize with a given config."""
    print(f"Testing training initialization with: {config_path}")
    print("="*60)
    
    try:
        # Load config
        config = load_config(config_path)
        exp_name = config.get("experiment", {}).get("name", "unnamed")
        print(f"✓ Loaded config: {exp_name}")
        
        # Get device
        device = get_device(config)
        device_obj = to_device(device)
        print(f"✓ Device: {device}")
        
        # Build dataset (this might fail if manifest doesn't exist, but that's okay for this test)
        try:
            dataset = build_dataset(config)
            print(f"✓ Built dataset: {len(dataset)} samples")
            
            # Split dataset
            train_dataset, val_dataset = split_dataset(dataset, config["training"])
            print(f"✓ Split dataset: {len(train_dataset)} train, {len(val_dataset)} val")
        except Exception as e:
            print(f"⚠ Dataset build failed (expected if manifest missing): {e}")
            print("  Continuing with model build test...")
        
        # Build model from config
        diffusion_cfg = config.get("diffusion", {})
        if not diffusion_cfg:
            diffusion_cfg = {
                "autoencoder": config.get("autoencoder"),
                "unet": config.get("unet", {}),
                "scheduler": config.get("scheduler", {}),
                "embedding_projection": config.get("embedding_projection")
            }
        else:
            if "embedding_projection" not in diffusion_cfg and "embedding_projection" in config:
                diffusion_cfg["embedding_projection"] = config.get("embedding_projection")
        
        # Add scale_factor and latent_clamp if available
        scale_factor = config.get("scale_factor", 1.0)
        if scale_factor is not None:
            diffusion_cfg["scale_factor"] = scale_factor
        
        latent_clamp_min = config.get("latent_clamp_min")
        latent_clamp_max = config.get("latent_clamp_max")
        if latent_clamp_min is not None:
            diffusion_cfg["latent_clamp_min"] = latent_clamp_min
        if latent_clamp_max is not None:
            diffusion_cfg["latent_clamp_max"] = latent_clamp_max
        
        # Remove type if present
        if "type" in diffusion_cfg:
            diffusion_cfg = {k: v for k, v in diffusion_cfg.items() if k != "type"}
        
        # Pass save_path from experiment config
        exp_cfg = config.get("experiment", {})
        if exp_cfg.get("save_path"):
            diffusion_cfg["save_path"] = exp_cfg["save_path"]
        
        # Build model
        model = DiffusionModel(**diffusion_cfg)
        model = model.to(device_obj)
        print(f"✓ Built model successfully")
        
        # Check model components
        has_decoder = hasattr(model, 'decoder') and model.decoder is not None
        has_unet = hasattr(model, 'unet') and model.unet is not None
        has_scheduler = hasattr(model, 'scheduler') and model.scheduler is not None
        has_embedding_proj = hasattr(model, 'embedding_projection') and model.embedding_projection is not None
        
        print(f"\nModel Components:")
        print(f"  Decoder: {'✓' if has_decoder else '✗'}")
        print(f"  UNet: {'✓' if has_unet else '✗'}")
        print(f"  Scheduler: {'✓' if has_scheduler else '✗'}")
        print(f"  Embedding Projection: {'✓' if has_embedding_proj else '✗'}")
        
        if has_decoder:
            decoder_frozen = all(not p.requires_grad for p in model.decoder.parameters())
            print(f"    Decoder frozen: {decoder_frozen}")
        
        if has_unet:
            unet_trainable = sum(p.numel() for p in model.unet.parameters() if p.requires_grad)
            unet_total = sum(p.numel() for p in model.unet.parameters())
            print(f"    UNet trainable: {unet_trainable:,} / {unet_total:,} parameters")
        
        if has_embedding_proj:
            ep_trainable = sum(p.numel() for p in model.embedding_projection.parameters() if p.requires_grad)
            ep_total = sum(p.numel() for p in model.embedding_projection.parameters())
            print(f"    Embedding Projection trainable: {ep_trainable:,} / {ep_total:,} parameters")
        
        # Test forward pass with dummy data
        print(f"\nTesting forward pass...")
        batch_size = 2
        latent_shape = (4, 32, 32)  # Typical latent shape
        dummy_latents = torch.randn(batch_size, *latent_shape, device=device_obj)
        dummy_t = torch.randint(0, model.scheduler.num_steps, (batch_size,), device=device_obj)
        
        # Get dummy embeddings if needed
        text_emb = None
        pov_emb = None
        if has_embedding_proj:
            # Check if embeddings are configured
            dataset_outputs = config.get("dataset", {}).get("outputs", {})
            if "text_emb" in dataset_outputs:
                text_emb = torch.randn(batch_size, 384, device=device_obj)
            if "pov_emb" in dataset_outputs:
                pov_emb = torch.randn(batch_size, 512, device=device_obj)
        
        with torch.no_grad():
            outputs = model(dummy_latents, dummy_t, text_emb=text_emb, pov_emb=pov_emb)
            print(f"✓ Forward pass successful")
            if isinstance(outputs, dict):
                print(f"  Output keys: {list(outputs.keys())}")
        
        print(f"\n{'='*60}")
        print(f"✓ Training initialization test PASSED")
        print(f"{'='*60}")
        return True
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ Training initialization test FAILED")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test training initialization")
    parser.add_argument(
        "config",
        type=str,
        help="Path to experiment config YAML file"
    )
    
    args = parser.parse_args()
    
    success = test_training_init(args.config)
    sys.exit(0 if success else 1)

