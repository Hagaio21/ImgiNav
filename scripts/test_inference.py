#!/usr/bin/env python3
"""
Test inference with migrated standalone checkpoint and text embeddings.
"""

import sys
from pathlib import Path
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.diffusion import DiffusionModel


def test_inference():
    """Test inference with migrated checkpoint and text embedding."""
    
    # Paths
    experiment_config = "experiments/diffusion/clip/regular_rooms/small_down_bottleneck_text_only.yaml"
    checkpoint_path = "checkpoints/diff_clip_regular_rooms_small_down_bottleneck_text_only_checkpoint_best.pt"
    text_emb_path = "checkpoints/text_embedding.pt"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load experiment config
    print(f"\nLoading experiment config: {experiment_config}")
    with open(experiment_config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model from checkpoint (standalone checkpoint contains all config)
    print(f"\nLoading model from checkpoint: {checkpoint_path}")
    model = DiffusionModel.load_checkpoint(checkpoint_path, map_location=device, config=config)
    model = model.to(device)
    model.eval()
    print("✓ Model loaded from checkpoint")
    
    # Load text embedding
    print(f"\nLoading text embedding: {text_emb_path}")
    text_emb = torch.load(text_emb_path, map_location=device)
    if text_emb.dim() > 2:
        text_emb = text_emb.flatten(start_dim=1)
    print(f"✓ Text embedding loaded: {text_emb.shape}")
    
    # Test forward pass (not full sampling, just check it works)
    print("\nTesting forward pass...")
    batch_size = 1
    
    # Get latent shape from decoder config
    latent_ch = model.decoder._init_kwargs.get('latent_channels', 4)
    up_steps = model.decoder._init_kwargs.get('upsampling_steps', 4)
    spatial_res = 512 // (2 ** up_steps)
    latent_shape = (latent_ch, spatial_res, spatial_res)
    print(f"  Latent shape: {latent_shape}")
    
    # Create dummy latents and timestep
    latents = torch.randn(batch_size, *latent_shape, device=device)
    t = torch.randint(0, model.scheduler.num_steps, (batch_size,), device=device)
    
    with torch.no_grad():
        # Expand text_emb to batch size
        if text_emb.shape[0] == 1 and batch_size > 1:
            text_emb_batch = text_emb.repeat(batch_size, 1)
        else:
            text_emb_batch = text_emb
        
        output = model.forward(latents, t, text_emb=text_emb_batch, pov_emb=None)
        if isinstance(output, dict):
            print(f"✓ Forward pass successful")
            if "pred_noise" in output:
                print(f"  pred_noise shape: {output['pred_noise'].shape}")
        else:
            print(f"✓ Forward pass successful: output shape = {output.shape if hasattr(output, 'shape') else 'dict'}")
    
    # Test sampling (small number of steps for quick test)
    print("\nTesting sampling (5 steps for quick test)...")
    with torch.no_grad():
        # Expand text_emb to batch size
        if text_emb.shape[0] == 1:
            text_emb_sample = text_emb.repeat(batch_size, 1)
        else:
            text_emb_sample = text_emb
        
        samples = model.sample(
            batch_size=batch_size,
            num_steps=5,  # Small for quick test
            method="ddim",
            eta=0.0,
            device=device,
            text_emb=text_emb_sample,
            pov_emb=None,
            guidance_scale=1.0,
            verbose=True
        )
        
        print(f"✓ Sampling successful")
        if isinstance(samples, dict):
            if "rgb" in samples:
                print(f"  RGB shape: {samples['rgb'].shape}")
            if "latent" in samples:
                print(f"  Latent shape: {samples['latent'].shape}")
        else:
            print(f"  Sample shape: {samples.shape}")
    
    print("\n" + "="*60)
    print("Inference test complete! ✓")
    print("="*60)
    
    return True


if __name__ == "__main__":
    try:
        test_inference()
    except Exception as e:
        print(f"\n❌ Error during inference test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

