#!/usr/bin/env python3
"""
Test script to debug CLIP loss gradient flow.
Creates a minimal model with fake embeddings and checks if gradients flow correctly.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.autoencoder import Autoencoder
from models.losses.clip_loss import CLIPLoss, CLIPProjections
from models.losses.base_loss import CompositeLoss

def test_clip_loss_gradients():
    """Test if CLIP loss gradients flow correctly."""
    print("=" * 80)
    print("Testing CLIP Loss Gradient Flow")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create a minimal VAE model with CLIP projections
    print("\n1. Creating model with CLIP projections...")
    encoder_cfg = {
        "type": "Encoder",
        "in_channels": 3,
        "base_channels": 64,
        "downsampling_steps": 4,  # Standard: 256 -> 128 -> 64 -> 32 -> 16
        "latent_channels": 4,
        "variational": True,  # VAE mode to get latent_features
    }
    
    decoder_cfg = {
        "type": "Decoder",
        "base_channels": 64,
        "upsampling_steps": 4,  # Match encoder downsampling
        "latent_channels": 4,
        "activation": "SiLU",
        "norm_groups": 8,
        "heads": [
            {
                "type": "RGBHead",
                "name": "rgb",
                "out_channels": 3,
                "final_activation": "tanh",
            }
        ],
    }
    
    clip_projection_cfg = {
        "projection_dim": 256,
        "text_dim": 384,
        "pov_dim": 512,
        "latent_dim": None,  # Will be inferred
    }
    
    model = Autoencoder(
        encoder=encoder_cfg,
        decoder=decoder_cfg,
        clip_projection=clip_projection_cfg
    ).to(device)
    
    model.train()
    print(f"  Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Has clip_projections: {hasattr(model, 'clip_projections') and model.clip_projections is not None}")
    
    if hasattr(model, 'clip_projections') and model.clip_projections is not None:
        proj_params = sum(p.numel() for p in model.clip_projections.parameters())
        proj_trainable = sum(p.numel() for p in model.clip_projections.parameters() if p.requires_grad)
        print(f"  CLIP projections: {proj_trainable}/{proj_params} trainable")
    
    # Create CLIP loss
    print("\n2. Creating CLIP loss...")
    clip_loss_cfg = {
        "type": "CLIPLoss",
        "key": "latent_features",
        "text_key": "text_emb",
        "pov_key": "pov_emb",
        "temperature": 0.07,
        "weight": 0.1,
        "projection_dim": 256,
        "text_dim": 384,
        "pov_dim": 512,
        "combine_method": "average",
        "use_model_projections": True,  # Use model's projections
    }
    
    clip_loss = CLIPLoss.from_config(clip_loss_cfg)
    
    # Connect model's projections to CLIP loss
    if hasattr(model, 'clip_projections') and model.clip_projections is not None:
        clip_loss.set_projections(model.clip_projections)
        print(f"  Connected model.clip_projections to CLIP loss")
        print(f"  Same instance: {clip_loss.projections is model.clip_projections}")
    
    # Create composite loss with both reconstruction (MSE) and CLIP loss
    # This matches real training where decoder gets gradients from reconstruction loss
    mse_loss_cfg = {
        "type": "MSELoss",
        "key": "rgb",
        "target_key": "rgb",
        "weight": 1.0,
    }
    
    composite_loss = CompositeLoss.from_config({
        "type": "CompositeLoss",
        "losses": [mse_loss_cfg, clip_loss_cfg]  # Both reconstruction and CLIP
    })
    
    # Connect projections to composite loss's CLIP loss
    if hasattr(model, 'clip_projections') and model.clip_projections is not None:
        for sub_loss in composite_loss.losses:
            if isinstance(sub_loss, CLIPLoss):
                sub_loss.set_projections(model.clip_projections)
                print(f"  Connected projections to composite loss's CLIP loss")
    
    print(f"  Composite loss has {len(composite_loss.losses)} components: {[type(l).__name__ for l in composite_loss.losses]}")
    
    # Create fake data with correct dimensions
    print("\n3. Creating fake data...")
    batch_size = 4
    img_size = 256  # Standard size
    
    fake_images = torch.randn(batch_size, 3, img_size, img_size, device=device)
    
    # Get actual dimensions from model after forward pass to ensure compatibility
    with torch.no_grad():
        test_outputs = model(fake_images[:1])  # Single sample to get dimensions
        if "latent_features" in test_outputs:
            latent_features_shape = test_outputs["latent_features"].shape
            # latent_features is [B, C, H, W], we need to flatten spatial dims for projection
            if len(latent_features_shape) == 4:
                latent_dim = latent_features_shape[1] * latent_features_shape[2] * latent_features_shape[3]
            else:
                latent_dim = latent_features_shape[1]
        else:
            latent_dim = 128 * 16 * 16  # Default fallback
    
    # Use correct embedding dimensions
    text_dim = 384  # SentenceTransformer dimension
    pov_dim = 512   # ResNet18 dimension
    
    fake_text_emb = torch.randn(batch_size, text_dim, device=device)  # Detached (pre-computed)
    fake_pov_emb = torch.randn(batch_size, pov_dim, device=device)   # Detached (pre-computed)
    
    print(f"  Images: {fake_images.shape}")
    print(f"  Text embeddings: {fake_text_emb.shape} (requires_grad={fake_text_emb.requires_grad})")
    print(f"  POV embeddings: {fake_pov_emb.shape} (requires_grad={fake_pov_emb.requires_grad})")
    if "latent_features" in test_outputs:
        print(f"  Latent features shape: {test_outputs['latent_features'].shape}")
        print(f"  Latent dim (flattened): {latent_dim}")
    
    # Forward pass
    print("\n4. Running forward pass...")
    outputs = model(fake_images)
    print(f"  Model outputs keys: {list(outputs.keys())}")
    
    # Check if latent_features exists
    if "latent_features" in outputs:
        latent_features = outputs["latent_features"]
        print(f"  latent_features: {latent_features.shape}")
        print(f"  latent_features.requires_grad: {latent_features.requires_grad}")
    else:
        print("  ERROR: latent_features not found in outputs!")
        print(f"  Available keys: {list(outputs.keys())}")
        return False
    
    # Prepare preds and targets (include RGB for reconstruction loss)
    preds = {
        "latent_features": outputs["latent_features"],
        "rgb": outputs.get("rgb", None)  # Decoder output for reconstruction loss
    }
    targets = {
        "rgb": fake_images,  # Ground truth for reconstruction loss
        "text_emb": fake_text_emb,
        "pov_emb": fake_pov_emb,
    }
    
    # Check if RGB output exists (decoder output)
    if preds["rgb"] is None:
        print("  ERROR: No RGB output from decoder! Decoder won't get gradients from reconstruction loss.")
        print(f"  Available output keys: {list(outputs.keys())}")
        return False
    else:
        print(f"  RGB output: {preds['rgb'].shape} (for reconstruction loss)")
        print(f"  RGB output range: [{preds['rgb'].min().item():.3f}, {preds['rgb'].max().item():.3f}]")
    
    # Compute loss
    print("\n5. Computing CLIP loss...")
    loss, logs = composite_loss(preds, targets)
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Loss.requires_grad: {loss.requires_grad}")
    print(f"  Loss logs: {list(logs.keys())}")
    
    if not loss.requires_grad:
        print("  ERROR: Loss does not require gradients!")
        return False
    
    # Create optimizer
    print("\n6. Creating optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Count parameters in optimizer
    opt_params = sum(len(group['params']) for group in optimizer.param_groups)
    print(f"  Optimizer parameter groups: {len(optimizer.param_groups)}")
    print(f"  Total parameters in optimizer: {opt_params}")
    
    # Check if clip_projections are in optimizer
    if hasattr(model, 'clip_projections') and model.clip_projections is not None:
        # Get all projection parameters (use id() for comparison to avoid tensor ops)
        proj_param_ids = {id(p) for p in model.clip_projections.parameters()}
        # Check if any projection param is in any optimizer group
        proj_in_opt = False
        for group in optimizer.param_groups:
            group_param_ids = {id(p) for p in group['params']}
            if proj_param_ids & group_param_ids:  # Intersection
                proj_in_opt = True
                break
        print(f"  CLIP projections in optimizer: {proj_in_opt}")
        if not proj_in_opt:
            print("  WARNING: CLIP projections not found in optimizer!")
            print(f"    Projection params count: {len(proj_param_ids)}")
            print(f"    Optimizer groups: {len(optimizer.param_groups)}")
            for i, group in enumerate(optimizer.param_groups):
                print(f"      Group {i}: {len(group['params'])} params")
    
    # Backward pass
    print("\n7. Running backward pass...")
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    print("\n8. Checking gradients...")
    has_grads = False
    grad_info = []
    
    # Check encoder gradients
    encoder_grads = sum(1 for p in model.encoder.parameters() if p.requires_grad and p.grad is not None)
    encoder_total = sum(1 for p in model.encoder.parameters() if p.requires_grad)
    grad_info.append(f"Encoder: {encoder_grads}/{encoder_total} parameters have gradients")
    if encoder_grads > 0:
        has_grads = True
    
    # Check decoder gradients
    decoder_grads = sum(1 for p in model.decoder.parameters() if p.requires_grad and p.grad is not None)
    decoder_total = sum(1 for p in model.decoder.parameters() if p.requires_grad)
    grad_info.append(f"Decoder: {decoder_grads}/{decoder_total} parameters have gradients")
    if decoder_grads > 0:
        has_grads = True
    
    # Check CLIP projection gradients
    if hasattr(model, 'clip_projections') and model.clip_projections is not None:
        proj_grads = sum(1 for p in model.clip_projections.parameters() if p.requires_grad and p.grad is not None)
        proj_total = sum(1 for p in model.clip_projections.parameters() if p.requires_grad)
        grad_info.append(f"CLIP projections: {proj_grads}/{proj_total} parameters have gradients")
        if proj_grads > 0:
            has_grads = True
    
    for info in grad_info:
        print(f"  {info}")
    
    # Check specific gradient values
    print("\n9. Sample gradient values:")
    sample_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  {name}: grad_norm={grad_norm:.6f}")
            sample_count += 1
            if sample_count >= 5:  # Show first 5
                break
    
    if not has_grads:
        print("\n  ERROR: No gradients found after backward pass!")
        return False
    
    # Test optimizer step
    print("\n10. Testing optimizer step...")
    try:
        optimizer.step()
        print("  Optimizer step successful!")
    except Exception as e:
        print(f"  ERROR in optimizer step: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("TEST PASSED: Gradients are flowing correctly!")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_clip_loss_gradients()
    sys.exit(0 if success else 1)

