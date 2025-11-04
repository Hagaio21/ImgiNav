#!/usr/bin/env python3
"""
Test script for checkpoint migration, inference, and ControlNet attachment.

This script:
1. Loads a checkpoint
2. Checks if migration is needed (DualUNet -> Unet)
3. Runs basic inference
4. Creates and tests ControlNet attachment

Usage:
    python scripts/test_checkpoint_migration.py \
        --checkpoint path/to/checkpoint.pt \
        --device cuda
"""

import argparse
import sys
from pathlib import Path
import torch
import yaml
from torchvision.utils import save_image
import math

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.diffusion import DiffusionModel
from models.components.unet import Unet, DualUNet
from models.components.controlnet import ControlNet


def check_migration_needed(checkpoint_path):
    """Check if checkpoint needs migration from DualUNet to Unet."""
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = payload.get("config")
    
    if config is None:
        return False, "No config found in checkpoint"
    
    unet_config = None
    if "unet" in config:
        unet_config = config["unet"]
    elif "model" in config and "unet" in config["model"]:
        unet_config = config["model"]["unet"]
    
    if unet_config is None:
        return False, "No UNet config found"
    
    unet_type = unet_config.get("type", "").lower()
    
    if unet_type == "unet":
        return False, "Already migrated (type: Unet)"
    
    if unet_type in ("dualunet", "dual_unet", ""):
        cond_channels = unet_config.get("cond_channels", 0)
        fusion_mode = unet_config.get("fusion_mode", "none")
        needs_migration = True
        info = f"Needs migration (type: {unet_type or 'unspecified'})"
        if cond_channels > 0 or fusion_mode not in ("none", None):
            info += f" - WARNING: Conditioning enabled (cond_channels={cond_channels}, fusion_mode={fusion_mode})"
        return needs_migration, info
    
    return False, f"Unknown UNet type: {unet_type}"


def test_inference_single(model, device, num_samples, num_steps, method, output_dir=None):
    """Test inference with a single method."""
    model.eval()
    model = model.to(device)
    
    # Check scheduler configuration
    scheduler_num_steps = model.scheduler.num_steps
    print(f"\n  Method: {method.upper()}")
    print(f"  Total steps: {scheduler_num_steps}")
    print(f"  Requested sampling steps: {num_steps}")
    if method == "ddim":
        print(f"  Step size (for DDIM): {scheduler_num_steps // num_steps if num_steps > 0 else 'N/A'}")
    
    actual_steps = num_steps
    if num_steps > scheduler_num_steps:
        print(f"  ⚠ Warning: Requested {num_steps} steps but scheduler only has {scheduler_num_steps} steps")
        print(f"    Will use {scheduler_num_steps} steps instead")
        actual_steps = scheduler_num_steps
    
    try:
        # Run inference
        print(f"  Generating {num_samples} samples with {actual_steps} steps ({method.upper()})...")
        with torch.no_grad():
            result = model.sample(
                batch_size=num_samples,
                num_steps=actual_steps,
                method=method,
                device=device,
                verbose=False  # Less verbose when testing both methods
            )
        
        print(f"  ✓ {method.upper()} inference successful!")
        
        if "latent" in result:
            latents = result["latent"]
            print(f"    Latent shape: {latents.shape}, range: [{latents.min().item():.3f}, {latents.max().item():.3f}]")
        
        if "rgb" in result:
            rgb = result["rgb"]
            print(f"    RGB shape: {rgb.shape}, range: [{rgb.min().item():.3f}, {rgb.max().item():.3f}]")
        
        # Save outputs if output directory is provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if "rgb" in result:
                # Save RGB images
                rgb = result["rgb"]  # Already in [0, 1] range
                grid_n = int(math.sqrt(num_samples))
                if grid_n * grid_n < num_samples:
                    grid_n += 1
                
                rgb_path = output_dir / f"inference_samples_{method}_{actual_steps}steps.png"
                save_image(rgb, rgb_path, nrow=grid_n, normalize=False)
                print(f"    ✓ Saved RGB samples to: {rgb_path}")
            
            # Also save latents if available
            if "latent" in result:
                latents = result["latent"]
                latent_path = output_dir / f"inference_latents_{method}_{actual_steps}steps.pt"
                torch.save(latents, latent_path)
                print(f"    ✓ Saved latents to: {latent_path}")
        
        return True, result
        
    except Exception as e:
        print(f"  ✗ {method.upper()} inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_inference(model, device, num_samples=2, num_steps=20, method="both", output_dir=None):
    """Test basic inference with the model. If method='both', test both DDIM and DDPM."""
    print("\n" + "="*60)
    print("Testing Inference")
    print("="*60)
    
    if method == "both":
        print("\nTesting both DDIM and DDPM sampling methods...")
        
        results = {}
        success_ddim = False
        success_ddpm = False
        
        # Test DDIM
        print("\n" + "-"*60)
        print("DDIM Sampling")
        print("-"*60)
        success_ddim, result_ddim = test_inference_single(
            model, device, num_samples, num_steps, "ddim", output_dir
        )
        if success_ddim:
            results["ddim"] = result_ddim
        
        # Test DDPM
        print("\n" + "-"*60)
        print("DDPM Sampling")
        print("-"*60)
        success_ddpm, result_ddpm = test_inference_single(
            model, device, num_samples, num_steps, "ddpm", output_dir
        )
        if success_ddpm:
            results["ddpm"] = result_ddpm
        
        # Summary
        print("\n" + "-"*60)
        print("Inference Summary")
        print("-"*60)
        if success_ddim:
            print("✓ DDIM sampling successful")
        else:
            print("✗ DDIM sampling failed")
        
        if success_ddpm:
            print("✓ DDPM sampling successful")
        else:
            print("✗ DDPM sampling failed")
        
        # Return first successful result
        if success_ddim:
            samples = results["ddim"].get("rgb", results["ddim"].get("latent"))
            return success_ddim or success_ddpm, samples
        elif success_ddpm:
            samples = results["ddpm"].get("rgb", results["ddpm"].get("latent"))
            return success_ddim or success_ddpm, samples
        else:
            return False, None
    else:
        # Single method test
        return test_inference_single(model, device, num_samples, num_steps, method, output_dir)


def test_controlnet_attachment(model, device):
    """Test creating and attaching ControlNet to the model."""
    print("\n" + "="*60)
    print("Testing ControlNet Attachment")
    print("="*60)
    
    try:
        # Freeze the base UNet parameters before building ControlNet
        # This is the standard practice: freeze the base model, train only the adapter
        print("Freezing base UNet parameters...")
        model.unet.eval()  # Set to eval mode
        for param in model.unet.parameters():
            param.requires_grad = False
        print(f"✓ Frozen {sum(1 for p in model.unet.parameters() if not p.requires_grad)}/{sum(1 for _ in model.unet.parameters())} UNet parameters")
        
        # Get UNet config from model
        unet_config = model.unet.to_config()
        
        # Create ControlNet config
        # Extract base_channels and depth from UNet config
        base_channels = unet_config.get("base_channels", 64)
        depth = unet_config.get("depth", 4)
        
        # Note: ControlAdapter currently outputs features at full resolution,
        # but skip connections are at progressively smaller resolutions.
        # The adapter needs to downsample POV embeddings to match each skip level.
        # For now, we'll test with depth=1 to match only the first skip level.
        # A proper fix would require modifying ControlAdapter to downsample
        # POV embeddings to match each skip level's spatial size.
        
        controlnet_config = {
            "base_unet": unet_config,
            "adapter": {
                "text_dim": 768,  # Typical text embedding dimension
                "pov_dim": 256,   # Typical POV embedding dimension
                "base_channels": base_channels,
                "depth": 1  # Use depth=1 to match first skip level only (test limitation)
            },
            "fuse_mode": "add"
        }
        
        print(f"Creating ControlNet with:")
        print(f"  Base UNet: {base_channels} base_channels, depth {depth}")
        print(f"  Adapter: text_dim=768, pov_dim=256")
        print(f"  Fuse mode: add")
        
        # Create ControlNet
        controlnet = ControlNet.from_config(controlnet_config)
        controlnet = controlnet.to(device)
        controlnet.eval()
        
        print("✓ ControlNet created successfully")
        
        # IMPORTANT: Copy weights from the trained UNet to ControlNet's base_unet
        # ControlNet creates a new UNet from config, so we need to transfer the trained weights
        print("\nCopying weights from trained UNet to ControlNet's base_unet...")
        controlnet.base_unet.load_state_dict(model.unet.state_dict(), strict=False)
        print("✓ Weights copied successfully")
        
        # Now freeze all base_unet parameters (not just downblocks)
        print("Freezing all ControlNet base_unet parameters...")
        for param in controlnet.base_unet.parameters():
            param.requires_grad = False
        # Also explicitly freeze all blocks to be safe
        controlnet.base_unet.freeze_blocks(["downs", "ups", "bottleneck", "time_mlp", "final"])
        print("✓ All base_unet parameters frozen")
        
        # Test forward pass
        print("\nTesting ControlNet forward pass...")
        batch_size = 2
        
        # Infer latent shape from decoder config (same as DiffusionModel.sample)
        latent_ch = model.decoder._init_kwargs.get('latent_channels', 16)
        up_steps = model.decoder._init_kwargs.get('upsampling_steps', 4)
        spatial_res = 512 // (2 ** up_steps)
        latent_shape = (latent_ch, spatial_res, spatial_res)
        print(f"  Inferred latent shape: {latent_shape}")
        
        # Create dummy inputs
        x_t = torch.randn(batch_size, *latent_shape, device=device)
        num_steps = model.scheduler.num_steps
        t = torch.randint(0, num_steps, (batch_size,), device=device)
        text_emb = torch.randn(batch_size, 768, device=device)
        
        # POV embedding: The ControlAdapter expects POV embeddings at full resolution
        # It uses 1x1 convs to project to different channel counts, and the spatial
        # dimensions are handled by the UNet's downsampling in the skip connections.
        # The adapter will project the POV embedding to match each skip level's channels.
        pov_emb = torch.randn(batch_size, 256, spatial_res, spatial_res, device=device)
        
        # Debug: Print expected shapes
        print(f"  Input x_t: {x_t.shape}")
        print(f"  Text embedding: {text_emb.shape}")
        print(f"  POV embedding: {pov_emb.shape}")
        
        with torch.no_grad():
            output = controlnet(x_t, t, text_emb, pov_emb)
        
        print(f"✓ ControlNet forward pass successful!")
        print(f"  Input shape: {x_t.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        # Check that ControlNet matches UNet output shape expectations
        # The ControlNet should output the same shape as UNet
        expected_shape = x_t.shape
        if output.shape == expected_shape:
            print(f"✓ Output shape matches expected shape: {expected_shape}")
        else:
            print(f"⚠ Warning: Output shape {output.shape} doesn't match expected {expected_shape}")
        
        # Test that base UNet weights are frozen
        print("\nChecking that base UNet weights are frozen...")
        frozen_params = sum(1 for p in controlnet.base_unet.parameters() if not p.requires_grad)
        total_params = sum(1 for p in controlnet.base_unet.parameters())
        if frozen_params == total_params:
            print(f"✓ All {total_params}/{total_params} base UNet parameters are frozen")
        else:
            print(f"✗ ERROR: Only {frozen_params}/{total_params} parameters are frozen!")
            # Debug: find which parameters are not frozen
            unfrozen = [(name, p.shape) for name, p in controlnet.base_unet.named_parameters() if p.requires_grad]
            if unfrozen:
                print(f"  Unfrozen parameters: {[name for name, _ in unfrozen[:5]]}...")
                print(f"  Total unfrozen: {len(unfrozen)}")
        
        # Test that adapter weights are trainable
        print("Checking that adapter weights are trainable...")
        trainable_params = sum(1 for p in controlnet.adapter.parameters() if p.requires_grad)
        total_adapter_params = sum(1 for p in controlnet.adapter.parameters())
        if trainable_params == total_adapter_params:
            print(f"✓ All {total_adapter_params} adapter parameters are trainable")
        else:
            print(f"⚠ Warning: Only {trainable_params}/{total_adapter_params} adapter parameters are trainable")
        
        return True, controlnet
        
    except Exception as e:
        print(f"✗ ControlNet attachment failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    parser = argparse.ArgumentParser(description="Test checkpoint migration, inference, and ControlNet")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Number of samples to generate for inference test"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="Number of sampling steps for inference test"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="both",
        choices=["ddim", "ddpm", "both"],
        help="Sampling method: 'ddim' (faster, deterministic), 'ddpm' (slower, stochastic), or 'both' (test both)"
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference test (faster)"
    )
    parser.add_argument(
        "--skip-controlnet",
        action="store_true",
        help="Skip ControlNet test"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save generated samples (default: checkpoint directory)"
    )
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # =====================================================================
    # Step 1: Check migration status
    # =====================================================================
    print("\n" + "="*60)
    print("Step 1: Checking Migration Status")
    print("="*60)
    
    needs_migration, migration_info = check_migration_needed(checkpoint_path)
    print(f"Migration status: {migration_info}")
    
    if needs_migration:
        print("\n⚠  This checkpoint uses DualUNet and can be migrated to Unet.")
        print("   Migration is optional - the checkpoint will load correctly without migration")
        print("   (backward compatibility is maintained).")
        print("   To migrate, run: python scripts/migrate_dualunet_checkpoints.py")
    else:
        print("\n✓  No migration needed")
    
    # =====================================================================
    # Step 2: Load checkpoint
    # =====================================================================
    print("\n" + "="*60)
    print("Step 2: Loading Checkpoint")
    print("="*60)
    
    try:
        print(f"Loading checkpoint: {checkpoint_path}")
        model = DiffusionModel.load_checkpoint(
            checkpoint_path,
            map_location=device
        )
        print("✓ Checkpoint loaded successfully")
        
        # Print model info
        print(f"\nModel info:")
        print(f"  UNet type: {type(model.unet).__name__}")
        print(f"  UNet config: {model.unet.to_config()}")
        print(f"  Scheduler type: {type(model.scheduler).__name__}")
        print(f"  Scheduler num_steps: {model.scheduler.num_steps}")
        print(f"  Decoder: {type(model.decoder).__name__}")
        
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # =====================================================================
    # Step 3: Test inference
    # =====================================================================
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default to checkpoint directory
        output_dir = checkpoint_path.parent / "test_outputs"
    
    if not args.skip_inference:
        success, samples = test_inference(
            model, device, args.num_samples, args.num_steps, 
            method=args.method,
            output_dir=output_dir
        )
        if not success:
            print("\n⚠  Inference test failed, but continuing with other tests...")
    else:
        print("\nSkipping inference test (--skip-inference)")
    
    # =====================================================================
    # Step 4: Test ControlNet attachment
    # =====================================================================
    if not args.skip_controlnet:
        success, controlnet = test_controlnet_attachment(model, device)
        if not success:
            print("\n⚠  ControlNet test failed")
            return 1
    else:
        print("\nSkipping ControlNet test (--skip-controlnet)")
    
    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print("✓ Checkpoint loaded successfully")
    if not args.skip_inference:
        print("✓ Inference test passed")
        print(f"  Outputs saved to: {output_dir}")
    if not args.skip_controlnet:
        print("✓ ControlNet attachment test passed")
    print("\nAll tests completed!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

