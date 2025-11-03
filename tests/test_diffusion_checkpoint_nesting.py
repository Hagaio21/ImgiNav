#!/usr/bin/env python3
"""
Test that diffusion model checkpoint properly nests all components.

Tests:
1. Create decoder, UNet, scheduler with random weights
2. Save them separately
3. Create diffusion model with these components
4. Save diffusion checkpoint
5. Load separate checkpoints and diffusion checkpoint
6. Verify all weights match
"""

import torch
import tempfile
import shutil
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.diffusion import DiffusionModel
from models.decoder import Decoder
from models.components.unet import DualUNet
from models.components.scheduler import CosineScheduler


def test_checkpoint_nesting():
    """Test that all components are properly nested in diffusion checkpoint."""
    
    print("="*60)
    print("Testing Diffusion Model Checkpoint Nesting")
    print("="*60)
    
    # Create temp directory for checkpoints
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Using temp directory: {temp_dir}")
    
    try:
        # ============================================================
        # Step 1: Create components with random weights
        # ============================================================
        print("\n1. Creating components with random weights...")
        
        # Decoder
        decoder = Decoder(
            latent_channels=16,
            base_channels=64,
            upsampling_steps=4,
            norm_groups=8,
            heads=[{"type": "DecoderHead", "name": "rgb", "out_channels": 3}]
        )
        print(f"   Decoder created: {sum(p.numel() for p in decoder.parameters())} parameters")
        
        # UNet
        unet = DualUNet(
            in_channels=16,
            out_channels=16,
            base_channels=64,
            depth=4,
            num_res_blocks=2,
            time_dim=128,
            norm_groups=8
        )
        print(f"   UNet created: {sum(p.numel() for p in unet.parameters())} parameters")
        
        # Scheduler
        scheduler = CosineScheduler(num_steps=1000)
        print(f"   Scheduler created: {scheduler.alphas.shape[0]} steps")
        
        # Get initial weights (before saving)
        decoder_weights = {k: v.clone() for k, v in decoder.state_dict().items()}
        unet_weights = {k: v.clone() for k, v in unet.state_dict().items()}
        scheduler_buffers = {k: v.clone() for k, v in scheduler.state_dict().items()}
        
        # ============================================================
        # Step 2: Save components separately
        # ============================================================
        print("\n2. Saving components separately...")
        
        decoder_path = temp_dir / "decoder_checkpoint.pt"
        unet_path = temp_dir / "unet_checkpoint.pt"
        scheduler_path = temp_dir / "scheduler_checkpoint.pt"
        
        decoder.save_checkpoint(decoder_path, include_config=True)
        unet.save_checkpoint(unet_path, include_config=True)
        scheduler.save_checkpoint(scheduler_path, include_config=True)
        
        print(f"   Decoder saved to: {decoder_path}")
        print(f"   UNet saved to: {unet_path}")
        print(f"   Scheduler saved to: {scheduler_path}")
        
        # ============================================================
        # Step 3: Create diffusion model with these components
        # ============================================================
        print("\n3. Creating diffusion model with components...")
        
        # Create diffusion model using decoder config (without checkpoint path)
        # This simulates loading from a diffusion checkpoint
        diffusion_model = DiffusionModel.from_config({
            "decoder": decoder.to_config(),  # Use config without checkpoint path
            "unet": unet.to_config(),
            "scheduler": scheduler.to_config()
        })
        
        # Manually copy weights from separate components
        diffusion_model.decoder.load_state_dict(decoder.state_dict())
        diffusion_model.unet.load_state_dict(unet.state_dict())
        diffusion_model.scheduler.load_state_dict(scheduler.state_dict())
        
        print(f"   Diffusion model created")
        print(f"     - Decoder: {sum(p.numel() for p in diffusion_model.decoder.parameters())} params")
        print(f"     - UNet: {sum(p.numel() for p in diffusion_model.unet.parameters())} params")
        print(f"     - Scheduler: {len(diffusion_model.scheduler.state_dict())} buffers")
        
        # ============================================================
        # Step 4: Save diffusion checkpoint
        # ============================================================
        print("\n4. Saving diffusion checkpoint...")
        
        diffusion_path = temp_dir / "diffusion_checkpoint.pt"
        diffusion_model.save_checkpoint(diffusion_path, include_config=True)
        
        print(f"   Diffusion checkpoint saved to: {diffusion_path}")
        
        # Check state_dict keys
        diffusion_state = torch.load(diffusion_path, map_location="cpu")["state_dict"]
        print(f"\n   State dict keys in diffusion checkpoint:")
        decoder_keys = [k for k in diffusion_state.keys() if k.startswith("decoder.")]
        unet_keys = [k for k in diffusion_state.keys() if k.startswith("unet.")]
        scheduler_keys = [k for k in diffusion_state.keys() if k.startswith("scheduler.")]
        
        print(f"     - Decoder keys: {len(decoder_keys)}")
        print(f"     - UNet keys: {len(unet_keys)}")
        print(f"     - Scheduler keys: {len(scheduler_keys)}")
        
        if not decoder_keys:
            raise RuntimeError("No decoder keys found in diffusion checkpoint!")
        if not unet_keys:
            raise RuntimeError("No UNet keys found in diffusion checkpoint!")
        if not scheduler_keys:
            raise RuntimeError("No scheduler keys found in diffusion checkpoint!")
        
        # ============================================================
        # Step 5: Load checkpoints separately
        # ============================================================
        print("\n5. Loading separate checkpoints...")
        
        decoder_loaded = Decoder.load_checkpoint(decoder_path, map_location="cpu")
        unet_loaded = DualUNet.load_checkpoint(unet_path, map_location="cpu")
        scheduler_loaded = CosineScheduler.load_checkpoint(scheduler_path, map_location="cpu")
        
        print("   All separate checkpoints loaded")
        
        # ============================================================
        # Step 6: Load diffusion checkpoint
        # ============================================================
        print("\n6. Loading diffusion checkpoint...")
        
        diffusion_loaded = DiffusionModel.load_checkpoint(
            diffusion_path,
            map_location="cpu"
        )
        
        print("   Diffusion checkpoint loaded")
        
        # ============================================================
        # Step 7: Verify weights match
        # ============================================================
        print("\n7. Verifying weights match...")
        
        # Compare decoder
        print("\n   Comparing decoder weights...")
        decoder_loaded_dict = decoder_loaded.state_dict()
        diffusion_decoder_dict = diffusion_loaded.decoder.state_dict()
        
        if set(decoder_loaded_dict.keys()) != set(diffusion_decoder_dict.keys()):
            raise ValueError(f"Decoder key mismatch!\n"
                           f"  Separate: {set(decoder_loaded_dict.keys())}\n"
                           f"  Diffusion: {set(diffusion_decoder_dict.keys())}")
        
        decoder_match = True
        for key in decoder_loaded_dict.keys():
            if not torch.allclose(decoder_loaded_dict[key], diffusion_decoder_dict[key], atol=1e-6):
                print(f"     MISMATCH at decoder.{key}")
                print(f"       Separate: {decoder_loaded_dict[key].mean().item():.6f}")
                print(f"       Diffusion: {diffusion_decoder_dict[key].mean().item():.6f}")
                decoder_match = False
        
        if decoder_match:
            print("     ✓ All decoder weights match!")
        
        # Compare UNet
        print("\n   Comparing UNet weights...")
        unet_loaded_dict = unet_loaded.state_dict()
        diffusion_unet_dict = diffusion_loaded.unet.state_dict()
        
        if set(unet_loaded_dict.keys()) != set(diffusion_unet_dict.keys()):
            raise ValueError(f"UNet key mismatch!\n"
                           f"  Separate: {set(unet_loaded_dict.keys())}\n"
                           f"  Diffusion: {set(diffusion_unet_dict.keys())}")
        
        unet_match = True
        for key in unet_loaded_dict.keys():
            if not torch.allclose(unet_loaded_dict[key], diffusion_unet_dict[key], atol=1e-6):
                print(f"     MISMATCH at unet.{key}")
                print(f"       Separate: {unet_loaded_dict[key].mean().item():.6f}")
                print(f"       Diffusion: {diffusion_unet_dict[key].mean().item():.6f}")
                unet_match = False
        
        if unet_match:
            print("     ✓ All UNet weights match!")
        
        # Compare scheduler
        print("\n   Comparing scheduler buffers...")
        scheduler_loaded_dict = scheduler_loaded.state_dict()
        diffusion_scheduler_dict = diffusion_loaded.scheduler.state_dict()
        
        if set(scheduler_loaded_dict.keys()) != set(diffusion_scheduler_dict.keys()):
            raise ValueError(f"Scheduler key mismatch!\n"
                           f"  Separate: {set(scheduler_loaded_dict.keys())}\n"
                           f"  Diffusion: {set(diffusion_scheduler_dict.keys())}")
        
        scheduler_match = True
        for key in scheduler_loaded_dict.keys():
            if not torch.allclose(scheduler_loaded_dict[key], diffusion_scheduler_dict[key], atol=1e-6):
                print(f"     MISMATCH at scheduler.{key}")
                if scheduler_loaded_dict[key].numel() > 0:
                    print(f"       Separate: {scheduler_loaded_dict[key].mean().item():.6f}")
                    print(f"       Diffusion: {diffusion_scheduler_dict[key].mean().item():.6f}")
                scheduler_match = False
        
        if scheduler_match:
            print("     ✓ All scheduler buffers match!")
        
        # ============================================================
        # Step 8: Verify against original weights
        # ============================================================
        print("\n8. Verifying against original weights...")
        
        # Decoder
        decoder_original_match = True
        for key in decoder_weights.keys():
            if not torch.allclose(decoder_weights[key], diffusion_decoder_dict[key], atol=1e-6):
                decoder_original_match = False
                break
        
        # UNet
        unet_original_match = True
        for key in unet_weights.keys():
            if not torch.allclose(unet_weights[key], diffusion_unet_dict[key], atol=1e-6):
                unet_original_match = False
                break
        
        # Scheduler
        scheduler_original_match = True
        for key in scheduler_buffers.keys():
            if not torch.allclose(scheduler_buffers[key], diffusion_scheduler_dict[key], atol=1e-6):
                scheduler_original_match = False
                break
        
        # ============================================================
        # Final results
        # ============================================================
        print("\n" + "="*60)
        print("Test Results:")
        print("="*60)
        
        all_match = decoder_match and unet_match and scheduler_match
        all_original_match = decoder_original_match and unet_original_match and scheduler_original_match
        
        print(f"Decoder weights match (separate vs diffusion): {decoder_match}")
        print(f"UNet weights match (separate vs diffusion): {unet_match}")
        print(f"Scheduler buffers match (separate vs diffusion): {scheduler_match}")
        print(f"\nAll components match original weights: {all_original_match}")
        
        if all_match and all_original_match:
            print("\n✓ SUCCESS: All components properly nested in diffusion checkpoint!")
            return True
        else:
            print("\n✗ FAILURE: Some components don't match!")
            return False
        
    finally:
        # Cleanup
        print(f"\nCleaning up temp directory: {temp_dir}")
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    success = test_checkpoint_nesting()
    sys.exit(0 if success else 1)

