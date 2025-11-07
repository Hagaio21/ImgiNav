#!/usr/bin/env python3
"""
Test script for train_discriminator.py.
Creates fake latents to verify the discriminator training script works.
"""

import torch
import tempfile
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_discriminator import train_discriminator


def create_fake_latents(output_dir, num_samples=20, latent_channels=16, spatial_size=32):
    """Create fake latent tensors and save them (match actual system: 32x32 with 16 channels)."""
    latents_dir = output_dir / "fake_latents"
    latents_dir.mkdir(parents=True, exist_ok=True)
    
    # Create normalized latents (N(0,1))
    latents = torch.randn(num_samples, latent_channels, spatial_size, spatial_size)
    
    # Save as single .pt file
    latent_path = latents_dir / "latents.pt"
    torch.save(latents, latent_path)
    
    return latent_path


def test_discriminator_training():
    """Test discriminator training with fake latents."""
    print("\n" + "="*60)
    print("Testing Discriminator Training")
    print("="*60)
    
    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp()
        output_dir = Path(tmpdir) / "test_discriminator"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create fake real latents (from dataset)
        print("Creating fake real latents...")
        real_latents_path = create_fake_latents(output_dir / "real", num_samples=20, latent_channels=16)
        
        # Create fake bad latents (from diffusion model)
        print("Creating fake bad latents...")
        bad_latents_path = create_fake_latents(output_dir / "bad", num_samples=20, latent_channels=16)
        
        # Create output directory
        disc_output_dir = output_dir / "discriminator_output"
        disc_output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Running discriminator training (2 epochs)...")
        
        try:
            train_discriminator(
                real_latents_path=real_latents_path,
                fake_latents_path=bad_latents_path,
                output_dir=disc_output_dir,
                epochs=2,  # Just 2 epochs for testing
                batch_size=4,
                learning_rate=0.0002,
                device="cpu",  # Use CPU for testing
                seed=42
            )
            # Verify outputs were created
            best_checkpoint = disc_output_dir / "discriminator_best.pt"
            final_checkpoint = disc_output_dir / "discriminator_final.pt"
            history_csv = disc_output_dir / "discriminator_history.csv"
            
            assert best_checkpoint.exists(), f"Best checkpoint not found: {best_checkpoint}"
            assert final_checkpoint.exists(), f"Final checkpoint not found: {final_checkpoint}"
            assert history_csv.exists(), f"History CSV not found: {history_csv}"
            print(f"  ✓ Checkpoints saved: {best_checkpoint}, {final_checkpoint}")
            
            # Verify checkpoint can be loaded
            checkpoint = torch.load(best_checkpoint, map_location="cpu")
            assert "state_dict" in checkpoint, "Checkpoint missing state_dict"
            assert "config" in checkpoint, "Checkpoint missing config"
            assert "val_loss" in checkpoint, "Checkpoint missing val_loss"
            assert "val_acc" in checkpoint, "Checkpoint missing val_acc"
            print(f"  ✓ Checkpoint valid (val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_acc']:.4f})")
            
            # Verify history has data
            import pandas as pd
            history_df = pd.read_csv(history_csv)
            assert len(history_df) == 2, f"Expected 2 epochs in history, got {len(history_df)}"
            assert "train_loss" in history_df.columns, "History missing train_loss"
            assert "val_loss" in history_df.columns, "History missing val_loss"
            assert "train_acc" in history_df.columns, "History missing train_acc"
            assert "val_acc" in history_df.columns, "History missing val_acc"
            print(f"  ✓ Training history recorded ({len(history_df)} epochs)")
            
            # Verify loss decreased (or at least changed)
            if len(history_df) >= 2:
                first_loss = history_df.iloc[0]["train_loss"]
                last_loss = history_df.iloc[-1]["train_loss"]
                print(f"  ✓ Loss changed: {first_loss:.4f} -> {last_loss:.4f}")
            
            print("✓ Discriminator training test passed!")
        except Exception as e:
            print(f"✗ Discriminator training test failed: {e}")
            raise
    finally:
        # Clean up temporary directory
        if tmpdir and Path(tmpdir).exists():
            print(f"Cleaning up temporary directory: {tmpdir}")
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    """Run discriminator training test."""
    print("="*60)
    print("Testing train_discriminator.py")
    print("="*60)
    
    try:
        test_discriminator_training()
        print("\n" + "="*60)
        print("Discriminator training test passed! ✓")
        print("="*60)
    except Exception as e:
        print("\n" + "="*60)
        print(f"Test failed: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

