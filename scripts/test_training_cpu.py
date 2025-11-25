#!/usr/bin/env python3
"""
Training test script for CPU verification.

Tests all training components (autoencoder and diffusion) on CPU with synthetic data
to verify everything works after refactoring. No actual dataset files required.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import tempfile
import shutil
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.utils import (
    build_model,
    build_loss,
    build_optimizer,
    build_scheduler,
    get_device,
    to_device,
    set_deterministic,
)
from training.engine import Trainer
from training.train import ae_step_fn, ae_eval_step_fn
from training.train_diffusion import diffusion_step_fn, diffusion_eval_step_fn
from models.autoencoder import Autoencoder
from models.diffusion import DiffusionModel
from models.losses.base_loss import LOSS_REGISTRY


class SyntheticDataset(torch.utils.data.Dataset):
    """Synthetic dataset that generates dummy tensors for testing."""
    
    def __init__(self, num_samples=4, image_size=256, latent_shape=None, has_text_emb=False, has_pov_emb=False):
        """
        Args:
            num_samples: Number of samples in dataset
            image_size: Size of RGB images (assumes square)
            latent_shape: Optional tuple (C, H, W) for latent tensors. If None, not provided.
            has_text_emb: Whether to include text embeddings
            has_pov_emb: Whether to include POV embeddings
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.latent_shape = latent_shape
        self.has_text_emb = has_text_emb
        self.has_pov_emb = has_pov_emb
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Generate a synthetic batch item."""
        batch = {}
        
        # RGB images in [-1, 1] range (normalized)
        batch["rgb"] = torch.randn(3, self.image_size, self.image_size) * 0.5  # Roughly [-1, 1]
        batch["rgb"] = torch.clamp(batch["rgb"], -1.0, 1.0)
        
        # Latents if specified
        if self.latent_shape is not None:
            C, H, W = self.latent_shape
            batch["latent"] = torch.randn(C, H, W) * 0.5
            batch["latent"] = torch.clamp(batch["latent"], -1.0, 1.0)
        
        # Optional embeddings
        if self.has_text_emb:
            batch["text_emb"] = torch.randn(384)  # Standard CLIP text embedding size
        
        if self.has_pov_emb:
            batch["pov_emb"] = torch.randn(512)  # Standard CLIP image embedding size
        
        return batch


def create_synthetic_dataloader(dataset, batch_size=2, shuffle=False):
    """Create a DataLoader from synthetic dataset."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # No multiprocessing for testing
        pin_memory=False
    )


def test_autoencoder_training():
    """Test autoencoder training path on CPU."""
    print("\n" + "="*70)
    print("TESTING AUTOENCODER TRAINING PATH")
    print("="*70)
    
    device = "cpu"
    device_obj = to_device(device)
    
    # Create temporary output directory
    temp_dir = Path(tempfile.mkdtemp(prefix="test_ae_cpu_"))
    
    try:
        # Build minimal config
        config = {
            "experiment": {
                "name": "test_ae_cpu",
                "save_path": str(temp_dir)
            },
            "autoencoder": {
                "encoder": {
                    "in_channels": 3,
                    "latent_channels": 4,
                    "base_channels": 16,  # Very small for CPU testing
                    "downsampling_steps": 3,  # 256 -> 128 -> 64 -> 32
                    "activation": "SiLU",
                    "norm_groups": 4,
                    "variational": False
                },
                "decoder": {
                    "latent_channels": 4,
                    "base_channels": 16,
                    "upsampling_steps": 3,  # 32 -> 64 -> 128 -> 256
                    "activation": "SiLU",
                    "norm_groups": 4,
                    "heads": [
                        {
                            "type": "RGBHead",
                            "name": "rgb",
                            "out_channels": 3,
                            "final_activation": "tanh"
                        }
                    ]
                }
            },
            "training": {
                "device": device,
                "batch_size": 2,
                "epochs": 1,
                "learning_rate": 0.001,
                "optimizer": "AdamW",
                "weight_decay": 0.0,
                "use_amp": False,  # No AMP on CPU
                "loss": {
                    "type": "CompositeLoss",
                    "losses": [
                        {
                            "type": "MSELoss",
                            "key": "rgb",
                            "target": "rgb",
                            "weight": 1.0
                        }
                    ]
                }
            }
        }
        
        # Test 1: Build model
        print("\n[1/8] Building autoencoder model...")
        try:
            model = build_model(config)
            model = model.to(device_obj)
            print(f"    ✓ Model built successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")
        except Exception as e:
            print(f"    ✗ Failed to build model: {e}")
            raise
        
        # Test 2: Build loss
        print("\n[2/8] Building loss function...")
        try:
            loss_fn = build_loss(config)
            print(f"    ✓ Loss function built: {type(loss_fn).__name__}")
        except Exception as e:
            print(f"    ✗ Failed to build loss: {e}")
            raise
        
        # Test 3: Build optimizer
        print("\n[3/8] Building optimizer...")
        try:
            optimizer = build_optimizer(model, config)
            print(f"    ✓ Optimizer built: {type(optimizer).__name__}")
        except Exception as e:
            print(f"    ✗ Failed to build optimizer: {e}")
            raise
        
        # Test 4: Build scheduler (optional)
        print("\n[4/8] Building scheduler...")
        try:
            scheduler = build_scheduler(optimizer, config)
            if scheduler:
                print(f"    ✓ Scheduler built: {type(scheduler).__name__}")
            else:
                print(f"    ✓ No scheduler (optional)")
        except Exception as e:
            print(f"    ✗ Failed to build scheduler: {e}")
            raise
        
        # Test 5: Create synthetic dataset and dataloader
        print("\n[5/8] Creating synthetic dataset...")
        try:
            dataset = SyntheticDataset(num_samples=4, image_size=256)
            train_loader = create_synthetic_dataloader(dataset, batch_size=2, shuffle=True)
            val_loader = create_synthetic_dataloader(dataset, batch_size=2, shuffle=False)
            print(f"    ✓ Dataset created: {len(dataset)} samples")
        except Exception as e:
            print(f"    ✗ Failed to create dataset: {e}")
            raise
        
        # Test 6: Create Trainer
        print("\n[6/8] Creating Trainer...")
        try:
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                use_amp=False,  # No AMP on CPU
                gradient_accumulation_steps=1,
                max_grad_norm=None
            )
            print(f"    ✓ Trainer created")
        except Exception as e:
            print(f"    ✗ Failed to create trainer: {e}")
            raise
        
        # Test 7: Run training epoch
        print("\n[7/8] Running training epoch...")
        try:
            avg_loss, avg_logs = trainer.train_epoch(
                dataloader=train_loader,
                loss_fn=loss_fn,
                epoch=1,
                step_fn=ae_step_fn
            )
            print(f"    ✓ Training epoch completed. Loss: {avg_loss:.6f}")
            
            # Test evaluation
            val_loss, val_logs = trainer.eval_epoch(
                dataloader=val_loader,
                loss_fn=loss_fn,
                step_fn=ae_eval_step_fn
            )
            print(f"    ✓ Evaluation completed. Val Loss: {val_loss:.6f}")
        except Exception as e:
            print(f"    ✗ Failed during training: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Test 8: Checkpoint saving/loading
        print("\n[8/8] Testing checkpoint save/load...")
        try:
            checkpoint_path = temp_dir / "test_checkpoint.pt"
            model.save_checkpoint(checkpoint_path, include_config=True)
            print(f"    ✓ Checkpoint saved")
            
            # Load checkpoint
            loaded_model = Autoencoder.load_checkpoint(checkpoint_path, map_location=device_obj)
            loaded_model = loaded_model.to(device_obj)
            print(f"    ✓ Checkpoint loaded")
            
            # Verify model works
            test_batch = next(iter(train_loader))
            test_batch = {k: v.to(device_obj) if isinstance(v, torch.Tensor) else v 
                         for k, v in test_batch.items()}
            with torch.no_grad():
                _ = loaded_model(test_batch["rgb"])
            print(f"    ✓ Loaded model forward pass works")
        except Exception as e:
            print(f"    ✗ Failed checkpoint test: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print("\n" + "="*70)
        print("✓ AUTOENCODER TRAINING TEST PASSED")
        print("="*70)
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print(f"✗ AUTOENCODER TRAINING TEST FAILED: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def test_diffusion_training():
    """Test diffusion training path on CPU."""
    print("\n" + "="*70)
    print("TESTING DIFFUSION TRAINING PATH")
    print("="*70)
    
    device = "cpu"
    device_obj = to_device(device)
    
    # Create temporary output directory
    temp_dir = Path(tempfile.mkdtemp(prefix="test_diff_cpu_"))
    
    try:
        # Build minimal diffusion config with autoencoder config
        # (This is more representative of actual usage and works better with checkpointing)
        # This is more representative of actual usage and works better with checkpointing
        config = {
            "experiment": {
                "name": "test_diff_cpu",
                "save_path": str(temp_dir)
            },
            "diffusion": {
                "autoencoder": {
                    "encoder": {
                        "in_channels": 3,
                        "latent_channels": 4,
                        "base_channels": 16,
                        "downsampling_steps": 3,
                        "activation": "SiLU",
                        "norm_groups": 4,
                        "variational": False
                    },
                    "decoder": {
                        "latent_channels": 4,
                        "base_channels": 16,
                        "upsampling_steps": 3,
                        "activation": "SiLU",
                        "norm_groups": 4,
                        "heads": [
                            {
                                "type": "RGBHead",
                                "name": "rgb",
                                "out_channels": 3,
                                "final_activation": "tanh"
                            }
                        ]
                    }
                },
                "unet": {
                    "type": "UnetWithAttention",
                    "in_channels": 4,  # Latent channels
                    "out_channels": 4,  # Must match in_channels for diffusion
                    "base_channels": 16,  # Very small for CPU
                    "depth": 3,  # Number of downsampling levels
                    "num_res_blocks": 1,
                    "time_dim": 128,
                    "norm_groups": 8,
                    "dropout": 0.0,
                    "use_attention": False  # Disable attention for minimal test
                },
                "scheduler": {
                    "type": "LinearScheduler",
                    "num_steps": 100
                }
            },
            "training": {
                "device": device,
                "batch_size": 2,
                "epochs": 1,
                "learning_rate": 0.001,
                "optimizer": "AdamW",
                "weight_decay": 0.0,
                "use_amp": False,  # No AMP on CPU
                "loss": {
                    "type": "CompositeLoss",
                    "losses": [
                        {
                            "type": "MSELoss",
                            "key": "pred_noise",
                            "target": "noise",
                            "weight": 1.0
                        }
                    ]
                }
            }
        }
        
        # Test 1: Build diffusion model
        print("\n[1/9] Building diffusion model...")
        try:
            diffusion_cfg = config["diffusion"].copy()
            # Pass save_path from experiment config
            exp_cfg = config.get("experiment", {})
            if exp_cfg.get("save_path"):
                diffusion_cfg["save_path"] = exp_cfg["save_path"]
            model = DiffusionModel(**diffusion_cfg)
            model = model.to(device_obj)
            print(f"    ✓ Diffusion model built. Parameters: {sum(p.numel() for p in model.parameters()):,}")
        except Exception as e:
            print(f"    ✗ Failed to build diffusion model: {e}")
            raise
        
        # Test 2: Build loss
        print("\n[2/9] Building loss function...")
        try:
            loss_fn = build_loss(config)
            print(f"    ✓ Loss function built: {type(loss_fn).__name__}")
        except Exception as e:
            print(f"    ✗ Failed to build loss: {e}")
            raise
        
        # Test 3: Build optimizer
        print("\n[3/9] Building optimizer...")
        try:
            optimizer = build_optimizer(model, config)
            print(f"    ✓ Optimizer built: {type(optimizer).__name__}")
        except Exception as e:
            print(f"    ✗ Failed to build optimizer: {e}")
            raise
        
        # Test 4: Build scheduler
        print("\n[4/9] Building scheduler...")
        try:
            scheduler = build_scheduler(optimizer, config)
            if scheduler:
                print(f"    ✓ Scheduler built: {type(scheduler).__name__}")
            else:
                print(f"    ✓ No scheduler (optional)")
        except Exception as e:
            print(f"    ✗ Failed to build scheduler: {e}")
            raise
        
        # Test 5: Create synthetic dataset with latents
        print("\n[5/9] Creating synthetic dataset with latents...")
        try:
            # Latent shape: (C, H, W) = (4, 32, 32) for 256x256 input with 3 downsampling steps
            latent_shape = (4, 32, 32)
            dataset = SyntheticDataset(
                num_samples=4,
                image_size=256,
                latent_shape=latent_shape,
                has_text_emb=False,
                has_pov_emb=False
            )
            train_loader = create_synthetic_dataloader(dataset, batch_size=2, shuffle=True)
            val_loader = create_synthetic_dataloader(dataset, batch_size=2, shuffle=False)
            print(f"    ✓ Dataset created: {len(dataset)} samples with latents")
        except Exception as e:
            print(f"    ✗ Failed to create dataset: {e}")
            raise
        
        # Test 6: Create Trainer
        print("\n[6/9] Creating Trainer...")
        try:
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                use_amp=False,  # No AMP on CPU
                gradient_accumulation_steps=1,
                max_grad_norm=None
            )
            # Set diffusion-specific attributes
            trainer.use_non_uniform_sampling = False
            trainer.cfg_dropout_rate = 0.0
            print(f"    ✓ Trainer created")
        except Exception as e:
            print(f"    ✗ Failed to create trainer: {e}")
            raise
        
        # Test 7: Run training epoch
        print("\n[7/9] Running training epoch...")
        try:
            avg_loss, avg_logs = trainer.train_epoch(
                dataloader=train_loader,
                loss_fn=loss_fn,
                epoch=1,
                step_fn=diffusion_step_fn
            )
            print(f"    ✓ Training epoch completed. Loss: {avg_loss:.6f}")
            
            # Test evaluation
            val_loss, val_logs = trainer.eval_epoch(
                dataloader=val_loader,
                loss_fn=loss_fn,
                step_fn=diffusion_eval_step_fn,
                limit_batches=1  # Just test 1 batch for speed
            )
            print(f"    ✓ Evaluation completed. Val Loss: {val_loss:.6f}")
        except Exception as e:
            print(f"    ✗ Failed during training: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Test 8: Checkpoint saving/loading
        print("\n[8/9] Testing checkpoint save/load...")
        try:
            checkpoint_path = temp_dir / "test_checkpoint.pt"
            model.save_checkpoint(checkpoint_path, include_config=True)
            print(f"    ✓ Checkpoint saved")
            
            # Load checkpoint (use saved config from checkpoint)
            loaded_model = DiffusionModel.load_checkpoint(
                checkpoint_path,
                map_location=device_obj,
                config=None  # Use saved config from checkpoint
            )
            loaded_model = loaded_model.to(device_obj)
            print(f"    ✓ Checkpoint loaded")
            
            # Verify model works
            test_batch = next(iter(train_loader))
            test_batch = {k: v.to(device_obj) if isinstance(v, torch.Tensor) else v 
                         for k, v in test_batch.items()}
            # Test forward pass with latents
            latents = test_batch["latent"]
            t = torch.randint(0, model.scheduler.num_steps, (latents.shape[0],), device=device_obj)
            noise = model.scheduler.randn_like(latents)
            with torch.no_grad():
                _ = loaded_model(latents, t, noise=noise)
            print(f"    ✓ Loaded model forward pass works")
        except Exception as e:
            print(f"    ✗ Failed checkpoint test: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Test 9: Test on-the-fly encoding (if encoder available)
        print("\n[9/9] Testing on-the-fly encoding...")
        try:
            if model._has_encoder:
                # Create dataset with RGB only (no latents)
                rgb_dataset = SyntheticDataset(
                    num_samples=2,
                    image_size=256,
                    latent_shape=None,  # No latents - will encode on-the-fly
                    has_text_emb=False,
                    has_pov_emb=False
                )
                rgb_loader = create_synthetic_dataloader(rgb_dataset, batch_size=2, shuffle=False)
                
                test_batch = next(iter(rgb_loader))
                test_batch = {k: v.to(device_obj) if isinstance(v, torch.Tensor) else v 
                             for k, v in test_batch.items()}
                
                # This should encode RGB to latents automatically
                loss, logs, _ = diffusion_step_fn(model, test_batch, 0, loss_fn, trainer)
                print(f"    ✓ On-the-fly encoding works. Loss: {loss.item():.6f}")
            else:
                print(f"    ✓ Skipped (no encoder in model)")
        except Exception as e:
            print(f"    ✗ Failed on-the-fly encoding test: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print("\n" + "="*70)
        print("✓ DIFFUSION TRAINING TEST PASSED")
        print("="*70)
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print(f"✗ DIFFUSION TRAINING TEST FAILED: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main():
    """Run all training tests."""
    print("\n" + "="*70)
    print("CPU TRAINING TEST SUITE")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: CPU (forced)")
    print("="*70)
    
    # Set deterministic seed for reproducibility
    set_deterministic(42)
    
    results = []
    
    # Test autoencoder
    results.append(("Autoencoder", test_autoencoder_training()))
    
    # Test diffusion
    results.append(("Diffusion", test_diffusion_training()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20s}: {status}")
    
    all_passed = all(passed for _, passed in results)
    print("="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

