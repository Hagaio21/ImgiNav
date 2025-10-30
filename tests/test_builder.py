"""
Comprehensive tests for model builder functions.
Tests build_model, build_autoencoder, build_scheduler, build_unet,
and error handling with various configurations.
"""
import os
import sys
import tempfile
import shutil
import unittest
import torch
import yaml
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.builder import (
    build_model, build_autoencoder, build_scheduler, build_unet
)
from models.autoencoder import AutoEncoder
from models.diffusion import LatentDiffusion
from models.components.scheduler import LinearScheduler, CosineScheduler, QuadraticScheduler
from models.components.unet import DualUNet


class TestModelBuilder(unittest.TestCase):
    """Test build_model function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_build_autoencoder_from_shape(self):
        """Test building AutoEncoder from shape parameters."""
        model_cfg = {
            "type": "autoencoder",
            "in_channels": 3,
            "out_channels": 3,
            "base_channels": 32,
            "latent_channels": 4,
            "image_size": 64,
            "latent_base": 16,
            "norm": "batch",
            "act": "relu",
            "dropout": 0.1
        }
        
        model, aux_model = build_model(model_cfg, self.device)
        
        self.assertIsInstance(model, AutoEncoder)
        self.assertIsNone(aux_model)
        self.assertEqual(model.encoder.image_size, 64)
        self.assertEqual(model.encoder.latent_channels, 4)
        self.assertEqual(model.encoder.latent_base, 16)
    
    def test_build_vae_from_shape(self):
        """Test building VAE from shape parameters."""
        model_cfg = {
            "type": "vae",
            "in_channels": 3,
            "out_channels": 3,
            "base_channels": 32,
            "latent_channels": 4,
            "image_size": 64,
            "latent_base": 16,
            "num_classes": 10
        }
        
        model, aux_model = build_model(model_cfg, self.device)
        
        self.assertIsInstance(model, AutoEncoder)
        self.assertIsNone(aux_model)
        self.assertFalse(model.deterministic)  # VAE should be stochastic
    
    def test_build_ae_from_shape(self):
        """Test building AE from shape parameters."""
        model_cfg = {
            "type": "ae",
            "in_channels": 3,
            "out_channels": 3,
            "base_channels": 32,
            "latent_channels": 4,
            "image_size": 64,
            "latent_base": 16
        }
        
        model, aux_model = build_model(model_cfg, self.device)
        
        self.assertIsInstance(model, AutoEncoder)
        self.assertIsNone(aux_model)
        self.assertTrue(model.deterministic)  # AE should be deterministic
    
    def test_build_autoencoder_from_config(self):
        """Test building AutoEncoder from config dict."""
        # Create a config dict
        config = {
            "encoder": {
                "in_channels": 3,
                "layers": [
                    {"out_channels": 32, "stride": 2},
                    {"out_channels": 64, "stride": 2}
                ],
                "image_size": 64,
                "latent_channels": 4,
                "latent_base": 16,
                "global_norm": "batch",
                "global_act": "relu"
            },
            "decoder": {
                "out_channels": 3,
                "image_size": 64,
                "latent_channels": 4,
                "latent_base": 16,
                "global_norm": "batch",
                "global_act": "relu"
            }
        }
        
        model_cfg = {
            "type": "autoencoder",
            "config": config
        }
        
        model, aux_model = build_model(model_cfg, self.device)
        
        self.assertIsInstance(model, AutoEncoder)
        self.assertIsNone(aux_model)
    
    def test_build_diffusion_model(self):
        """Test building Diffusion model."""
        # Create autoencoder config
        ae_config = {
            "encoder": {
                "in_channels": 3,
                "layers": [
                    {"out_channels": 32, "stride": 2},
                    {"out_channels": 64, "stride": 2}
                ],
                "image_size": 64,
                "latent_channels": 4,
                "latent_base": 16
            },
            "decoder": {
                "out_channels": 3,
                "image_size": 64,
                "latent_channels": 4,
                "latent_base": 16
            }
        }
        
        # Create UNet config
        unet_config = {
            "in_channels": 4,
            "out_channels": 4,
            "cond_channels": 0,
            "base_channels": 32,
            "depth": 2,
            "num_res_blocks": 1,
            "time_dim": 64,
            "fusion_mode": "none"
        }
        
        model_cfg = {
            "type": "diffusion",
            "autoencoder": {
                "config": ae_config
            },
            "diffusion": {
                "unet": {
                    "config": unet_config
                },
                "scheduler": {
                    "type": "linear",
                    "num_steps": 100
                }
            }
        }
        
        model, aux_model = build_model(model_cfg, self.device)
        
        self.assertIsInstance(model, LatentDiffusion)
        self.assertIsInstance(aux_model, AutoEncoder)
        self.assertIsNotNone(model.autoencoder)
        self.assertEqual(model.latent_shape, (4, 16, 16))
    
    def test_invalid_model_type(self):
        """Test error handling for invalid model type."""
        model_cfg = {
            "type": "invalid_model"
        }
        
        with self.assertRaises(ValueError):
            build_model(model_cfg, self.device)
    
    def test_missing_autoencoder_config(self):
        """Test error handling for missing autoencoder config in diffusion."""
        model_cfg = {
            "type": "diffusion",
            "diffusion": {
                "unet": {"config": {}},
                "scheduler": {"type": "linear", "num_steps": 100}
            }
        }
        
        with self.assertRaises(ValueError):
            build_model(model_cfg, self.device)


class TestAutoEncoderBuilder(unittest.TestCase):
    """Test build_autoencoder function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_build_autoencoder_from_config(self):
        """Test building autoencoder from config."""
        config = {
            "encoder": {
                "in_channels": 3,
                "layers": [
                    {"out_channels": 32, "stride": 2},
                    {"out_channels": 64, "stride": 2}
                ],
                "image_size": 64,
                "latent_channels": 4,
                "latent_base": 16
            },
            "decoder": {
                "out_channels": 3,
                "image_size": 64,
                "latent_channels": 4,
                "latent_base": 16
            }
        }
        
        ae_cfg = {
            "config": config
        }
        
        autoencoder = build_autoencoder(ae_cfg)
        
        self.assertIsInstance(autoencoder, AutoEncoder)
        self.assertEqual(autoencoder.encoder.image_size, 64)
        self.assertEqual(autoencoder.encoder.latent_channels, 4)
    
    def test_build_autoencoder_with_checkpoint(self):
        """Test building autoencoder with checkpoint."""
        # Create config
        config = {
            "encoder": {
                "in_channels": 3,
                "layers": [{"out_channels": 32, "stride": 2}],
                "image_size": 64,
                "latent_channels": 4,
                "latent_base": 16
            },
            "decoder": {
                "out_channels": 3,
                "image_size": 64,
                "latent_channels": 4,
                "latent_base": 16
            }
        }
        
        # Create checkpoint
        checkpoint_path = os.path.join(self.temp_dir, "test_checkpoint.pt")
        dummy_state = {"conv1.weight": torch.randn(32, 3, 3, 3)}
        torch.save(dummy_state, checkpoint_path)
        
        ae_cfg = {
            "config": config,
            "checkpoint": checkpoint_path
        }
        
        autoencoder = build_autoencoder(ae_cfg)
        
        # Check that autoencoder was created successfully
        self.assertIsInstance(autoencoder, AutoEncoder)
    
    def test_missing_config(self):
        """Test error handling for missing config."""
        ae_cfg = {}
        
        with self.assertRaises(ValueError):
            build_autoencoder(ae_cfg)
    
    def test_nonexistent_checkpoint(self):
        """Test handling of nonexistent checkpoint."""
        config = {
            "encoder": {
                "in_channels": 3,
                "layers": [{"out_channels": 32, "stride": 2}],
                "image_size": 64,
                "latent_channels": 4,
                "latent_base": 16
            },
            "decoder": {
                "out_channels": 3,
                "image_size": 64,
                "latent_channels": 4,
                "latent_base": 16
            }
        }
        
        ae_cfg = {
            "config": config,
            "checkpoint": "nonexistent_checkpoint.pt"
        }
        
        # Should not raise error, just print warning
        autoencoder = build_autoencoder(ae_cfg)
        self.assertIsInstance(autoencoder, AutoEncoder)


class TestSchedulerBuilder(unittest.TestCase):
    """Test build_scheduler function."""
    
    def test_build_linear_scheduler(self):
        """Test building LinearScheduler."""
        sched_cfg = {
            "type": "linear",
            "num_steps": 1000
        }
        
        scheduler = build_scheduler(sched_cfg)
        
        self.assertIsInstance(scheduler, LinearScheduler)
        self.assertEqual(scheduler.num_steps, 1000)
    
    def test_build_cosine_scheduler(self):
        """Test building CosineScheduler."""
        sched_cfg = {
            "type": "cosine",
            "num_steps": 500
        }
        
        scheduler = build_scheduler(sched_cfg)
        
        self.assertIsInstance(scheduler, CosineScheduler)
        self.assertEqual(scheduler.num_steps, 500)
    
    def test_build_quadratic_scheduler(self):
        """Test building QuadraticScheduler."""
        sched_cfg = {
            "type": "quadratic",
            "num_steps": 200
        }
        
        scheduler = build_scheduler(sched_cfg)
        
        self.assertIsInstance(scheduler, QuadraticScheduler)
        self.assertEqual(scheduler.num_steps, 200)
    
    def test_default_parameters(self):
        """Test default parameters."""
        sched_cfg = {}
        
        scheduler = build_scheduler(sched_cfg)
        
        self.assertIsInstance(scheduler, CosineScheduler)  # Default type
        self.assertEqual(scheduler.num_steps, 1000)  # Default num_steps
    
    def test_invalid_scheduler_type(self):
        """Test error handling for invalid scheduler type."""
        sched_cfg = {
            "type": "invalid_scheduler",
            "num_steps": 100
        }
        
        with self.assertRaises(ValueError):
            build_scheduler(sched_cfg)


class TestUNetBuilder(unittest.TestCase):
    """Test build_unet function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_build_unet_from_dict(self):
        """Test building UNet from config dict."""
        unet_cfg = {
            "in_channels": 4,
            "out_channels": 4,
            "cond_channels": 0,
            "base_channels": 64,  # Use 64 to match existing configs (divisible by 8)
            "depth": 2,
            "num_res_blocks": 1,
            "time_dim": 64,
            "fusion_mode": "none"
        }
        
        unet = build_unet(unet_cfg)
        
        self.assertIsInstance(unet, DualUNet)
        self.assertEqual(unet.in_channels, 4)
        self.assertEqual(unet.out_channels, 4)
        self.assertEqual(unet.base_channels, 64)
        self.assertEqual(unet.depth, 2)
    
    def test_build_unet_from_file(self):
        """Test building UNet from config file."""
        # Create config file
        unet_cfg = {
            "in_channels": 8,  # Changed from 4 to 8 (divisible by 8)
            "out_channels": 8,  # Changed from 4 to 8 (divisible by 8)
            "cond_channels": 0,
            "base_channels": 16,  # Changed from 32 to 16 (divisible by 8)
            "depth": 2,
            "num_res_blocks": 1,
            "time_dim": 64,
            "fusion_mode": "none"
        }
        
        config_path = os.path.join(self.temp_dir, "unet_config.yaml")
        with open(config_path, 'w') as f:
            yaml.safe_dump(unet_cfg, f)
        
        unet = build_unet(config_path)
        
        self.assertIsInstance(unet, DualUNet)
        self.assertEqual(unet.in_channels, 8)
        self.assertEqual(unet.out_channels, 8)
    
    def test_build_unet_with_nested_config(self):
        """Test building UNet with nested config."""
        unet_cfg = {
            "config": "path/to/config.yaml"
        }
        
        # Create config file
        actual_config = {
            "in_channels": 8,  # Changed from 4 to 8 (divisible by 8)
            "out_channels": 8,  # Changed from 4 to 8 (divisible by 8)
            "cond_channels": 0,
            "base_channels": 16,  # Changed from 32 to 16 (divisible by 8)
            "depth": 2,
            "num_res_blocks": 1,
            "time_dim": 64,
            "fusion_mode": "none"
        }
        
        config_path = os.path.join(self.temp_dir, "unet_config.yaml")
        with open(config_path, 'w') as f:
            yaml.safe_dump(actual_config, f)
        
        unet_cfg["config"] = config_path
        unet = build_unet(unet_cfg)
        
        self.assertIsInstance(unet, DualUNet)
        self.assertEqual(unet.in_channels, 8)
    
    def test_build_unet_with_model_key(self):
        """Test building UNet with 'model' key in config."""
        unet_cfg = {
            "model": {
                "in_channels": 8,  # Changed from 4 to 8 (divisible by 8)
                "out_channels": 8,  # Changed from 4 to 8 (divisible by 8)
                "cond_channels": 0,
                "base_channels": 16,  # Changed from 32 to 16 (divisible by 8)
                "depth": 2,
                "num_res_blocks": 1,
                "time_dim": 64,
                "fusion_mode": "none"
            }
        }
        
        unet = build_unet(unet_cfg)
        
        self.assertIsInstance(unet, DualUNet)
        self.assertEqual(unet.in_channels, 8)
    
    def test_build_unet_with_conditioning(self):
        """Test building UNet with conditioning."""
        unet_cfg = {
            "in_channels": 8,  # Changed from 4 to 8 (divisible by 8)
            "out_channels": 8,  # Changed from 4 to 8 (divisible by 8)
            "cond_channels": 64,
            "base_channels": 16,  # Changed from 32 to 16 (divisible by 8)
            "depth": 2,
            "num_res_blocks": 1,
            "time_dim": 64,
            "fusion_mode": "concat"
        }
        
        unet = build_unet(unet_cfg)
        
        self.assertIsInstance(unet, DualUNet)
        self.assertEqual(unet.cond_channels, 64)
        self.assertEqual(unet.fusion_mode, "concat")
    
    def test_build_unet_forward_pass(self):
        """Test UNet forward pass after building."""
        unet_cfg = {
            "in_channels": 8,  # Changed from 4 to 8 (divisible by 8)
            "out_channels": 8,  # Changed from 4 to 8 (divisible by 8)
            "cond_channels": 0,
            "base_channels": 16,
            "depth": 2,
            "num_res_blocks": 1,
            "time_dim": 32,
            "fusion_mode": "none"
        }
        
        unet = build_unet(unet_cfg)
        
        # Test forward pass
        x = torch.randn(2, 8, 16, 16)  # Changed from 4 to 8 channels
        t = torch.randint(0, 100, (2,))
        
        output = unet(x, t)
        
        self.assertEqual(output.shape, (2, 8, 16, 16))  # Changed from 4 to 8 channels
    
    def test_build_unet_with_conditioning_forward(self):
        """Test UNet forward pass with conditioning."""
        unet_cfg = {
            "in_channels": 8,  # Changed from 4 to 8 (divisible by 8)
            "out_channels": 8,  # Changed from 4 to 8 (divisible by 8)
            "cond_channels": 32,
            "base_channels": 16,
            "depth": 2,
            "num_res_blocks": 1,
            "time_dim": 32,
            "fusion_mode": "concat"
        }
        
        unet = build_unet(unet_cfg)
        
        # Test forward pass with condition
        x = torch.randn(2, 8, 16, 16)  # Changed from 4 to 8 channels
        t = torch.randint(0, 100, (2,))
        cond = torch.randn(2, 32, 16, 16)
        
        output = unet(x, t, cond)
        
        self.assertEqual(output.shape, (2, 8, 16, 16))  # Changed from 4 to 8 channels
    
    def test_build_unet_with_none_condition(self):
        """Test UNet forward pass with None condition."""
        unet_cfg = {
            "in_channels": 8,  # Changed from 4 to 8 (divisible by 8)
            "out_channels": 8,  # Changed from 4 to 8 (divisible by 8)
            "cond_channels": 32,
            "base_channels": 16,
            "depth": 2,
            "num_res_blocks": 1,
            "time_dim": 32,
            "fusion_mode": "concat"
        }
        
        unet = build_unet(unet_cfg)
        
        # Test forward pass with None condition
        x = torch.randn(2, 8, 16, 16)  # Changed from 4 to 8 channels
        t = torch.randint(0, 100, (2,))
        
        output = unet(x, t, None)
        
        self.assertEqual(output.shape, (2, 8, 16, 16))  # Changed from 4 to 8 channels


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
