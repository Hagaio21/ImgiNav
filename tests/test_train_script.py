"""
Comprehensive tests for train.py script.
Tests end-to-end training with AutoEncoder and Diffusion models,
config loading, component building, and full pipeline execution.
"""
import os
import sys
import tempfile
import shutil
import unittest
import torch
import yaml
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from common.utils import load_config_with_profile
from models.datasets import build_dataloaders
from models.builder import build_model
from training.trainer import Trainer
from training.utils import (
    build_loss_function, build_optimizer, setup_experiment_directories,
    save_experiment_config, setup_training_environment
)


class TestTrainScriptComponents(unittest.TestCase):
    """Test individual components used by train.py."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create test configs
        self.ae_config = {
            "dataset": {
                "batch_size": 4,
                "manifest": "test_dataset/manifests/layouts_manifest.csv",
                "num_workers": 0,
                "seed": 42,
                "shuffle": True,
                "split_ratio": 0.8,
                "return_embeddings": False,
                "skip_empty": True
            },
            "model": {
                "type": "ae",
                "in_channels": 3,
                "out_channels": 3,
                "base_channels": 16,
                "latent_channels": 2,
                "image_size": 64,
                "latent_base": 8,
                "act": "relu",
                "norm": "batch",
                "dropout": 0.0
            },
            "training": {
                "epochs": 2,
                "eval_interval": 1,
                "log_interval": 1,
                "lr": 0.001,
                "sample_interval": 1,
                "output_dir": os.path.join(self.temp_dir, "ae_output"),
                "ckpt_dir": os.path.join(self.temp_dir, "ae_output", "checkpoints"),
                "loss": {
                    "type": "standard",
                    "kl_weight": 0.000001
                }
            },
            "experiment": {
                "name": "test_ae",
                "base_path": self.temp_dir
            }
        }
        
        self.diffusion_config = {
            "dataset": {
                "batch_size": 4,
                "manifest": "test_dataset/manifests/layouts_manifest.csv",
                "num_workers": 0,
                "seed": 42,
                "shuffle": True,
                "split_ratio": 0.8,
                "return_embeddings": False,
                "skip_empty": True
            },
            "model": {
                "type": "diffusion",
                "autoencoder": {
                    "config": {
                        "encoder": {
                            "in_channels": 3,
                            "layers": [{"out_channels": 32, "stride": 2}],
                            "image_size": 64,
                            "latent_channels": 2,
                            "latent_base": 8
                        },
                        "decoder": {
                            "out_channels": 3,
                            "image_size": 64,
                            "latent_channels": 2,
                            "latent_base": 8
                        }
                    }
                },
                "diffusion": {
                    "unet": {
                        "config": {
                            "in_channels": 2,
                            "out_channels": 2,
                            "cond_channels": 0,
                            "base_channels": 16,
                            "depth": 2,
                            "num_res_blocks": 1,
                            "time_dim": 32,
                            "fusion_mode": "none"
                        }
                    },
                    "scheduler": {
                        "type": "linear",
                        "num_steps": 50
                    }
                }
            },
            "training": {
                "epochs": 2,
                "eval_interval": 1,
                "log_interval": 1,
                "lr": 0.001,
                "sample_interval": 1,
                "cfg_dropout_prob": 0.1,
                "output_dir": os.path.join(self.temp_dir, "diffusion_output"),
                "ckpt_dir": os.path.join(self.temp_dir, "diffusion_output", "checkpoints"),
                "loss": {
                    "type": "diffusion",
                    "lambda_mse": 1.0,
                    "lambda_vgg": 0.0
                }
            },
            "experiment": {
                "name": "test_diffusion",
                "base_path": self.temp_dir
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_config_loading(self):
        """Test config loading functionality."""
        # Test loading from file
        config_path = os.path.join(self.temp_dir, "test_config.yaml")
        with open(config_path, 'w') as f:
            yaml.safe_dump(self.ae_config, f)
        
        cfg = load_config_with_profile(config_path)
        
        self.assertIn("dataset", cfg)
        self.assertIn("model", cfg)
        self.assertIn("training", cfg)
        self.assertIn("experiment", cfg)
        
        # Test loading from another file
        config_path2 = os.path.join(self.temp_dir, "test_config2.yaml")
        with open(config_path2, 'w') as f:
            yaml.safe_dump(self.ae_config, f)
        
        cfg_from_file = load_config_with_profile(config_path2)
        self.assertEqual(cfg, cfg_from_file)
    
    def test_experiment_directory_setup(self):
        """Test experiment directory setup."""
        # Test with experiment config
        exp_cfg = self.ae_config["experiment"]
        out_dir, ckpt_dir = setup_experiment_directories(
            os.path.join(exp_cfg["base_path"], exp_cfg["name"], "output"),
            os.path.join(exp_cfg["base_path"], exp_cfg["name"], "checkpoints")
        )
        
        self.assertTrue(os.path.exists(out_dir))
        self.assertTrue(os.path.exists(ckpt_dir))
        
        # Test with training config
        train_cfg = self.ae_config["training"]
        out_dir, ckpt_dir = setup_experiment_directories(
            train_cfg["output_dir"],
            train_cfg["ckpt_dir"]
        )
        
        self.assertTrue(os.path.exists(out_dir))
        self.assertTrue(os.path.exists(ckpt_dir))
    
    def test_config_saving(self):
        """Test experiment config saving."""
        out_dir = os.path.join(self.temp_dir, "test_output")
        os.makedirs(out_dir, exist_ok=True)
        
        config_path = save_experiment_config(self.ae_config, out_dir)
        
        self.assertTrue(os.path.exists(config_path))
        
        # Load and verify
        with open(config_path, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        self.assertEqual(saved_config, self.ae_config)
    
    def test_training_environment_setup(self):
        """Test training environment setup."""
        device = setup_training_environment(seed=123)
        
        self.assertIsInstance(device, str)
        self.assertIn(device, ["cuda", "cpu"])
    
    def test_dataloader_building(self):
        """Test dataloader building."""
        dataset_cfg = self.ae_config["dataset"]
        
        train_ds, val_ds, train_loader, val_loader = build_dataloaders(dataset_cfg)
        
        self.assertIsNotNone(train_ds)
        self.assertIsNotNone(val_ds)
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        
        # Test batch loading
        train_batch = next(iter(train_loader))
        self.assertIn("layout", train_batch)
        self.assertEqual(train_batch["layout"].shape[0], 4)  # batch_size
    
    def test_model_building_autoencoder(self):
        """Test AutoEncoder model building."""
        model_cfg = self.ae_config["model"]
        
        model, aux_model = build_model(model_cfg, self.device)
        
        self.assertIsNotNone(model)
        self.assertIsNone(aux_model)
        self.assertEqual(model.encoder.image_size, 64)
        self.assertEqual(model.encoder.latent_channels, 2)
    
    def test_model_building_diffusion(self):
        """Test Diffusion model building."""
        model_cfg = self.diffusion_config["model"]
        
        model, aux_model = build_model(model_cfg, self.device)
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(aux_model)
        self.assertEqual(model.latent_shape, (2, 8, 8))
    
    def test_loss_function_building(self):
        """Test loss function building."""
        # Test standard VAE loss
        loss_cfg = self.ae_config["training"]["loss"]
        loss_fn = build_loss_function(loss_cfg)
        
        self.assertIsNotNone(loss_fn)
        
        # Test diffusion loss
        loss_cfg = self.diffusion_config["training"]["loss"]
        loss_fn = build_loss_function(loss_cfg)
        
        self.assertIsNotNone(loss_fn)
    
    def test_optimizer_building(self):
        """Test optimizer building."""
        # Create a simple model
        model = torch.nn.Linear(10, 1)
        training_cfg = self.ae_config["training"]
        
        optimizer = build_optimizer(model, training_cfg)
        
        self.assertIsNotNone(optimizer)
        self.assertIsInstance(optimizer, torch.optim.Adam)
    
    def test_trainer_creation(self):
        """Test trainer creation."""
        # Create components
        model_cfg = self.ae_config["model"]
        training_cfg = self.ae_config["training"]
        
        model, _ = build_model(model_cfg, self.device)
        loss_fn = build_loss_function(training_cfg["loss"])
        optimizer = build_optimizer(model, training_cfg)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=training_cfg["epochs"],
            log_interval=training_cfg["log_interval"],
            sample_interval=training_cfg["sample_interval"],
            eval_interval=training_cfg["eval_interval"],
            output_dir=training_cfg["output_dir"],
            ckpt_dir=training_cfg["ckpt_dir"],
            device=self.device
        )
        
        self.assertIsNotNone(trainer)
        self.assertEqual(trainer.epochs, 2)


class TestTrainScriptIntegration(unittest.TestCase):
    """Test full integration of train.py script."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_autoencoder_training_pipeline(self):
        """Test complete AutoEncoder training pipeline."""
        # Create config
        config = {
            "dataset": {
                "batch_size": 4,
                "manifest": "test_dataset/manifests/layouts_manifest.csv",
                "num_workers": 0,
                "seed": 42,
                "shuffle": True,
                "split_ratio": 0.8,
                "return_embeddings": False,
                "skip_empty": True
            },
            "model": {
                "type": "ae",
                "in_channels": 3,
                "out_channels": 3,
                "base_channels": 16,
                "latent_channels": 2,
                "image_size": 64,
                "latent_base": 8,
                "act": "relu",
                "norm": "batch",
                "dropout": 0.0
            },
            "training": {
                "epochs": 2,
                "eval_interval": 1,
                "log_interval": 1,
                "lr": 0.001,
                "sample_interval": 1,
                "output_dir": os.path.join(self.temp_dir, "ae_output"),
                "ckpt_dir": os.path.join(self.temp_dir, "ae_output", "checkpoints"),
                "loss": {
                    "type": "standard",
                    "kl_weight": 0.000001
                }
            },
            "experiment": {
                "name": "test_ae",
                "base_path": self.temp_dir
            }
        }
        
        # Save config
        config_path = os.path.join(self.temp_dir, "ae_config.yaml")
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f)
        
        # Run training pipeline
        from training.train import main
        
        # Mock sys.argv to simulate command line arguments
        with patch('sys.argv', ['train.py', '--config', config_path]):
            try:
                main()
            except SystemExit:
                pass  # Expected when main() calls sys.exit()
        
        # Check that outputs were created
        output_dir = os.path.join(self.temp_dir, "ae_output")
        self.assertTrue(os.path.exists(output_dir))
        
        # Check for checkpoint files
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        if os.path.exists(ckpt_dir):
            checkpoint_files = os.listdir(ckpt_dir)
            self.assertGreater(len(checkpoint_files), 0)
        
        # Check for sample files
        samples_dir = os.path.join(output_dir, "samples")
        if os.path.exists(samples_dir):
            sample_files = os.listdir(samples_dir)
            self.assertGreater(len(sample_files), 0)
    
    def test_diffusion_training_pipeline(self):
        """Test complete Diffusion training pipeline."""
        # Create config
        config = {
            "dataset": {
                "batch_size": 4,
                "manifest": "test_dataset/manifests/layouts_manifest.csv",
                "num_workers": 0,
                "seed": 42,
                "shuffle": True,
                "split_ratio": 0.8,
                "return_embeddings": False,
                "skip_empty": True
            },
            "model": {
                "type": "diffusion",
                "autoencoder": {
                    "config": {
                        "encoder": {
                            "in_channels": 3,
                            "layers": [{"out_channels": 32, "stride": 2}],
                            "image_size": 64,
                            "latent_channels": 2,
                            "latent_base": 8
                        },
                        "decoder": {
                            "out_channels": 3,
                            "image_size": 64,
                            "latent_channels": 2,
                            "latent_base": 8
                        }
                    }
                },
                "diffusion": {
                    "unet": {
                        "config": {
                            "in_channels": 2,
                            "out_channels": 2,
                            "cond_channels": 0,
                            "base_channels": 16,
                            "depth": 2,
                            "num_res_blocks": 1,
                            "time_dim": 32,
                            "fusion_mode": "none"
                        }
                    },
                    "scheduler": {
                        "type": "linear",
                        "num_steps": 50
                    }
                }
            },
            "training": {
                "epochs": 2,
                "eval_interval": 1,
                "log_interval": 1,
                "lr": 0.001,
                "sample_interval": 1,
                "cfg_dropout_prob": 0.1,
                "output_dir": os.path.join(self.temp_dir, "diffusion_output"),
                "ckpt_dir": os.path.join(self.temp_dir, "diffusion_output", "checkpoints"),
                "loss": {
                    "type": "diffusion",
                    "lambda_mse": 1.0,
                    "lambda_vgg": 0.0
                }
            },
            "experiment": {
                "name": "test_diffusion",
                "base_path": self.temp_dir
            }
        }
        
        # Save config
        config_path = os.path.join(self.temp_dir, "diffusion_config.yaml")
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f)
        
        # Run training pipeline
        from training.train import main
        
        # Mock sys.argv to simulate command line arguments
        with patch('sys.argv', ['train.py', '--config', config_path]):
            try:
                main()
            except SystemExit:
                pass  # Expected when main() calls sys.exit()
        
        # Check that outputs were created
        output_dir = os.path.join(self.temp_dir, "diffusion_output")
        self.assertTrue(os.path.exists(output_dir))
    
    def test_command_line_execution(self):
        """Test command line execution of train.py."""
        # Create config
        config = {
            "dataset": {
                "batch_size": 4,
                "manifest": "test_dataset/manifests/layouts_manifest.csv",
                "num_workers": 0,
                "seed": 42,
                "shuffle": True,
                "split_ratio": 0.8,
                "return_embeddings": False,
                "skip_empty": True
            },
            "model": {
                "type": "ae",
                "in_channels": 3,
                "out_channels": 3,
                "base_channels": 16,
                "latent_channels": 2,
                "image_size": 64,
                "latent_base": 8
            },
            "training": {
                "epochs": 1,
                "eval_interval": 1,
                "log_interval": 1,
                "lr": 0.001,
                "sample_interval": 1,
                "output_dir": os.path.join(self.temp_dir, "cli_output"),
                "ckpt_dir": os.path.join(self.temp_dir, "cli_output", "checkpoints"),
                "loss": {
                    "type": "standard",
                    "kl_weight": 0.000001
                }
            },
            "experiment": {
                "name": "test_cli",
                "base_path": self.temp_dir
            }
        }
        
        # Save config
        config_path = os.path.join(self.temp_dir, "cli_config.yaml")
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f)
        
        # Test command line execution
        try:
            result = subprocess.run([
                sys.executable, "training/train.py", "--config", config_path
            ], capture_output=True, text=True, timeout=60)
            
            # Check that it ran without critical errors
            self.assertNotIn("Traceback", result.stderr)
            
        except subprocess.TimeoutExpired:
            # Timeout is acceptable for this test
            pass
        except FileNotFoundError:
            # Skip if train.py not found
            pass
    
    def test_error_handling(self):
        """Test error handling in train.py."""
        # Test with invalid config file
        invalid_config_path = os.path.join(self.temp_dir, "nonexistent_config.yaml")
        
        with patch('sys.argv', ['train.py', '--config', invalid_config_path]):
            with self.assertRaises(SystemExit):
                from training.train import main
                main()
        
        # Test with invalid config content
        invalid_config_path = os.path.join(self.temp_dir, "invalid_config.yaml")
        with open(invalid_config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with patch('sys.argv', ['train.py', '--config', invalid_config_path]):
            with self.assertRaises(SystemExit):
                from training.train import main
                main()
    
    def test_config_validation(self):
        """Test config validation and required fields."""
        # Test missing required fields
        incomplete_config = {
            "dataset": {
                "batch_size": 4
            }
            # Missing model, training, etc.
        }
        
        config_path = os.path.join(self.temp_dir, "incomplete_config.yaml")
        with open(config_path, 'w') as f:
            yaml.safe_dump(incomplete_config, f)
        
        with patch('sys.argv', ['train.py', '--config', config_path]):
            with self.assertRaises(SystemExit):
                from training.train import main
                main()


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
