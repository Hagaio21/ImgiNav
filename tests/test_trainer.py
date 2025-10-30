"""
Comprehensive tests for Trainer class.
Tests training loop, checkpointing, evaluation, metric logging,
and gradient clipping with both AutoEncoder and Diffusion models.
"""
import os
import sys
import tempfile
import shutil
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from training.trainer import Trainer
from training.base_trainer import BaseTrainer
from models.autoencoder import AutoEncoder
from models.diffusion import LatentDiffusion
from models.components.scheduler import LinearScheduler
from models.losses.custom_loss import StandardVAELoss, DiffusionLoss
from models.datasets import LayoutDataset
import torchvision.transforms as T


class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self, input_shape=(3, 64, 64), output_shape=(3, 64, 64)):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        # Simple conv layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, output_shape[0], 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, batch):
        if isinstance(batch, dict):
            x = batch.get("layout", batch.get("image", batch.get("x")))
        else:
            x = batch
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        
        return {
            "recon": x,
            "mu": torch.randn(x.size(0), 4, 8, 8),
            "logvar": torch.randn(x.size(0), 4, 8, 8),
            "input": x
        }
    
    def training_sample(self, batch_size=4, device=None):
        """Generate training samples."""
        if device is None:
            device = next(self.parameters()).device
        
        return torch.randn(batch_size, *self.output_shape, device=device)


class MockDiffusionModel(nn.Module):
    """Mock diffusion model for testing."""
    
    def __init__(self, latent_shape=(4, 8, 8)):
        super().__init__()
        self.latent_shape = latent_shape
        
        # Simple conv layers
        self.conv1 = nn.Conv2d(latent_shape[0], 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, latent_shape[0], 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, batch, cfg_dropout_prob=0.0):
        if isinstance(batch, dict):
            x = batch.get("layout", batch.get("image", batch.get("x")))
        else:
            x = batch
        
        # Mock diffusion forward pass
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        pred_noise = self.conv3(x)
        
        # Create mock outputs
        target_noise = torch.randn_like(pred_noise)
        timesteps = torch.randint(0, 100, (x.size(0),), device=x.device)
        
        return {
            "pred_noise": pred_noise,
            "target_noise": target_noise,
            "timesteps": timesteps,
            "x_t": x,
            "pred_x0": x,
            "original_latent": x
        }
    
    def training_sample(self, batch_size=4, device=None, num_steps=10):
        """Generate training samples."""
        if device is None:
            device = next(self.parameters()).device
        
        return torch.randn(batch_size, *self.latent_shape, device=device)


class TestBaseTrainer(unittest.TestCase):
    """Test BaseTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test BaseTrainer initialization."""
        trainer = BaseTrainer(
            epochs=10,
            log_interval=5,
            sample_interval=20,
            eval_interval=2,
            output_dir=self.temp_dir,
            ckpt_dir=os.path.join(self.temp_dir, "checkpoints"),
            device=self.device
        )
        
        self.assertEqual(trainer.epochs, 10)
        self.assertEqual(trainer.log_interval, 5)
        self.assertEqual(trainer.sample_interval, 20)
        self.assertEqual(trainer.eval_interval, 2)
        self.assertEqual(trainer.device, self.device)
        
        # Check directories were created
        self.assertTrue(os.path.exists(trainer.output_dir))
        self.assertTrue(os.path.exists(trainer.ckpt_dir))
        self.assertTrue(os.path.exists(os.path.join(trainer.output_dir, "samples")))
    
    def test_should_methods(self):
        """Test should_log, should_sample, should_validate methods."""
        trainer = BaseTrainer(
            log_interval=5,
            sample_interval=10,
            eval_interval=3,
            output_dir=self.temp_dir
        )
        
        # Test should_log
        self.assertTrue(trainer._should_log(5))
        self.assertTrue(trainer._should_log(10))
        self.assertFalse(trainer._should_log(3))
        self.assertFalse(trainer._should_log(7))
        
        # Test should_sample
        self.assertTrue(trainer._should_sample(10))
        self.assertTrue(trainer._should_sample(20))
        self.assertFalse(trainer._should_sample(5))
        self.assertFalse(trainer._should_sample(15))
        
        # Test should_validate
        self.assertTrue(trainer._should_validate(3))
        self.assertTrue(trainer._should_validate(6))
        self.assertFalse(trainer._should_validate(1))
        self.assertFalse(trainer._should_validate(4))
    
    def test_metric_accumulation(self):
        """Test metric accumulation and averaging."""
        trainer = BaseTrainer(output_dir=self.temp_dir)
        
        # Test accumulation
        metrics1 = {"loss": 1.0, "acc": 0.8}
        metrics2 = {"loss": 2.0, "acc": 0.9}
        metrics3 = {"loss": 3.0, "acc": 0.7}
        
        accumulator = {}
        accumulator = trainer._accumulate_metrics(metrics1, accumulator)
        accumulator = trainer._accumulate_metrics(metrics2, accumulator)
        accumulator = trainer._accumulate_metrics(metrics3, accumulator)
        
        self.assertEqual(accumulator["loss"], 6.0)
        self.assertAlmostEqual(accumulator["acc"], 2.4, places=5)
        
        # Test averaging
        avg_metrics = trainer._average_metrics(accumulator, 3)
        self.assertEqual(avg_metrics["loss"], 2.0)
        self.assertEqual(avg_metrics["acc"], 0.8)
    
    def test_metric_formatting(self):
        """Test metric string formatting."""
        trainer = BaseTrainer(output_dir=self.temp_dir)
        
        metrics = {"loss": 1.23456, "acc": 0.98765}
        formatted = trainer._format_metric_string(metrics)
        
        self.assertIn("loss=1.23456", formatted)
        self.assertIn("acc=0.98765", formatted)


class TestTrainer(unittest.TestCase):
    """Test Trainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create mock model and loss
        self.model = MockModel()
        self.loss_fn = StandardVAELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test Trainer initialization."""
        trainer = Trainer(
            model=self.model,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            epochs=5,
            log_interval=2,
            sample_interval=4,
            eval_interval=1,
            grad_clip=1.0,
            cfg_dropout_prob=0.1,
            num_training_samples=2,
            output_dir=self.temp_dir,
            ckpt_dir=os.path.join(self.temp_dir, "checkpoints"),
            device=self.device
        )
        
        self.assertEqual(trainer.epochs, 5)
        self.assertEqual(trainer.log_interval, 2)
        self.assertEqual(trainer.sample_interval, 4)
        self.assertEqual(trainer.eval_interval, 1)
        self.assertEqual(trainer.grad_clip, 1.0)
        self.assertEqual(trainer.cfg_dropout_prob, 0.1)
        self.assertEqual(trainer.num_training_samples, 2)
        self.assertEqual(trainer.device, self.device)
    
    def test_batch_device_movement(self):
        """Test batch device movement."""
        trainer = Trainer(
            model=self.model,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            output_dir=self.temp_dir,
            device=self.device
        )
        
        # Test dict batch
        batch = {
            "layout": torch.randn(2, 3, 64, 64),
            "scene_id": ["test1", "test2"]
        }
        
        device_batch = trainer._move_batch_to_device(batch)
        
        if self.device.type == "cuda":
            self.assertEqual(device_batch["layout"].device.type, "cuda")
        self.assertEqual(device_batch["scene_id"], ["test1", "test2"])
        
        # Test tensor batch
        tensor_batch = torch.randn(2, 3, 64, 64)
        device_tensor = trainer._move_batch_to_device(tensor_batch)
        
        if self.device.type == "cuda":
            self.assertEqual(device_tensor.device.type, "cuda")
    
    def test_training_loop_with_autoencoder(self):
        """Test training loop with AutoEncoder model."""
        # Create small AutoEncoder
        ae = AutoEncoder.from_shape(
            in_channels=3,
            out_channels=3,
            base_channels=16,
            latent_channels=2,
            image_size=64,
            latent_base=8
        )
        
        # Create trainer
        trainer = Trainer(
            model=ae,
            loss_fn=self.loss_fn,
            optimizer=optim.Adam(ae.parameters(), lr=1e-3),
            epochs=2,
            log_interval=1,
            sample_interval=1,
            eval_interval=1,
            output_dir=self.temp_dir,
            device=self.device
        )
        
        # Create mock dataloader
        train_data = [torch.randn(4, 3, 64, 64) for _ in range(2)]  # 2 batches of 4
        val_data = [torch.randn(4, 3, 64, 64) for _ in range(1)]    # 1 batch of 4
        
        train_loader = train_data  # Use list directly instead of iter()
        val_loader = val_data
        
        # Test training
        trainer.fit(train_loader, val_loader)
        
        # Check that checkpoints were created
        checkpoint_files = os.listdir(trainer.ckpt_dir)
        self.assertGreater(len(checkpoint_files), 0)
        
        # Check that samples were created
        sample_files = os.listdir(trainer.samples_dir)
        self.assertGreater(len(sample_files), 0)
    
    def test_training_loop_with_diffusion(self):
        """Test training loop with Diffusion model."""
        # Create mock diffusion model
        diffusion_model = MockDiffusionModel()
        
        # Create diffusion loss
        diffusion_loss = DiffusionLoss()
        
        # Create trainer
        trainer = Trainer(
            model=diffusion_model,
            loss_fn=diffusion_loss,
            optimizer=optim.Adam(diffusion_model.parameters(), lr=1e-3),
            epochs=2,
            log_interval=1,
            sample_interval=1,
            eval_interval=1,
            cfg_dropout_prob=0.1,
            output_dir=self.temp_dir,
            device=self.device
        )
        
        # Create mock dataloader
        train_data = [{"layout": torch.randn(4, 4, 8, 8)} for _ in range(2)]  # 2 batches
        val_data = [{"layout": torch.randn(4, 4, 8, 8)} for _ in range(1)]    # 1 batch
        
        train_loader = train_data  # Use list directly instead of iter()
        val_loader = val_data
        
        # Test training
        trainer.fit(train_loader, val_loader)
        
        # Check that checkpoints were created
        checkpoint_files = os.listdir(trainer.ckpt_dir)
        self.assertGreater(len(checkpoint_files), 0)
    
    def test_evaluation(self):
        """Test evaluation method."""
        trainer = Trainer(
            model=self.model,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            output_dir=self.temp_dir,
            device=self.device
        )
        
        # Create mock validation data
        val_data = [torch.randn(4, 3, 64, 64) for _ in range(2)]  # 2 batches
        val_loader = val_data  # Use list directly instead of iter()
        
        # Test evaluation
        val_metrics = trainer.evaluate(val_loader)
        
        self.assertIsInstance(val_metrics, dict)
        self.assertIn("mse", val_metrics)
        self.assertIn("kl", val_metrics)
    
    def test_gradient_clipping(self):
        """Test gradient clipping functionality."""
        # Create model with large gradients
        class LargeGradModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, batch):
                x = torch.randn(2, 10)
                return {"recon": self.linear(x), "mu": x, "logvar": x, "input": x}
            
            def training_sample(self, batch_size=4, device=None):
                return torch.randn(batch_size, 1)
        
        model = LargeGradModel()
        loss_fn = StandardVAELoss()
        optimizer = optim.Adam(model.parameters(), lr=1.0)  # High LR to create large gradients
        
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            grad_clip=0.5,  # Small clip value
            output_dir=self.temp_dir,
            device=self.device
        )
        
        # Create mock data
        train_data = [torch.randn(2, 10) for _ in range(1)]
        train_loader = train_data  # Use list directly instead of iter()
        
        # Test one training step
        batch = train_loader[0]  # Get first batch from list
        batch = trainer._move_batch_to_device(batch)
        
        # Forward pass
        outputs = trainer.model(batch)
        x = outputs.get("input", outputs.get("original_latent"))
        if x is None:
            x = batch
        loss_result = trainer.loss_fn(x, outputs)
        
        if isinstance(loss_result, tuple):
            total_loss = loss_result[0]
        else:
            total_loss = loss_result
        
        # Backward pass
        trainer.optimizer.zero_grad()
        total_loss.backward()
        
        # Check gradient clipping
        total_norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.grad_clip)
        self.assertLessEqual(total_norm.item(), trainer.grad_clip + 1e-6)
    
    def test_checkpoint_save_load(self):
        """Test checkpoint save/load functionality."""
        trainer = Trainer(
            model=self.model,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            output_dir=self.temp_dir,
            device=self.device
        )
        
        # Test checkpoint saving
        trainer._save_checkpoint(1)
        trainer._save_final_checkpoint()
        
        # Check that checkpoints were created
        checkpoint_files = os.listdir(trainer.ckpt_dir)
        self.assertIn("checkpoint_epoch_1.pt", checkpoint_files)
        self.assertIn("checkpoint_latest.pt", checkpoint_files)
        
        # Test state dict extraction
        state_dict = trainer._get_model_state_dict()
        self.assertIsInstance(state_dict, dict)
        self.assertIn("conv1.weight", state_dict)
    
    def test_sample_saving(self):
        """Test sample saving functionality."""
        trainer = Trainer(
            model=self.model,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            output_dir=self.temp_dir,
            device=self.device
        )
        
        # Test sample saving
        samples = torch.randn(4, 3, 64, 64)
        trainer._save_samples(samples, 100)
        
        # Check that sample was saved
        sample_files = os.listdir(trainer.samples_dir)
        self.assertIn("sample_step_100.png", sample_files)
        
        # Test with None samples
        trainer._save_samples(None, 101)
        # Should not crash
    
    def test_with_real_data(self):
        """Test Trainer with real data from test_dataset."""
        # Load a small dataset
        manifest_path = "test_dataset/manifests/layouts_manifest.csv"
        transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
        
        dataset = LayoutDataset(
            manifest_path=manifest_path,
            transform=transform,
            mode="all",
            skip_empty=True,
            return_embeddings=False
        )
        
        # Create small AutoEncoder
        ae = AutoEncoder.from_shape(
            in_channels=3,
            out_channels=3,
            base_channels=16,
            latent_channels=2,
            image_size=64,
            latent_base=8
        )
        
        # Create trainer
        trainer = Trainer(
            model=ae,
            loss_fn=StandardVAELoss(),
            optimizer=optim.Adam(ae.parameters(), lr=1e-3),
            epochs=2,
            log_interval=1,
            sample_interval=1,
            eval_interval=1,
            output_dir=self.temp_dir,
            device=self.device
        )
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
        
        # Test training with real data
        trainer.fit(train_loader, val_loader)
        
        # Check that training completed successfully
        checkpoint_files = os.listdir(trainer.ckpt_dir)
        self.assertGreater(len(checkpoint_files), 0)
        
        # Test evaluation with real data
        val_metrics = trainer.evaluate(val_loader)
        self.assertIsInstance(val_metrics, dict)
        self.assertIn("mse", val_metrics)
        self.assertIn("kl", val_metrics)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
