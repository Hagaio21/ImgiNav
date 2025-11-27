"""
Tests for autoencoder training - focusing on structure, not correctness.
"""
import pytest
import torch
from pathlib import Path
import tempfile

from training.engine import Trainer
from models.autoencoder import Autoencoder
from models.losses.base_loss import LOSS_REGISTRY


class TestTrainingLoopStructure:
    """Test training loop structure."""
    
    def test_trainer_can_be_created(self, autoencoder_model):
        """Test that Trainer can be created for autoencoder."""
        optimizer = torch.optim.AdamW(autoencoder_model.parameters(), lr=0.001)
        
        trainer = Trainer(
            model=autoencoder_model,
            optimizer=optimizer,
            device="cpu",
            use_amp=False,
        )
        
        assert trainer is not None
    
    def test_loss_function_structure(self, test_autoencoder_config):
        """Test that loss function can be built from config."""
        from training.utils import build_loss
        
        loss_fn = build_loss(test_autoencoder_config)
        
        assert loss_fn is not None
        # Should be a callable
        assert callable(loss_fn)
    
    def test_optimizer_structure(self, autoencoder_model, test_autoencoder_config):
        """Test that optimizer can be built from config."""
        from training.utils import build_optimizer
        
        optimizer = build_optimizer(autoencoder_model, test_autoencoder_config)
        
        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.Optimizer)
    
    def test_scheduler_structure(self, autoencoder_model, test_autoencoder_config):
        """Test that scheduler can be built from config."""
        from training.utils import build_optimizer, build_scheduler
        
        optimizer = build_optimizer(autoencoder_model, test_autoencoder_config)
        scheduler = build_scheduler(optimizer, test_autoencoder_config)
        
        # Scheduler may be None if not configured
        assert scheduler is None or isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler)


class TestCheckpointSavingDuringTraining:
    """Test checkpoint saving during training structure."""
    
    def test_save_training_checkpoint_structure(self, autoencoder_model, temp_dir):
        """Test that training checkpoint can be saved."""
        optimizer = torch.optim.AdamW(autoencoder_model.parameters(), lr=0.001)
        
        trainer = Trainer(
            model=autoencoder_model,
            optimizer=optimizer,
            device="cpu",
        )
        
        output_dir = temp_dir / "outputs"
        output_dir.mkdir()
        
        # Test save structure
        trainer.save_training_checkpoint(
            output_dir=output_dir,
            exp_name="test",
            epoch=1,
            best_val_loss=0.5,
            training_history=[],
            is_best=False
        )
        
        # Check that checkpoint directory was created
        checkpoint_dir = output_dir / "checkpoints"
        assert checkpoint_dir.exists()
        
        # Check that checkpoint file exists
        checkpoint_path = checkpoint_dir / "test_checkpoint_latest.pt"
        assert checkpoint_path.exists()
    
    def test_checkpoint_contains_training_state(self, autoencoder_model, temp_dir):
        """Test that training checkpoint contains training state."""
        optimizer = torch.optim.AdamW(autoencoder_model.parameters(), lr=0.001)
        
        trainer = Trainer(
            model=autoencoder_model,
            optimizer=optimizer,
            device="cpu",
        )
        
        output_dir = temp_dir / "outputs"
        output_dir.mkdir()
        
        training_history = [{"epoch": 1, "train_loss": 0.5}]
        
        trainer.save_training_checkpoint(
            output_dir=output_dir,
            exp_name="test",
            epoch=1,
            best_val_loss=0.5,
            training_history=training_history,
            is_best=False
        )
        
        checkpoint_path = output_dir / "checkpoints" / "test_checkpoint_latest.pt"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Should contain training state
        assert "epoch" in checkpoint
        assert "best_val_loss" in checkpoint
        assert "training_history" in checkpoint
        assert "optimizer_state" in checkpoint


class TestResumeFromCheckpoint:
    """Test resuming from checkpoint structure."""
    
    def test_load_training_checkpoint_structure(self, autoencoder_model, temp_dir):
        """Test that training checkpoint can be loaded."""
        optimizer = torch.optim.AdamW(autoencoder_model.parameters(), lr=0.001)
        
        trainer = Trainer(
            model=autoencoder_model,
            optimizer=optimizer,
            device="cpu",
        )
        
        output_dir = temp_dir / "outputs"
        output_dir.mkdir()
        
        # Save checkpoint
        trainer.save_training_checkpoint(
            output_dir=output_dir,
            exp_name="test",
            epoch=1,
            best_val_loss=0.5,
            training_history=[],
            is_best=False
        )
        
        checkpoint_path = output_dir / "checkpoints" / "test_checkpoint_latest.pt"
        
        # Load checkpoint
        extra_state = trainer.load_training_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location="cpu"
        )
        
        assert isinstance(extra_state, dict)
        assert "epoch" in extra_state
        assert "best_val_loss" in extra_state

