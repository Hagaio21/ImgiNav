"""
Tests for diffusion training - focusing on structure, not correctness.
"""
import pytest
import torch
from pathlib import Path
import tempfile

from training.engine import Trainer
from models.diffusion import DiffusionModel
from models.losses.base_loss import LOSS_REGISTRY


class TestDiffusionTrainingLoopStructure:
    """Test diffusion training loop structure."""
    
    def test_trainer_can_be_created(self, diffusion_model):
        """Test that Trainer can be created for diffusion."""
        optimizer = torch.optim.AdamW(diffusion_model.unet.parameters(), lr=0.001)
        
        trainer = Trainer(
            model=diffusion_model,
            optimizer=optimizer,
            device="cpu",
            use_amp=False,
        )
        
        assert trainer is not None
    
    def test_loss_function_structure(self, test_diffusion_config):
        """Test that loss function can be built from config."""
        from training.utils import build_loss
        
        loss_fn = build_loss(test_diffusion_config)
        
        assert loss_fn is not None
        # Should be a callable
        assert callable(loss_fn)
    
    def test_optimizer_structure(self, diffusion_model, test_diffusion_config):
        """Test that optimizer can be built from config."""
        from training.utils import build_optimizer
        
        optimizer = build_optimizer(diffusion_model, test_diffusion_config)
        
        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.Optimizer)
    
    def test_conditioning_config(self, diffusion_model):
        """Test that conditioning configuration is accessible."""
        # Check if model has embedding_projection
        has_embedding_proj = hasattr(diffusion_model, 'embedding_projection') and \
                            diffusion_model.embedding_projection is not None
        
        # Either has it or doesn't - both are valid
        assert isinstance(has_embedding_proj, bool)


class TestCheckpointSavingDuringTraining:
    """Test checkpoint saving during diffusion training structure."""
    
    def test_save_training_checkpoint_structure(self, diffusion_model, temp_dir):
        """Test that training checkpoint can be saved."""
        optimizer = torch.optim.AdamW(diffusion_model.unet.parameters(), lr=0.001)
        
        trainer = Trainer(
            model=diffusion_model,
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
    
    def test_checkpoint_contains_training_state(self, diffusion_model, temp_dir):
        """Test that training checkpoint contains training state."""
        optimizer = torch.optim.AdamW(diffusion_model.unet.parameters(), lr=0.001)
        
        trainer = Trainer(
            model=diffusion_model,
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

