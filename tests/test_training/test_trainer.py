"""
Tests for Trainer class - focusing on structure and initialization.
"""
import pytest
import torch
from pathlib import Path

from training.engine import Trainer
from models.autoencoder import Autoencoder


class TestTrainerInitialization:
    """Test Trainer initialization."""
    
    def test_trainer_init(self, autoencoder_model):
        """Test Trainer can be initialized."""
        optimizer = torch.optim.AdamW(autoencoder_model.parameters(), lr=0.001)
        
        trainer = Trainer(
            model=autoencoder_model,
            optimizer=optimizer,
            device="cpu",
            use_amp=False,
        )
        
        assert trainer is not None
        assert trainer.model == autoencoder_model
        assert trainer.optimizer == optimizer
    
    def test_trainer_with_scheduler(self, autoencoder_model):
        """Test Trainer with learning rate scheduler."""
        optimizer = torch.optim.AdamW(autoencoder_model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        trainer = Trainer(
            model=autoencoder_model,
            optimizer=optimizer,
            scheduler=scheduler,
            device="cpu",
            use_amp=False,
        )
        
        assert trainer.scheduler == scheduler
    
    def test_trainer_amp_disabled(self, autoencoder_model):
        """Test Trainer with AMP disabled."""
        optimizer = torch.optim.AdamW(autoencoder_model.parameters(), lr=0.001)
        
        trainer = Trainer(
            model=autoencoder_model,
            optimizer=optimizer,
            device="cpu",
            use_amp=False,
        )
        
        assert trainer.use_amp == False
        assert trainer.scaler is None
    
    def test_trainer_gradient_accumulation(self, autoencoder_model):
        """Test Trainer with gradient accumulation."""
        optimizer = torch.optim.AdamW(autoencoder_model.parameters(), lr=0.001)
        
        trainer = Trainer(
            model=autoencoder_model,
            optimizer=optimizer,
            device="cpu",
            gradient_accumulation_steps=4,
        )
        
        assert trainer.gradient_accumulation_steps == 4
    
    def test_trainer_gradient_clipping(self, autoencoder_model):
        """Test Trainer with gradient clipping."""
        optimizer = torch.optim.AdamW(autoencoder_model.parameters(), lr=0.001)
        
        trainer = Trainer(
            model=autoencoder_model,
            optimizer=optimizer,
            device="cpu",
            max_grad_norm=1.0,
        )
        
        assert trainer.max_grad_norm == 1.0


class TestTrainerStructure:
    """Test Trainer structure and attributes."""
    
    def test_trainer_has_required_attributes(self, autoencoder_model):
        """Test Trainer has required attributes."""
        optimizer = torch.optim.AdamW(autoencoder_model.parameters(), lr=0.001)
        
        trainer = Trainer(
            model=autoencoder_model,
            optimizer=optimizer,
            device="cpu",
        )
        
        assert hasattr(trainer, 'model')
        assert hasattr(trainer, 'optimizer')
        assert hasattr(trainer, 'device')
        assert hasattr(trainer, 'use_amp')
        assert hasattr(trainer, 'gradient_accumulation_steps')
    
    def test_trainer_device(self, autoencoder_model):
        """Test Trainer device setting."""
        optimizer = torch.optim.AdamW(autoencoder_model.parameters(), lr=0.001)
        
        trainer = Trainer(
            model=autoencoder_model,
            optimizer=optimizer,
            device="cpu",
        )
        
        assert trainer.device.type == "cpu"

