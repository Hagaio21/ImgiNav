"""
Consolidated test utilities to eliminate duplication across test files.
Extracted from test_experiments.py, test_components.py, and test_data_pipeline.py
"""
import os
import sys
import tempfile
import shutil
import unittest
import torch
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.autoencoder import AutoEncoder
from models.diffusion import LatentDiffusion
from models.components.unet import DualUNet
from models.components.scheduler import LinearScheduler, CosineScheduler, QuadraticScheduler
from models.losses.custom_loss import StandardVAELoss, DiffusionLoss, VGGPerceptualLoss


class TestUtils:
    """Consolidated utility functions for all test files."""
    
    @staticmethod
    def load_config(config_name):
        """Load a test config by name."""
        config_path = f"tests/configs/{config_name}.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def get_autoencoder_latent_shape(config):
        """Extract latent shape from autoencoder config."""
        latent_channels = config['model']['latent_channels']
        latent_base = config['model']['latent_base']
        return (latent_channels, latent_base, latent_base)
    
    @staticmethod
    def create_test_tensor(shape, device='cpu'):
        """Create a test tensor with given shape."""
        return torch.randn(shape, device=device)
    
    @staticmethod
    def create_mock_layout_embeddings(batch_size, latent_shape):
        """Create mock layout embeddings matching autoencoder latent space."""
        return torch.randn(batch_size, *latent_shape)
    
    @staticmethod
    def create_mock_embeddings_manifest(temp_dir, num_samples=10, latent_shape=(8, 8, 8)):
        """Create a mock manifest with embedding paths for testing."""
        manifest_data = []
        
        for i in range(num_samples):
            # Create mock embedding file
            embedding_path = os.path.join(temp_dir, f"embedding_{i:04d}.pt")
            mock_embedding = torch.randn(*latent_shape)
            torch.save(mock_embedding, embedding_path)
            
            # Add to manifest
            manifest_data.append({
                'image_path': f"test_image_{i:04d}.png",  # Dummy image path
                'embedding_path': embedding_path,
                'scene_id': f"test_scene_{i:04d}",
                'room_id': f"test_room_{i:04d}",
                'pov_id': f"test_pov_{i:04d}"
            })
        
        # Save manifest
        manifest_path = os.path.join(temp_dir, "test_embeddings_manifest.csv")
        manifest_df = pd.DataFrame(manifest_data)
        manifest_df.to_csv(manifest_path, index=False)
        
        return manifest_path, manifest_data
    
    @staticmethod
    def create_mock_image_dataset(temp_dir, num_samples=10, image_size=(64, 64)):
        """Create a mock image dataset for testing."""
        manifest_data = []
        
        for i in range(num_samples):
            # Create mock image file
            image_path = os.path.join(temp_dir, f"image_{i:04d}.png")
            mock_image = torch.randn(3, *image_size) * 255
            mock_image = torch.clamp(mock_image, 0, 255).byte()
            
            # Convert to PIL and save
            from PIL import Image
            pil_image = Image.fromarray(mock_image.permute(1, 2, 0).numpy())
            pil_image.save(image_path)
            
            # Add to manifest
            manifest_data.append({
                'image_path': image_path,
                'scene_id': f"test_scene_{i:04d}",
                'room_id': f"test_room_{i:04d}",
                'pov_id': f"test_pov_{i:04d}"
            })
        
        # Save manifest
        manifest_path = os.path.join(temp_dir, "test_images_manifest.csv")
        manifest_df = pd.DataFrame(manifest_data)
        manifest_df.to_csv(manifest_path, index=False)
        
        return manifest_path, manifest_data
    
    @staticmethod
    def build_autoencoder_from_config(config):
        """Build autoencoder from config."""
        return AutoEncoder.from_shape(
            in_channels=3,
            out_channels=3,
            latent_channels=config['model']['latent_channels'],
            latent_base=config['model']['latent_base'],
            base_channels=config['model']['base_channels'],
            image_size=config['model']['image_size']
        )
    
    @staticmethod
    def build_diffusion_from_config(config, autoencoder=None):
        """Build diffusion model from config."""
        # Load autoencoder if not provided
        if autoencoder is None:
            ae_config = TestUtils.load_config(config['autoencoder']['config'])
            autoencoder = TestUtils.build_autoencoder_from_config(ae_config)
        
        # Get latent shape from autoencoder
        latent_shape = TestUtils.get_autoencoder_latent_shape(ae_config)
        
        # Build UNet
        unet_config = config['model']['unet']
        unet = DualUNet(
            in_channels=unet_config['in_channels'],
            out_channels=unet_config['out_channels'],
            base_channels=unet_config['base_channels'],
            num_layers=unet_config['num_layers'],
            time_embed_dim=unet_config['time_embed_dim']
        )
        
        # Build scheduler
        scheduler_config = config['model']['scheduler']
        scheduler_type = scheduler_config['type']
        if scheduler_type == 'linear':
            scheduler = LinearScheduler(num_steps=scheduler_config['num_steps'])
        elif scheduler_type == 'cosine':
            scheduler = CosineScheduler(num_steps=scheduler_config['num_steps'])
        elif scheduler_type == 'quadratic':
            scheduler = QuadraticScheduler(num_steps=scheduler_config['num_steps'])
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        # Build diffusion model
        diffusion = LatentDiffusion(
            unet=unet,
            autoencoder=autoencoder,
            scheduler=scheduler,
            latent_shape=latent_shape
        )
        
        return diffusion


class TestFixture:
    """Test fixture utilities for common test setups."""
    
    def __init__(self):
        self.temp_dir = None
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        return self.temp_dir
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def __enter__(self):
        return self.setUp()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tearDown()
