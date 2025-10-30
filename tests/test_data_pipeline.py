"""
Data pipeline tests for dataset loading, preprocessing, and data flow.
Tests are latent space agnostic and work with any autoencoder configuration.
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

from models.datasets import LayoutDataset, build_datasets, build_dataloaders
from models.autoencoder import AutoEncoder
import torchvision.transforms as T


from test_utils import TestUtils


class TestDatasetLoading(unittest.TestCase):
    """Test dataset loading with different configurations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    def test_layout_dataset_with_embeddings(self):
        """Test LayoutDataset with embeddings=True."""
        # Create mock embeddings manifest
        latent_shape = (8, 8, 8)  # Example latent shape
        manifest_path, manifest_data = TestUtils.create_mock_embeddings_manifest(
            self.temp_dir, num_samples=5, latent_shape=latent_shape
        )
        
        # Create dataset
        dataset = LayoutDataset(
            manifest_path=manifest_path,
            return_embeddings=True,
            transform=None
        )
        
        # Test dataset properties
        self.assertEqual(len(dataset), 5)
        
        # Test data loading
        sample = dataset[0]
        self.assertIn('layout', sample)
        self.assertIn('scene_id', sample)
        self.assertIn('room_id', sample)
        self.assertIn('pov_id', sample)
        
        # Test embedding shape
        self.assertEqual(sample['layout'].shape, torch.Size(latent_shape))
        self.assertIsInstance(sample['layout'], torch.Tensor)
    
    def test_layout_dataset_without_embeddings(self):
        """Test LayoutDataset with embeddings=False."""
        # Create mock image dataset
        manifest_path, manifest_data = TestUtils.create_mock_image_dataset(
            self.temp_dir, num_samples=5, image_size=(64, 64)
        )
        
        # Create transforms
        transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Create dataset
        dataset = LayoutDataset(
            manifest_path=manifest_path,
            return_embeddings=False,
            transform=transform
        )
        
        # Test dataset properties
        self.assertEqual(len(dataset), 5)
        
        # Test data loading
        sample = dataset[0]
        self.assertIn('layout', sample)
        self.assertIn('scene_id', sample)
        self.assertIn('room_id', sample)
        self.assertIn('pov_id', sample)
        
        # Test image shape
        self.assertEqual(sample['layout'].shape, torch.Size([3, 64, 64]))
        self.assertIsInstance(sample['layout'], torch.Tensor)
    
    def test_dataset_with_autoencoder_config(self):
        """Test dataset loading with autoencoder config to determine latent shape."""
        # Load autoencoder config
        config = TestUtils.load_config('test_AE_large_latent_seg')
        latent_shape = TestUtils.get_autoencoder_latent_shape(config)
        
        # Create mock embeddings with correct latent shape
        manifest_path, manifest_data = TestUtils.create_mock_embeddings_manifest(
            self.temp_dir, num_samples=3, latent_shape=latent_shape
        )
        
        # Create dataset
        dataset = LayoutDataset(
            manifest_path=manifest_path,
            return_embeddings=True,
            transform=None
        )
        
        # Test that embeddings match expected latent shape
        sample = dataset[0]
        self.assertEqual(sample['layout'].shape, torch.Size(latent_shape))
    
    def test_build_datasets_function(self):
        """Test the build_datasets utility function."""
        # Create mock data
        manifest_path, _ = TestUtils.create_mock_image_dataset(
            self.temp_dir, num_samples=10, image_size=(64, 64)
        )
        
        # Test config
        config = {
            'data': {
                'manifest': manifest_path,
                'return_embeddings': False,
                'image_size': 64,
                'split_ratio': 0.8
            }
        }
        
        # Build datasets
        train_dataset, val_dataset = build_datasets(config)
        
        # Test datasets
        self.assertIsInstance(train_dataset, LayoutDataset)
        self.assertIsInstance(val_dataset, LayoutDataset)
        self.assertGreater(len(train_dataset), 0)
        self.assertGreater(len(val_dataset), 0)
        self.assertEqual(len(train_dataset) + len(val_dataset), 10)
    
    def test_build_dataloaders_function(self):
        """Test the build_dataloaders utility function."""
        # Create mock data
        manifest_path, _ = TestUtils.create_mock_image_dataset(
            self.temp_dir, num_samples=10, image_size=(64, 64)
        )
        
        # Test config
        config = {
            'data': {
                'manifest': manifest_path,
                'return_embeddings': False,
                'image_size': 64,
                'split_ratio': 0.8
            },
            'training': {
                'batch_size': 2,
                'num_workers': 0
            }
        }
        
        # Build dataloaders
        train_loader, val_loader = build_dataloaders(config)
        
        # Test dataloaders
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        
        # Test batch loading
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        self.assertIn('layout', train_batch)
        self.assertIn('layout', val_batch)
        self.assertEqual(train_batch['layout'].shape[0], 2)  # batch_size

class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing and transformation pipelines."""
    
    def test_image_transforms(self):
        """Test image transformation pipeline."""
        # Create mock image
        mock_image = torch.randn(3, 128, 128) * 255
        mock_image = torch.clamp(mock_image, 0, 255).byte()
        
        # Define transforms
        transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Apply transforms
        transformed = transform(mock_image.permute(1, 2, 0).numpy())
        
        # Test output
        self.assertEqual(transformed.shape, torch.Size([3, 64, 64]))
        self.assertGreaterEqual(transformed.min(), -1.0)
        self.assertLessEqual(transformed.max(), 1.0)
    
    def test_embedding_loading(self):
        """Test loading of pre-computed embeddings."""
        # Create mock embedding
        latent_shape = (4, 16, 16)
        mock_embedding = torch.randn(*latent_shape)
        
        # Save embedding
        embedding_path = os.path.join(tempfile.mkdtemp(), "test_embedding.pt")
        torch.save(mock_embedding, embedding_path)
        
        # Load embedding
        loaded_embedding = torch.load(embedding_path)
        
        # Test
        self.assertEqual(loaded_embedding.shape, torch.Size(latent_shape))
        self.assertTrue(torch.allclose(loaded_embedding, mock_embedding))
        
        # Cleanup
        os.remove(embedding_path)

class TestDataFlow(unittest.TestCase):
    """Test data flow through the pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    def test_embeddings_to_model_flow(self):
        """Test data flow from embeddings to model input."""
        # Create mock embeddings
        latent_shape = (8, 8, 8)
        manifest_path, _ = TestUtils.create_mock_embeddings_manifest(
            self.temp_dir, num_samples=3, latent_shape=latent_shape
        )
        
        # Create dataset
        dataset = LayoutDataset(
            manifest_path=manifest_path,
            return_embeddings=True,
            transform=None
        )
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=False, num_workers=0
        )
        
        # Test batch loading
        batch = next(iter(dataloader))
        
        # Test batch structure
        self.assertIn('layout', batch)
        self.assertEqual(batch['layout'].shape, torch.Size([2] + list(latent_shape)))
        
        # Test that data is ready for model input
        model_input = {"layout": batch['layout']}
        self.assertIsInstance(model_input['layout'], torch.Tensor)
        self.assertEqual(model_input['layout'].device.type, 'cpu')
    
    def test_images_to_embeddings_flow(self):
        """Test data flow from images to embeddings (via autoencoder)."""
        # Create mock image dataset
        manifest_path, _ = TestUtils.create_mock_image_dataset(
            self.temp_dir, num_samples=3, image_size=(64, 64)
        )
        
        # Load autoencoder config
        config = TestUtils.load_config('test_AE_small_latent')
        latent_shape = TestUtils.get_autoencoder_latent_shape(config)
        
        # Create autoencoder
        autoencoder = AutoEncoder.from_shape(
            in_channels=3,
            out_channels=3,
            latent_channels=config['model']['latent_channels'],
            latent_base=config['model']['latent_base'],
            base_channels=config['model']['base_channels'],
            image_size=config['model']['image_size']
        )
        
        # Create dataset
        transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        dataset = LayoutDataset(
            manifest_path=manifest_path,
            return_embeddings=False,
            transform=transform
        )
        
        # Test encoding
        sample = dataset[0]
        image = sample['layout'].unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            encoded = autoencoder.encode(image)
        
        # Test encoded shape matches expected latent shape
        self.assertEqual(encoded.shape, torch.Size([1] + list(latent_shape)))

if __name__ == '__main__':
    unittest.main()
