"""
Experiment orchestration tests for end-to-end training workflows.
Tests are latent space agnostic and work with any autoencoder configuration.
"""
import os
import sys
import tempfile
import shutil
import unittest
import torch
import torch.nn as nn
import yaml
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.autoencoder import AutoEncoder
from models.diffusion import LatentDiffusion
from models.components.scheduler import LinearScheduler, CosineScheduler, QuadraticScheduler
from models.components.unet import DualUNet
from models.losses.custom_loss import StandardVAELoss, DiffusionLoss
from models.datasets import LayoutDataset, build_datasets, build_dataloaders
from training.trainer import Trainer
import torchvision.transforms as T


from test_utils import TestUtils


class TestConfigLoading(unittest.TestCase):
    """Test that all configs can be loaded and parsed correctly."""
    
    def test_all_configs_loadable(self):
        """Test that all configs can be loaded and parsed."""
        # Autoencoder configs
        ae_configs = [
            'test_AE_dropout', 'test_AE_large_latent_seg', 'test_AE_small_latent',
            'test_VAE_large_KL_seg', 'test_VAE_med_KL', 'test_VAE_small_KL_seg', 'test_VAE_stable'
        ]
        
        # Diffusion configs
        diffusion_configs = [
            'test_E1_Linear_64', 'test_E2_Cosine_64', 'test_E3_Quadratic_64',
            'test_E4_Linear_128', 'test_E5_Cosine_128', 'test_E6_Quadratic_128',
            'test_E7_Cosine_128_VGG', 'test_E8_Cosine_128_VGG_High'
        ]
        
        all_configs = ae_configs + diffusion_configs
        
        for config_name in all_configs:
            with self.subTest(config=config_name):
                config = TestUtils.load_config(config_name)
                self.assertIsInstance(config, dict)
                self.assertIn('model', config)
    
    def test_autoencoder_configs_structure(self):
        """Test autoencoder config structure."""
        ae_configs = [
            'test_AE_dropout', 'test_AE_large_latent_seg', 'test_AE_small_latent',
            'test_VAE_large_KL_seg', 'test_VAE_med_KL', 'test_VAE_small_KL_seg', 'test_VAE_stable'
        ]
        
        for config_name in ae_configs:
            with self.subTest(config=config_name):
                config = TestUtils.load_config(config_name)
                
                # Check required fields
                self.assertIn('model', config)
                self.assertIn('latent_channels', config['model'])
                self.assertIn('latent_base', config['model'])
                self.assertIn('base_channels', config['model'])
                self.assertIn('image_size', config['model'])
                
                # Check latent shape can be extracted
                latent_shape = TestUtils.get_autoencoder_latent_shape(config)
                self.assertIsInstance(latent_shape, tuple)
                self.assertEqual(len(latent_shape), 3)
    
    def test_diffusion_configs_structure(self):
        """Test diffusion config structure."""
        diffusion_configs = [
            'test_E1_Linear_64', 'test_E2_Cosine_64', 'test_E3_Quadratic_64',
            'test_E4_Linear_128', 'test_E5_Cosine_128', 'test_E6_Quadratic_128',
            'test_E7_Cosine_128_VGG', 'test_E8_Cosine_128_VGG_High'
        ]
        
        for config_name in diffusion_configs:
            with self.subTest(config=config_name):
                config = TestUtils.load_config(config_name)
                
                # Check required fields
                self.assertIn('model', config)
                self.assertIn('unet', config['model'])
                self.assertIn('scheduler', config['model'])
                self.assertIn('autoencoder', config)
                self.assertIn('config', config['autoencoder'])


class TestModelBuilding(unittest.TestCase):
    """Test model building from configs."""
    
    def test_autoencoder_model_building(self):
        """Test building all autoencoder models."""
        ae_configs = [
            'test_AE_dropout', 'test_AE_large_latent_seg', 'test_AE_small_latent',
            'test_VAE_large_KL_seg', 'test_VAE_med_KL', 'test_VAE_small_KL_seg', 'test_VAE_stable'
        ]
        
        for config_name in ae_configs:
            with self.subTest(config=config_name):
                config = TestUtils.load_config(config_name)
                model = TestUtils.build_autoencoder_from_config(config)
                
                # Test model structure
                self.assertIsInstance(model, AutoEncoder)
                self.assertIsNotNone(model.encoder)
                self.assertIsNotNone(model.decoder)
                
                # Test forward pass
                latent_shape = TestUtils.get_autoencoder_latent_shape(config)
                input_tensor = torch.randn(2, 3, 64, 64)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    encoded = model.encode(input_tensor)
                
                # Test output shapes
                self.assertEqual(output.shape, input_tensor.shape)
                self.assertEqual(encoded.shape, torch.Size([2] + list(latent_shape)))
    
    def test_diffusion_model_building(self):
        """Test building all diffusion models."""
        diffusion_configs = [
            'test_E1_Linear_64', 'test_E2_Cosine_64', 'test_E3_Quadratic_64',
            'test_E4_Linear_128', 'test_E5_Cosine_128', 'test_E6_Quadratic_128',
            'test_E7_Cosine_128_VGG', 'test_E8_Cosine_128_VGG_High'
        ]
        
        for config_name in diffusion_configs:
            with self.subTest(config=config_name):
                config = TestUtils.load_config(config_name)
                diffusion = TestUtils.build_diffusion_from_config(config)
                
                # Test model structure
                self.assertIsInstance(diffusion, LatentDiffusion)
                self.assertIsNotNone(diffusion.unet)
                self.assertIsNotNone(diffusion.autoencoder)
                self.assertIsNotNone(diffusion.scheduler)
                
                # Test forward pass
                batch = {"layout": torch.randn(2, 8, 8, 8)}
                output = diffusion(batch)
                self.assertIn('pred_noise', output)
                self.assertEqual(output['pred_noise'].shape, (2, 8, 8, 8))
    
    def test_embeddings_true_false_configs(self):
        """Test both embeddings true and false configurations."""
        # Test with embeddings=True
        config_with_emb = TestUtils.load_config('test_E1_Linear_64')
        config_with_emb['data']['return_embeddings'] = True
        
        # Test with embeddings=False
        config_without_emb = TestUtils.load_config('test_E1_Linear_64')
        config_without_emb['data']['return_embeddings'] = False
        
        # Both should build successfully
        diffusion_with_emb = TestUtils.build_diffusion_from_config(config_with_emb)
        diffusion_without_emb = TestUtils.build_diffusion_from_config(config_without_emb)
        
        self.assertIsInstance(diffusion_with_emb, LatentDiffusion)
        self.assertIsInstance(diffusion_without_emb, LatentDiffusion)


class TestDatasetLoading(unittest.TestCase):
    """Test dataset loading with different configurations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    def test_autoencoder_dataset_loading(self):
        """Test dataset loading for autoencoder configs."""
        # Create mock image dataset
        manifest_path, _ = TestUtils.create_mock_embeddings_manifest(
            self.temp_dir, num_samples=10, latent_shape=(3, 64, 64)  # RGB images
        )
        
        # Load autoencoder config
        config = TestUtils.load_config('test_AE_dropout')
        config['data']['manifest'] = manifest_path
        config['data']['return_embeddings'] = False
        
        # Build datasets
        train_dataset, val_dataset = build_datasets(config)
        
        # Test datasets
        self.assertIsInstance(train_dataset, LayoutDataset)
        self.assertIsInstance(val_dataset, LayoutDataset)
        self.assertGreater(len(train_dataset), 0)
        self.assertGreater(len(val_dataset), 0)
    
    def test_diffusion_dataset_loading_with_embeddings(self):
        """Test dataset loading for diffusion configs with embeddings."""
        # Load autoencoder config to get latent shape
        ae_config = TestUtils.load_config('test_AE_large_latent_seg')
        latent_shape = TestUtils.get_autoencoder_latent_shape(ae_config)
        
        # Create mock embeddings manifest
        manifest_path, _ = TestUtils.create_mock_embeddings_manifest(
            self.temp_dir, num_samples=10, latent_shape=latent_shape
        )
        
        # Load diffusion config
        config = TestUtils.load_config('test_E1_Linear_64')
        config['data']['manifest'] = manifest_path
        config['data']['return_embeddings'] = True
        
        # Build datasets
        train_dataset, val_dataset = build_datasets(config)
        
        # Test datasets
        self.assertIsInstance(train_dataset, LayoutDataset)
        self.assertIsInstance(val_dataset, LayoutDataset)
        
        # Test sample loading
        sample = train_dataset[0]
        self.assertIn('layout', sample)
        self.assertEqual(sample['layout'].shape, torch.Size(latent_shape))
    
    def test_diffusion_dataset_loading_without_embeddings(self):
        """Test dataset loading for diffusion configs without embeddings."""
        # Create mock image dataset
        manifest_path, _ = TestUtils.create_mock_embeddings_manifest(
            self.temp_dir, num_samples=10, latent_shape=(3, 64, 64)  # RGB images
        )
        
        # Load diffusion config
        config = TestUtils.load_config('test_E1_Linear_64')
        config['data']['manifest'] = manifest_path
        config['data']['return_embeddings'] = False
        
        # Build datasets
        train_dataset, val_dataset = build_datasets(config)
        
        # Test datasets
        self.assertIsInstance(train_dataset, LayoutDataset)
        self.assertIsInstance(val_dataset, LayoutDataset)
        
        # Test sample loading
        sample = train_dataset[0]
        self.assertIn('layout', sample)
        self.assertEqual(sample['layout'].shape, torch.Size([3, 64, 64]))


class TestEndToEndTraining(unittest.TestCase):
    """Test end-to-end training workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    def test_autoencoder_end_to_end_training(self):
        """Test end-to-end autoencoder training."""
        # Load config
        config = TestUtils.load_config('test_AE_dropout')
        
        # Create mock data
        latent_shape = TestUtils.get_autoencoder_latent_shape(config)
        manifest_path, _ = TestUtils.create_mock_embeddings_manifest(
            self.temp_dir, num_samples=20, latent_shape=(3, 64, 64)  # RGB images
        )
        
        # Update config for test data
        config['data']['manifest'] = manifest_path
        config['data']['return_embeddings'] = False
        config['training']['epochs'] = 2
        config['training']['batch_size'] = 4
        
        # Build model
        model = TestUtils.build_autoencoder_from_config(config)
        
        # Build datasets
        train_dataset, val_dataset = build_datasets(config)
        train_loader, val_loader = build_dataloaders(config)
        
        # Test training step
        model.train()
        train_batch = next(iter(train_loader))
        
        # Forward pass
        output = model(train_batch['layout'])
        self.assertEqual(output.shape, train_batch['layout'].shape)
        
        # Loss computation
        loss_fn = StandardVAELoss()
        loss = loss_fn(train_batch['layout'], output, 
                      torch.zeros_like(train_batch['layout']), 
                      torch.zeros_like(train_batch['layout']))
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0)
    
    def test_diffusion_end_to_end_training(self):
        """Test end-to-end diffusion training with sampling."""
        # Load config
        config = TestUtils.load_config('test_E2_Cosine_64')
        
        # Load autoencoder config to get latent shape
        ae_config = TestUtils.load_config(config['autoencoder']['config'])
        latent_shape = TestUtils.get_autoencoder_latent_shape(ae_config)
        
        # Create mock embeddings data
        manifest_path, _ = TestUtils.create_mock_embeddings_manifest(
            self.temp_dir, num_samples=20, latent_shape=latent_shape
        )
        
        # Update config for test data
        config['data']['manifest'] = manifest_path
        config['data']['return_embeddings'] = True
        config['training']['epochs'] = 2
        config['training']['batch_size'] = 4
        
        # Build models
        autoencoder = TestUtils.build_autoencoder_from_config(ae_config)
        diffusion = TestUtils.build_diffusion_from_config(config, autoencoder)
        
        # Build datasets
        train_dataset, val_dataset = build_datasets(config)
        train_loader, val_loader = build_dataloaders(config)
        
        # Test training step
        diffusion.train()
        train_batch = next(iter(train_loader))
        
        # Forward pass
        output = diffusion(train_batch)
        self.assertIn('pred_noise', output)
        self.assertEqual(output['pred_noise'].shape, 
                        torch.Size([config['training']['batch_size']] + list(latent_shape)))
        
        # Loss computation
        loss_fn = DiffusionLoss()
        loss = loss_fn(output['pred_noise'], torch.randn_like(output['pred_noise']))
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0)
        
        # Test sampling
        diffusion.eval()
        with torch.no_grad():
            # Test DDPM sampling
            samples = diffusion.sample_ddpm(batch_size=2, num_steps=10)
            self.assertEqual(samples.shape, torch.Size([2] + list(latent_shape)))
            
            # Test DDIM sampling
            samples = diffusion.sample_ddim(batch_size=2, num_steps=10)
            self.assertEqual(samples.shape, torch.Size([2] + list(latent_shape)))


class TestExperimentOrchestration(unittest.TestCase):
    """Test experiment orchestration and workflow management."""
    
    def test_config_validation(self):
        """Test that configs are properly validated before use."""
        # Test valid configs
        valid_configs = [
            'test_AE_dropout', 'test_AE_large_latent_seg', 'test_E1_Linear_64', 'test_E2_Cosine_64'
        ]
        
        for config_name in valid_configs:
            with self.subTest(config=config_name):
                config = TestUtils.load_config(config_name)
                
                # Validate required fields exist
                if 'latent_channels' in config.get('model', {}):
                    # Autoencoder config
                    self.assertIn('latent_channels', config['model'])
                    self.assertIn('latent_base', config['model'])
                    self.assertIn('base_channels', config['model'])
                else:
                    # Diffusion config
                    self.assertIn('unet', config['model'])
                    self.assertIn('scheduler', config['model'])
                    self.assertIn('autoencoder', config)
    
    def test_latent_space_compatibility(self):
        """Test that autoencoder and diffusion models have compatible latent spaces."""
        # Test different autoencoder-diffusion pairs
        test_pairs = [
            ('test_AE_large_latent_seg', 'test_E1_Linear_64'),
            ('test_AE_small_latent', 'test_E2_Cosine_64'),
            ('test_VAE_med_KL', 'test_E3_Quadratic_64'),
        ]
        
        for ae_config_name, diff_config_name in test_pairs:
            with self.subTest(pair=(ae_config_name, diff_config_name)):
                # Load configs
                ae_config = TestUtils.load_config(ae_config_name)
                diff_config = TestUtils.load_config(diff_config_name)
                
                # Get latent shapes
                ae_latent_shape = TestUtils.get_autoencoder_latent_shape(ae_config)
                diff_latent_shape = (diff_config['model']['unet']['in_channels'], 
                                   ae_latent_shape[1], ae_latent_shape[2])
                
                # Test compatibility
                self.assertEqual(ae_latent_shape[0], diff_latent_shape[0])
                self.assertEqual(ae_latent_shape[1], diff_latent_shape[1])
                self.assertEqual(ae_latent_shape[2], diff_latent_shape[2])
    
    def test_experiment_workflow(self):
        """Test complete experiment workflow from config to training."""
        # Load autoencoder config
        ae_config = TestUtils.load_config('test_AE_dropout')
        
        # Load diffusion config
        diff_config = TestUtils.load_config('test_E1_Linear_64')
        
        # Build models
        autoencoder = TestUtils.build_autoencoder_from_config(ae_config)
        diffusion = TestUtils.build_diffusion_from_config(diff_config, autoencoder)
        
        # Test that models are compatible
        self.assertIsNotNone(autoencoder)
        self.assertIsNotNone(diffusion)
        
        # Test that diffusion can use autoencoder
        test_input = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            encoded = autoencoder.encode(test_input)
            batch = {"layout": encoded}
            output = diffusion(batch)
        
        self.assertIn('pred_noise', output)
        self.assertEqual(output['pred_noise'].shape, encoded.shape)


if __name__ == '__main__':
    unittest.main()