"""
Component tests for individual model components.
Tests are architecture agnostic and work with any configuration.
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

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.autoencoder import AutoEncoder, ConvEncoder, ConvDecoder
from models.components.unet import DualUNet, DownBlock, UpBlock, ResidualBlock
from models.components.scheduler import LinearScheduler, CosineScheduler, QuadraticScheduler
from models.losses.custom_loss import StandardVAELoss, DiffusionLoss, VGGPerceptualLoss


from test_utils import TestUtils


class TestAutoEncoderComponents(unittest.TestCase):
    """Test AutoEncoder and its components."""
    
    def test_conv_encoder_creation(self):
        """Test ConvEncoder creation with various configurations."""
        # Test basic encoder
        encoder = ConvEncoder(
            in_channels=3,
            latent_channels=8,
            latent_base=8,
            base_channels=32,
            image_size=64
        )
        
        self.assertIsInstance(encoder, nn.Module)
        self.assertEqual(encoder.in_channels, 3)
        self.assertEqual(encoder.latent_channels, 8)
    
    def test_conv_decoder_creation(self):
        """Test ConvDecoder creation with various configurations."""
        # Test basic decoder
        decoder = ConvDecoder(
            out_channels=3,
            latent_channels=8,
            latent_base=8,
            base_channels=32,
            image_size=64
        )
        
        self.assertIsInstance(decoder, nn.Module)
        self.assertEqual(decoder.out_channels, 3)
        self.assertEqual(decoder.latent_channels, 8)
    
    def test_autoencoder_from_shape(self):
        """Test AutoEncoder creation using from_shape factory method."""
        # Test with different latent configurations
        configs = [
            {'latent_channels': 4, 'latent_base': 8, 'base_channels': 16},
            {'latent_channels': 8, 'latent_base': 16, 'base_channels': 32},
            {'latent_channels': 16, 'latent_base': 4, 'base_channels': 64},
        ]
        
        for config in configs:
            with self.subTest(config=config):
                autoencoder = AutoEncoder.from_shape(
                    in_channels=3,
                    out_channels=3,
                    latent_channels=config['latent_channels'],
                    latent_base=config['latent_base'],
                    base_channels=config['base_channels'],
                    image_size=64
                )
                
                self.assertIsInstance(autoencoder, AutoEncoder)
                self.assertIsInstance(autoencoder.encoder, ConvEncoder)
                self.assertIsInstance(autoencoder.decoder, ConvDecoder)
    
    def test_autoencoder_forward_pass(self):
        """Test AutoEncoder forward pass with different input sizes."""
        # Test configurations
        test_configs = [
            {'latent_channels': 4, 'latent_base': 8, 'base_channels': 16, 'image_size': 64},
            {'latent_channels': 8, 'latent_base': 16, 'base_channels': 32, 'image_size': 128},
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                autoencoder = AutoEncoder.from_shape(
                    in_channels=3,
                    out_channels=3,
                    **config
                )
                
                # Test forward pass
                batch_size = 2
                input_tensor = TestUtils.create_test_tensor(
                    (batch_size, 3, config['image_size'], config['image_size'])
                )
                
                with torch.no_grad():
                    output = autoencoder(input_tensor)
                
                # Test output shape
                expected_shape = (batch_size, 3, config['image_size'], config['image_size'])
                self.assertEqual(output.shape, torch.Size(expected_shape))
    
    def test_autoencoder_encode_decode(self):
        """Test AutoEncoder encode and decode methods."""
        autoencoder = AutoEncoder.from_shape(
            in_channels=3,
            out_channels=3,
            latent_channels=8,
            latent_base=8,
            base_channels=32,
            image_size=64
        )
        
        # Test input
        input_tensor = TestUtils.create_test_tensor((1, 3, 64, 64))
        
        with torch.no_grad():
            # Test encoding
            encoded = autoencoder.encode(input_tensor)
            expected_latent_shape = (1, 8, 8, 8)
            self.assertEqual(encoded.shape, torch.Size(expected_latent_shape))
            
            # Test decoding
            decoded = autoencoder.decode(encoded)
            self.assertEqual(decoded.shape, input_tensor.shape)
    
    def test_autoencoder_with_config(self):
        """Test AutoEncoder creation from config file."""
        config = TestUtils.load_config('test_AE_large_latent_seg')
        
        autoencoder = AutoEncoder.from_shape(
            in_channels=3,
            out_channels=3,
            latent_channels=config['model']['latent_channels'],
            latent_base=config['model']['latent_base'],
            base_channels=config['model']['base_channels'],
            image_size=config['model']['image_size']
        )
        
        # Test with config-specified latent shape
        latent_shape = TestUtils.get_autoencoder_latent_shape(config)
        input_tensor = TestUtils.create_test_tensor((1, 3, 64, 64))
        
        with torch.no_grad():
            encoded = autoencoder.encode(input_tensor)
            self.assertEqual(encoded.shape, torch.Size([1] + list(latent_shape)))


class TestUNetComponents(unittest.TestCase):
    """Test UNet and its components."""
    
    def test_residual_block_creation(self):
        """Test ResidualBlock creation."""
        # Test with different channel configurations
        test_configs = [
            {'in_channels': 32, 'out_channels': 64},
            {'in_channels': 64, 'out_channels': 128},
            {'in_channels': 128, 'out_channels': 64},
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                # Ensure channels are divisible by 8 for GroupNorm
                in_ch = ((config['in_channels'] + 7) // 8) * 8
                out_ch = ((config['out_channels'] + 7) // 8) * 8
                
                block = ResidualBlock(in_ch, out_ch)
                self.assertIsInstance(block, nn.Module)
    
    def test_down_block_creation(self):
        """Test DownBlock creation."""
        # Test with different configurations
        test_configs = [
            {'in_channels': 32, 'out_channels': 64, 'downsample': True},
            {'in_channels': 64, 'out_channels': 128, 'downsample': False},
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                # Ensure channels are divisible by 8
                in_ch = ((config['in_channels'] + 7) // 8) * 8
                out_ch = ((config['out_channels'] + 7) // 8) * 8
                
                block = DownBlock(in_ch, out_ch, downsample=config['downsample'])
                self.assertIsInstance(block, nn.Module)
    
    def test_up_block_creation(self):
        """Test UpBlock creation."""
        # Test with different configurations
        test_configs = [
            {'in_channels': 64, 'out_channels': 32, 'upsample': True},
            {'in_channels': 128, 'out_channels': 64, 'upsample': False},
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                # Ensure channels are divisible by 8
                in_ch = ((config['in_channels'] + 7) // 8) * 8
                out_ch = ((config['out_channels'] + 7) // 8) * 8
                
                block = UpBlock(in_ch, out_ch, upsample=config['upsample'])
                self.assertIsInstance(block, nn.Module)
    
    def test_dual_unet_creation(self):
        """Test DualUNet creation with various configurations."""
        # Test configurations that work with GroupNorm (channels divisible by 8)
        test_configs = [
            {'in_channels': 8, 'out_channels': 8, 'base_channels': 32},
            {'in_channels': 16, 'out_channels': 16, 'base_channels': 64},
            {'in_channels': 32, 'out_channels': 32, 'base_channels': 128},
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                unet = DualUNet(
                    in_channels=config['in_channels'],
                    out_channels=config['out_channels'],
                    base_channels=config['base_channels'],
                    num_layers=3,  # Small for testing
                    time_embed_dim=128
                )
                
                self.assertIsInstance(unet, nn.Module)
    
    def test_dual_unet_forward_pass(self):
        """Test DualUNet forward pass with different input shapes."""
        unet = DualUNet(
            in_channels=8,
            out_channels=8,
            base_channels=32,
            num_layers=2,
            time_embed_dim=128
        )
        
        # Test with different input shapes
        test_shapes = [
            (2, 8, 8, 8),
            (1, 8, 16, 16),
            (4, 8, 4, 4),
        ]
        
        for shape in test_shapes:
            with self.subTest(shape=shape):
                input_tensor = TestUtils.create_test_tensor(shape)
                timesteps = torch.randint(0, 1000, (shape[0],))
                
                with torch.no_grad():
                    output = unet(input_tensor, timesteps)
                
                # Output should have same shape as input
                self.assertEqual(output.shape, input_tensor.shape)
    
    def test_dual_unet_with_conditioning(self):
        """Test DualUNet with conditioning input."""
        unet = DualUNet(
            in_channels=8,
            out_channels=8,
            base_channels=32,
            num_layers=2,
            time_embed_dim=128,
            cond_channels=16  # Add conditioning
        )
        
        # Test input with conditioning
        input_tensor = TestUtils.create_test_tensor((2, 8, 8, 8))
        cond_tensor = TestUtils.create_test_tensor((2, 16, 8, 8))
        timesteps = torch.randint(0, 1000, (2,))
        
        with torch.no_grad():
            output = unet(input_tensor, timesteps, cond_tensor)
        
        self.assertEqual(output.shape, input_tensor.shape)


class TestSchedulerComponents(unittest.TestCase):
    """Test noise scheduler components."""
    
    def test_linear_scheduler_creation(self):
        """Test LinearScheduler creation."""
        scheduler = LinearScheduler(num_steps=1000)
        self.assertIsInstance(scheduler, nn.Module)
    
    def test_cosine_scheduler_creation(self):
        """Test CosineScheduler creation."""
        scheduler = CosineScheduler(num_steps=1000)
        self.assertIsInstance(scheduler, nn.Module)
    
    def test_quadratic_scheduler_creation(self):
        """Test QuadraticScheduler creation."""
        scheduler = QuadraticScheduler(num_steps=1000)
        self.assertIsInstance(scheduler, nn.Module)
    
    def test_scheduler_forward_pass(self):
        """Test scheduler forward pass."""
        schedulers = [
            LinearScheduler(num_steps=1000),
            CosineScheduler(num_steps=1000),
            QuadraticScheduler(num_steps=1000),
        ]
        
        for scheduler in schedulers:
            with self.subTest(scheduler=type(scheduler).__name__):
                # Test with different timesteps
                timesteps = torch.tensor([0, 100, 500, 999])
                
                with torch.no_grad():
                    noise_schedule = scheduler(timesteps)
                
                # Test output shape and range
                self.assertEqual(noise_schedule.shape, timesteps.shape)
                self.assertTrue(torch.all(noise_schedule >= 0))
                self.assertTrue(torch.all(noise_schedule <= 1))
    
    def test_scheduler_sampling(self):
        """Test scheduler sampling methods."""
        scheduler = LinearScheduler(num_steps=1000)
        
        # Test sample_timesteps
        batch_size = 4
        timesteps = scheduler.sample_timesteps(batch_size)
        
        self.assertEqual(timesteps.shape, (batch_size,))
        self.assertTrue(torch.all(timesteps >= 0))
        self.assertTrue(torch.all(timesteps < 1000))
        
        # Test add_noise
        x = TestUtils.create_test_tensor((batch_size, 8, 8, 8))
        noise = TestUtils.create_test_tensor((batch_size, 8, 8, 8))
        
        noisy_x = scheduler.add_noise(x, noise, timesteps)
        self.assertEqual(noisy_x.shape, x.shape)

class TestLossComponents(unittest.TestCase):
    """Test loss function components."""
    
    def test_vae_loss_creation(self):
        """Test VAE loss creation."""
        loss_fn = StandardVAELoss()
        self.assertIsInstance(loss_fn, nn.Module)
    
    def test_diffusion_loss_creation(self):
        """Test DiffusionLoss creation."""
        loss_fn = DiffusionLoss()
        self.assertIsInstance(loss_fn, nn.Module)
    
    def test_vgg_perceptual_loss_creation(self):
        """Test VGGPerceptualLoss creation."""
        loss_fn = VGGPerceptualLoss()
        self.assertIsInstance(loss_fn, nn.Module)
    
    def test_vae_loss_computation(self):
        """Test VAE loss computation."""
        loss_fn = StandardVAELoss()
        
        # Test with different input shapes
        test_shapes = [
            (2, 3, 64, 64),
            (1, 3, 128, 128),
            (4, 3, 32, 32),
        ]
        
        for shape in test_shapes:
            with self.subTest(shape=shape):
                x = TestUtils.create_test_tensor(shape)
                x_recon = TestUtils.create_test_tensor(shape)
                mu = TestUtils.create_test_tensor(shape)
                logvar = TestUtils.create_test_tensor(shape)
                
                loss = loss_fn(x, x_recon, mu, logvar)
                
                self.assertIsInstance(loss, torch.Tensor)
                self.assertEqual(loss.shape, ())
                self.assertGreaterEqual(loss.item(), 0)
    
    def test_diffusion_loss_computation(self):
        """Test DiffusionLoss computation."""
        loss_fn = DiffusionLoss()
        
        # Test with different input shapes
        test_shapes = [
            (2, 8, 8, 8),
            (1, 16, 16, 16),
            (4, 4, 4, 4),
        ]
        
        for shape in test_shapes:
            with self.subTest(shape=shape):
                pred_noise = TestUtils.create_test_tensor(shape)
                target_noise = TestUtils.create_test_tensor(shape)
                
                loss = loss_fn(pred_noise, target_noise)
                
                self.assertIsInstance(loss, torch.Tensor)
                self.assertEqual(loss.shape, ())
                self.assertGreaterEqual(loss.item(), 0)
    
    def test_vgg_perceptual_loss_computation(self):
        """Test VGGPerceptualLoss computation."""
        loss_fn = VGGPerceptualLoss()
        
        # Test with different input shapes
        test_shapes = [
            (2, 3, 64, 64),
            (1, 3, 128, 128),
        ]
        
        for shape in test_shapes:
            with self.subTest(shape=shape):
                x = TestUtils.create_test_tensor(shape)
                x_recon = TestUtils.create_test_tensor(shape)
                
                loss = loss_fn(x, x_recon)
                
                self.assertIsInstance(loss, torch.Tensor)
                self.assertEqual(loss.shape, ())
                self.assertGreaterEqual(loss.item(), 0)

class TestComponentIntegration(unittest.TestCase):
    """Test integration between different components."""
    
    def test_autoencoder_with_different_latent_sizes(self):
        """Test AutoEncoder with various latent size configurations."""
        # Test different latent configurations
        test_configs = [
            {'latent_channels': 4, 'latent_base': 8},
            {'latent_channels': 8, 'latent_base': 16},
            {'latent_channels': 16, 'latent_base': 4},
            {'latent_channels': 32, 'latent_base': 8},
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                autoencoder = AutoEncoder.from_shape(
                    in_channels=3,
                    out_channels=3,
                    latent_channels=config['latent_channels'],
                    latent_base=config['latent_base'],
                    base_channels=32,
                    image_size=64
                )
                
                # Test forward pass
                input_tensor = TestUtils.create_test_tensor((1, 3, 64, 64))
                
                with torch.no_grad():
                    output = autoencoder(input_tensor)
                    encoded = autoencoder.encode(input_tensor)
                
                # Test shapes
                self.assertEqual(output.shape, input_tensor.shape)
                expected_latent_shape = (1, config['latent_channels'], config['latent_base'], config['latent_base'])
                self.assertEqual(encoded.shape, torch.Size(expected_latent_shape))
    
    def test_unet_with_different_latent_sizes(self):
        """Test UNet with various latent size configurations."""
        # Test different latent configurations
        test_configs = [
            {'in_channels': 4, 'out_channels': 4, 'base_channels': 32},
            {'in_channels': 8, 'out_channels': 8, 'base_channels': 64},
            {'in_channels': 16, 'out_channels': 16, 'base_channels': 128},
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                unet = DualUNet(
                    in_channels=config['in_channels'],
                    out_channels=config['out_channels'],
                    base_channels=config['base_channels'],
                    num_layers=2,
                    time_embed_dim=128
                )
                
                # Test forward pass
                input_shape = (2, config['in_channels'], 8, 8)
                input_tensor = TestUtils.create_test_tensor(input_shape)
                timesteps = torch.randint(0, 1000, (2,))
                
                with torch.no_grad():
                    output = unet(input_tensor, timesteps)
                
                # Test output shape
                self.assertEqual(output.shape, input_tensor.shape)


if __name__ == '__main__':
    unittest.main()
