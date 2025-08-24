import torch
import torch.nn as nn
import torch.optim as optim
import os
import yaml
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from .test_utils import create_dummy_dataloader
from src.layout_generator.training.autoencoder_trainer import AutoEncoderTrainer
from src.layout_generator.training.diffusion_trainer import DiffusionTrainer



def test_autoencoder_lifecycle(tmp_path):
    """
    Tests the full build -> train -> save -> load cycle for the AutoEncoder.
    """
    # 1. SETUP: Create temporary directories and config files
    checkpoint_dir = tmp_path / "checkpoints"
    image_dir = tmp_path / "images"
    encoder_config_path = tmp_path / "encoder_config.yaml"
    decoder_config_path = tmp_path / "decoder_config.yaml"

    encoder_config = {
        'model': {'name': 'Encoder', 'params': {'architecture': [3, 16, 4]}}
    }
    decoder_config = {
        'model': {'name': 'Decoder', 'params': {'architecture': [4, 16, 3]}}
    }
    with open(encoder_config_path, 'w') as f: yaml.dump(encoder_config, f)
    with open(decoder_config_path, 'w') as f: yaml.dump(decoder_config, f)

    # 2. BUILD & TRAIN: Instantiate and train the model for a few epochs
    dataloader = create_dummy_dataloader(2, 4, {'image': (3, 32, 32)})

    trainer = AutoEncoderTrainer(
        encoder_config_path=str(encoder_config_path),
        decoder_config_path=str(decoder_config_path),
        dataloader=dataloader,
        optimizer_class=optim.Adam,
        optimizer_params={'lr': 1e-4},
        loss_fn=nn.MSELoss(),
        device='cpu',
        image_dir=str(image_dir),
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_interval=1 # Save every epoch
    )
    trainer.train(epochs=1, log_images_every=1)

    # 3. ASSERT (SAVE): Check if all expected files were created
    assert os.path.exists(checkpoint_dir / "encoder_epoch001.pt")
    assert os.path.exists(checkpoint_dir / "decoder_epoch001.pt")
    assert os.path.exists(checkpoint_dir / "encoder_config.yaml")
    assert os.path.exists(checkpoint_dir / "decoder_metadata.json")
    assert os.path.exists(image_dir / "recon_epoch_001.png")

    # 4. BUILD & LOAD: Create a new trainer instance to test auto-resume
    print("\n--- Testing Auto-Resume for AutoEncoder ---")
    new_trainer = AutoEncoderTrainer(
        encoder_config_path=str(encoder_config_path),
        decoder_config_path=str(decoder_config_path),
        dataloader=dataloader,
        optimizer_class=optim.Adam,
        optimizer_params={'lr': 1e-4},
        loss_fn=nn.MSELoss(),
        device='cpu',
        image_dir=str(image_dir),
        checkpoint_dir=str(checkpoint_dir)
    )

    # 5. ASSERT (LOAD): Check if the trainer resumed from the saved state
    assert new_trainer.start_epoch == 1
    print("✅ AutoEncoder lifecycle test passed!")


def test_diffusion_model_lifecycle(tmp_path):
    """
    Tests the full build -> train -> save -> load cycle for the DiffusionModel.
    """
    # 1. SETUP: Create temporary directories and a complete diffusion config
    checkpoint_dir = tmp_path / "checkpoints_diff"
    image_dir = tmp_path / "images_diff"
    diffusion_config_path = tmp_path / "diffusion_config.yaml"

    diffusion_config = {
        'model': {
            'name': 'DiffusionModel',
            'params': {
                'name': 'diffusion_test_model',
                'unet': {
                    'name': 'UNet',
                    'params': {
                        'in_channels': 4, 'out_channels': 4, 'time_dim': 32,
                        'conditioning': {'name': 'VectorAdditive', 'params': {'cond_dim': 16, 'time_dim': 32}},
                        'architecture': {'encoder': [16, 32], 'decoder': [16]}
                    }
                },
                'scheduler': {'num_timesteps': 100}
            }
        }
    }
    with open(diffusion_config_path, 'w') as f: yaml.dump(diffusion_config, f)

    # 2. BUILD & TRAIN
    item_spec = {'image_latent': (4, 16, 16), 'token_embedding': (16,)}
    dataloader = create_dummy_dataloader(2, 4, item_spec)

    trainer = DiffusionTrainer(
        config_path=str(diffusion_config_path),
        dataloader=dataloader,
        optimizer_class=optim.Adam,
        optimizer_params={'lr': 1e-4},
        loss_fn=nn.MSELoss(),
        device='cpu',
        checkpoint_dir=str(checkpoint_dir),
        image_dir=str(image_dir),
        checkpoint_interval=1,
        sample_interval=1
    )
    trainer.train(epochs=1)

    # 3. ASSERT (SAVE)
    assert os.path.exists(checkpoint_dir / "diffusion_test_model_epoch001.pt")
    assert os.path.exists(checkpoint_dir / "training_config.yaml")
    assert os.path.exists(checkpoint_dir / "diffusion_test_model_metadata.json")
    assert os.path.exists(image_dir / "sample_epoch_001.png")

    # 4. BUILD & LOAD
    print("\n--- Testing Auto-Resume for DiffusionModel ---")
    new_trainer = DiffusionTrainer(
        config_path=str(diffusion_config_path),
        dataloader=dataloader,
        optimizer_class=optim.Adam,
        optimizer_params={'lr': 1e-4},
        loss_fn=nn.MSELoss(),
        device='cpu',
        checkpoint_dir=str(checkpoint_dir),
        image_dir=str(image_dir)
    )

    # 5. ASSERT (LOAD)
    assert new_trainer.start_epoch == 1
    print("✅ DiffusionModel lifecycle test passed!")