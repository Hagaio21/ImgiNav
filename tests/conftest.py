"""
Pytest fixtures for model and training tests.
"""
import pytest
import torch
import yaml
from pathlib import Path
import sys
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.autoencoder import Autoencoder
from models.diffusion import DiffusionModel
from training.utils import load_config


@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    import tempfile
    import shutil
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_autoencoder_config(test_data_dir):
    """Load minimal autoencoder config."""
    config_path = test_data_dir / "configs" / "test_autoencoder_minimal.yaml"
    return load_config(config_path)


@pytest.fixture
def test_vae_config(test_data_dir):
    """Load minimal VAE config."""
    config_path = test_data_dir / "configs" / "test_vae_minimal.yaml"
    return load_config(config_path)


@pytest.fixture
def test_diffusion_config(test_data_dir):
    """Load minimal diffusion config."""
    config_path = test_data_dir / "configs" / "test_diffusion_minimal.yaml"
    return load_config(config_path)


@pytest.fixture
def autoencoder_model(test_autoencoder_config):
    """Create an autoencoder model from test config."""
    ae_cfg = test_autoencoder_config.get("autoencoder", {})
    model = Autoencoder.from_config(ae_cfg)
    return model


@pytest.fixture
def vae_model(test_vae_config):
    """Create a VAE model from test config."""
    ae_cfg = test_vae_config.get("autoencoder", {})
    model = Autoencoder.from_config(ae_cfg)
    return model


@pytest.fixture
def diffusion_model(test_diffusion_config):
    """Create a diffusion model from test config."""
    # Extract diffusion config from test config
    diffusion_cfg = {
        "autoencoder": test_diffusion_config.get("autoencoder"),
        "unet": test_diffusion_config.get("unet", {}),
        "scheduler": test_diffusion_config.get("scheduler", {}),
        "embedding_projection": test_diffusion_config.get("embedding_projection"),
        "scale_factor": test_diffusion_config.get("scale_factor", 1.0),
        "latent_clamp_min": test_diffusion_config.get("latent_clamp_min"),
        "latent_clamp_max": test_diffusion_config.get("latent_clamp_max"),
    }
    model = DiffusionModel.from_config(diffusion_cfg)
    return model


@pytest.fixture
def device():
    """Return device for testing (CPU for consistency)."""
    return torch.device("cpu")


@pytest.fixture
def dummy_image_batch():
    """Create a dummy image batch for testing."""
    return torch.randn(2, 3, 64, 64)


@pytest.fixture
def dummy_latent_batch():
    """Create a dummy latent batch for testing."""
    return torch.randn(2, 4, 16, 16)

