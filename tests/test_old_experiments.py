"""
Test suite for loading and validating old experiments from the experiments/ directory.
This test ensures that experiments are still intact after class refactoring by:
1. Discovering all experiments in the experiments/ directory
2. Loading their configurations and checkpoints
3. Building models and running inference
4. Validating that outputs are reasonable
"""
import os
import sys
import unittest
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import tempfile
import shutil

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.builder import build_model
from models.autoencoder import AutoEncoder
from models.diffusion import LatentDiffusion
from common.utils import load_config_with_profile


class OldExperimentTester:
    """Utility class for testing old experiments."""
    
    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.discovered_experiments = []
        
    def discover_experiments(self) -> List[Dict[str, Any]]:
        """Discover all experiments in the experiments directory."""
        experiments = []
        
        if not self.experiments_dir.exists():
            print(f"Warning: Experiments directory {self.experiments_dir} does not exist")
            return experiments
            
        # Look for experiment directories
        for exp_category in self.experiments_dir.iterdir():
            if not exp_category.is_dir():
                continue
                
            for exp_name in exp_category.iterdir():
                if not exp_name.is_dir():
                    continue
                    
                # Look for experiment config files
                config_files = list(exp_name.glob("**/experiment_config.yaml"))
                autoencoder_config_files = list(exp_name.glob("**/autoencoder_config.yaml"))
                checkpoint_dirs = list(exp_name.glob("**/checkpoints"))
                
                if config_files or autoencoder_config_files:
                    experiment_info = {
                        'category': exp_category.name,
                        'name': exp_name.name,
                        'path': exp_name,
                        'config_files': config_files,
                        'autoencoder_config_files': autoencoder_config_files,
                        'checkpoint_dirs': checkpoint_dirs,
                        'type': self._determine_experiment_type(config_files, autoencoder_config_files)
                    }
                    experiments.append(experiment_info)
                    
        self.discovered_experiments = experiments
        return experiments
    
    def _determine_experiment_type(self, config_files: List[Path], autoencoder_config_files: List[Path]) -> str:
        """Determine if this is an autoencoder or diffusion experiment."""
        if config_files:
            # Check if it's a diffusion experiment by looking at the config
            try:
                with open(config_files[0], 'r') as f:
                    config = yaml.safe_load(f)
                if 'model' in config and 'diffusion' in config['model']:
                    return 'diffusion'
                elif 'model' in config and config['model'].get('type') in ['ae', 'autoencoder', 'vae']:
                    return 'autoencoder'
            except Exception as e:
                print(f"Warning: Could not parse config {config_files[0]}: {e}")
        
        if autoencoder_config_files:
            return 'autoencoder'
            
        return 'unknown'
    
    def load_experiment_config(self, experiment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load the experiment configuration."""
        try:
            if experiment['type'] == 'diffusion' and experiment['config_files']:
                config_path = experiment['config_files'][0]
                return load_config_with_profile(str(config_path))
            elif experiment['type'] == 'autoencoder' and experiment['autoencoder_config_files']:
                # For autoencoder experiments, we need to create a full config
                # Load the autoencoder config and create a minimal experiment config
                ae_config_path = experiment['autoencoder_config_files'][0]
                with open(ae_config_path, 'r') as f:
                    ae_config = yaml.safe_load(f)
                
                # Create a minimal experiment config
                config = {
                    'model': {
                        'type': 'ae',
                        'in_channels': ae_config.get('encoder', {}).get('in_channels', 3),
                        'out_channels': ae_config.get('decoder', {}).get('out_channels', 3),
                        'latent_channels': ae_config.get('encoder', {}).get('latent_channels', 4),
                        'latent_base': ae_config.get('encoder', {}).get('latent_base', 32),
                        'base_channels': 64,  # Default value
                        'image_size': ae_config.get('encoder', {}).get('image_size', 512),
                        'num_classes': ae_config.get('encoder', {}).get('num_classes', 13),
                        'dropout': 0.0,
                        'act': 'relu',
                        'norm': 'batch',
                        'use_sigmoid': True
                    },
                    'dataset': {
                        'batch_size': 32,
                        'num_workers': 4,
                        'seed': 42,
                        'shuffle': True,
                        'split_ratio': 0.9
                    },
                    'training': {
                        'epochs': 25,
                        'lr': 0.0001,
                        'eval_interval': 1,
                        'log_interval': 10,
                        'sample_interval': 200
                    }
                }
                return config
            else:
                print(f"Warning: Could not determine config type for {experiment['name']}")
                return None
                
        except Exception as e:
            print(f"Error loading config for {experiment['name']}: {e}")
            return None
    
    def find_latest_checkpoint(self, experiment: Dict[str, Any]) -> Optional[Path]:
        """Find the latest checkpoint for an experiment."""
        if not experiment['checkpoint_dirs']:
            return None
            
        checkpoint_dir = experiment['checkpoint_dirs'][0]
        
        # Look for common checkpoint patterns
        patterns = ['*_latest.pt', '*_epoch_*.pt', '*.pt']
        
        for pattern in patterns:
            checkpoints = list(checkpoint_dir.glob(pattern))
            if checkpoints:
                # Sort by modification time and return the latest
                checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                return checkpoints[0]
                
        return None
    
    def build_model_from_experiment(self, experiment: Dict[str, Any]) -> Optional[Tuple[torch.nn.Module, Optional[torch.nn.Module]]]:
        """Build model from experiment configuration."""
        try:
            config = self.load_experiment_config(experiment)
            if not config:
                return None
                
            # Update paths in config to be relative to current working directory
            config = self._update_config_paths(config, experiment)
            
            # For diffusion experiments, we need to handle the autoencoder dependency differently
            if experiment['type'] == 'diffusion' and 'autoencoder' in config['model']:
                # Build autoencoder first
                ae_config = config['model']['autoencoder']
                ae_checkpoint = ae_config.get('checkpoint')
                
                # Load autoencoder config
                ae_config_path = ae_config.get('config')
                if ae_config_path and Path(ae_config_path).exists():
                    with open(ae_config_path, 'r') as f:
                        ae_config_dict = yaml.safe_load(f)
                    
                    # Build autoencoder using the config
                    # The config has encoder/decoder sections, so it should build directly
                    autoencoder = AutoEncoder.from_config(ae_config_dict)
                    
                    # Load checkpoint if available
                    if ae_checkpoint and Path(ae_checkpoint).exists():
                        ae_state = torch.load(ae_checkpoint, map_location=self.device)
                        autoencoder.load_state_dict(ae_state.get("model", ae_state), strict=False)
                    
                    autoencoder.eval().to(self.device)
                    
                    # Update the model config to pass the built autoencoder
                    print(f"DEBUG test: Setting autoencoder in config, type = {type(autoencoder)}")
                    config['model']['autoencoder'] = autoencoder
                    print(f"DEBUG test: Config autoencoder type after setting = {type(config['model']['autoencoder'])}")
                else:
                    print(f"Warning: Autoencoder config not found for {experiment['name']}")
                    # Remove autoencoder from config to avoid errors
                    del config['model']['autoencoder']
            
            # Add model type to config
            if experiment['type'] == 'diffusion':
                config['model']['type'] = 'diffusion'
            elif experiment['type'] == 'autoencoder':
                config['model']['type'] = 'autoencoder'
            
            # Build model using the builder
            model, aux_model = build_model(config['model'], device=self.device)
            
            return model, aux_model
            
        except Exception as e:
            print(f"Error building model for {experiment['name']}: {e}")
            return None
    
    def _update_config_paths(self, config: Dict[str, Any], experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Update absolute paths in config to be relative to current working directory."""
        # Update autoencoder paths for diffusion experiments
        if experiment['type'] == 'diffusion' and 'autoencoder' in config['model']:
            ae_config = config['model']['autoencoder']
            
            # Update autoencoder config path
            if 'config' in ae_config:
                old_config_path = ae_config['config']
                if '/work3/s233249/ImgiNav/' in old_config_path:
                    # Replace with local path
                    new_config_path = old_config_path.replace('/work3/s233249/ImgiNav/', '')
                    if Path(new_config_path).exists():
                        ae_config['config'] = new_config_path
                        print(f"Updated autoencoder config path: {new_config_path}")
            
            # Update autoencoder checkpoint path
            if 'checkpoint' in ae_config:
                old_checkpoint_path = ae_config['checkpoint']
                if '/work3/s233249/ImgiNav/' in old_checkpoint_path:
                    # Replace with local path
                    new_checkpoint_path = old_checkpoint_path.replace('/work3/s233249/ImgiNav/', '')
                    if Path(new_checkpoint_path).exists():
                        ae_config['checkpoint'] = new_checkpoint_path
                        print(f"Updated autoencoder checkpoint path: {new_checkpoint_path}")
        
        return config
    
    def load_checkpoint(self, model: torch.nn.Module, checkpoint_path: Path) -> bool:
        """Load checkpoint into model."""
        try:
            if not checkpoint_path.exists():
                print(f"Checkpoint not found: {checkpoint_path}")
                return False
                
            state = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model' in state:
                state_dict = state['model']
            elif 'state_dict' in state:
                state_dict = state['state_dict']
            else:
                state_dict = state
            
            # Handle DataParallel prefix
            if state_dict and list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            print(f"Successfully loaded checkpoint: {checkpoint_path}")
            return True
            
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
            return False
    
    def test_model_inference(self, model: torch.nn.Module, experiment: Dict[str, Any]) -> bool:
        """Test model inference with dummy data."""
        try:
            model.eval()
            
            if experiment['type'] == 'autoencoder':
                # Test autoencoder inference
                batch_size = 2
                image_size = 512  # Default from configs
                input_tensor = torch.randn(batch_size, 3, image_size, image_size, device=self.device)
                
                with torch.no_grad():
                    # Test forward pass
                    output = model(input_tensor)
                    assert output.shape == input_tensor.shape, f"Output shape {output.shape} != input shape {input_tensor.shape}"
                    
                    # Test encode/decode
                    encoded = model.encode(input_tensor)
                    decoded = model.decode(encoded, from_latent=True)
                    assert decoded.shape == input_tensor.shape, f"Decoded shape {decoded.shape} != input shape {input_tensor.shape}"
                    
                    # Test that encoded has reasonable shape
                    assert encoded.shape[1] > 0, "Encoded should have channels"
                    assert encoded.shape[2] > 0, "Encoded should have height"
                    assert encoded.shape[3] > 0, "Encoded should have width"
                    
                    print(f"  ✓ Autoencoder forward pass successful")
                    print(f"  ✓ Encode/decode cycle successful")
                    
            elif experiment['type'] == 'diffusion':
                # Test diffusion inference
                batch_size = 2
                # Get latent shape from autoencoder if available
                if hasattr(model, 'autoencoder') and model.autoencoder is not None:
                    test_input = torch.randn(batch_size, 3, 512, 512, device=self.device)
                    with torch.no_grad():
                        latent = model.autoencoder.encode(test_input)
                    print(f"  ✓ Autoencoder encoding successful, latent shape: {latent.shape}")
                else:
                    # Use default latent shape
                    latent = torch.randn(batch_size, 8, 32, 32, device=self.device)
                    print(f"  ⚠ Using default latent shape: {latent.shape}")
                
                batch = {"layout": latent}
                
                with torch.no_grad():
                    # Test forward pass
                    output = model(batch)
                    assert 'pred_noise' in output, "Output should contain 'pred_noise'"
                    assert output['pred_noise'].shape == latent.shape, f"Pred noise shape {output['pred_noise'].shape} != latent shape {latent.shape}"
                    print(f"  ✓ Diffusion forward pass successful")
                    
                    # Test sampling (if available)
                    if hasattr(model, 'sample_ddpm'):
                        samples = model.sample_ddpm(batch_size=1, num_steps=10)
                        assert samples.shape[0] == 1, "Sample batch size should be 1"
                        assert len(samples.shape) == 4, "Samples should be 4D tensor"
                        print(f"  ✓ DDPM sampling successful, sample shape: {samples.shape}")
                    
                    if hasattr(model, 'sample_ddim'):
                        samples = model.sample_ddim(batch_size=1, num_steps=10)
                        assert samples.shape[0] == 1, "Sample batch size should be 1"
                        assert len(samples.shape) == 4, "Samples should be 4D tensor"
                        print(f"  ✓ DDIM sampling successful, sample shape: {samples.shape}")
                        
            return True
            
        except Exception as e:
            print(f"Error during inference test for {experiment['name']}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_experiment(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single experiment."""
        result = {
            'experiment': experiment['name'],
            'category': experiment['category'],
            'type': experiment['type'],
            'success': False,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Build model
            model_result = self.build_model_from_experiment(experiment)
            if not model_result:
                result['errors'].append("Failed to build model")
                return result
                
            model, aux_model = model_result
            
            # Find and load checkpoint
            checkpoint_path = self.find_latest_checkpoint(experiment)
            if checkpoint_path:
                print(f"  Loading checkpoint: {checkpoint_path}")
                if not self.load_checkpoint(model, checkpoint_path):
                    result['warnings'].append("Failed to load checkpoint")
                    print(f"  ✗ Failed to load checkpoint")
                else:
                    print(f"  ✓ Checkpoint loaded successfully")
            else:
                result['warnings'].append("No checkpoint found")
                print(f"  ⚠ No checkpoint found")
            
            # Test inference
            if not self.test_model_inference(model, experiment):
                result['errors'].append("Inference test failed")
                return result
                
            result['success'] = True
            
        except Exception as e:
            result['errors'].append(f"Unexpected error: {str(e)}")
            
        return result


class TestOldExperiments(unittest.TestCase):
    """Test suite for old experiments."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tester = OldExperimentTester()
        self.experiments = self.tester.discover_experiments()
        
    def test_experiments_discovered(self):
        """Test that experiments are discovered."""
        self.assertGreater(len(self.experiments), 0, "No experiments discovered")
        print(f"Discovered {len(self.experiments)} experiments:")
        for exp in self.experiments:
            print(f"  - {exp['category']}/{exp['name']} ({exp['type']})")
    
    def test_autoencoder_experiments(self):
        """Test all autoencoder experiments."""
        autoencoder_experiments = [exp for exp in self.experiments if exp['type'] == 'autoencoder']
        
        self.assertGreater(len(autoencoder_experiments), 0, "No autoencoder experiments found")
        
        for experiment in autoencoder_experiments:
            with self.subTest(experiment=experiment['name']):
                result = self.tester.test_experiment(experiment)
                
                # Print result for debugging
                print(f"\nTesting {experiment['name']}:")
                print(f"  Success: {result['success']}")
                if result['errors']:
                    print(f"  Errors: {result['errors']}")
                if result['warnings']:
                    print(f"  Warnings: {result['warnings']}")
                
                # Assert success (you might want to make this more lenient)
                self.assertTrue(result['success'], f"Experiment {experiment['name']} failed: {result['errors']}")
    
    def test_diffusion_experiments(self):
        """Test all diffusion experiments."""
        diffusion_experiments = [exp for exp in self.experiments if exp['type'] == 'diffusion']
        
        self.assertGreater(len(diffusion_experiments), 0, "No diffusion experiments found")
        
        for experiment in diffusion_experiments:
            with self.subTest(experiment=experiment['name']):
                result = self.tester.test_experiment(experiment)
                
                # Print result for debugging
                print(f"\nTesting {experiment['name']}:")
                print(f"  Success: {result['success']}")
                if result['errors']:
                    print(f"  Errors: {result['errors']}")
                if result['warnings']:
                    print(f"  Warnings: {result['warnings']}")
                
                # Assert success (you might want to make this more lenient)
                self.assertTrue(result['success'], f"Experiment {experiment['name']} failed: {result['errors']}")
    
    def test_all_experiments(self):
        """Test all experiments and provide summary."""
        results = []
        
        print(f"Testing {len(self.experiments)} experiments...")
        for i, experiment in enumerate(self.experiments, 1):
            print(f"\n[{i}/{len(self.experiments)}] Testing {experiment['category']}/{experiment['name']} ({experiment['type']})...")
            result = self.tester.test_experiment(experiment)
            results.append(result)
        
        # Print summary
        print(f"\n{'='*60}")
        print("EXPERIMENT TEST SUMMARY")
        print(f"{'='*60}")
        
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        
        print(f"Total experiments: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")
        print(f"Success rate: {successful/total*100:.1f}%")
        
        print(f"\nDetailed Results:")
        for result in results:
            status = "✓" if result['success'] else "✗"
            print(f"  {status} {result['category']}/{result['experiment']} ({result['type']})")
            if result['errors']:
                for error in result['errors']:
                    print(f"    Error: {error}")
            if result['warnings']:
                for warning in result['warnings']:
                    print(f"    Warning: {warning}")
        
        # Assert that at least some experiments work
        self.assertGreater(successful, 0, "No experiments passed the test")
        
        # Optionally, you might want to be more lenient here
        # self.assertGreaterEqual(successful, total * 0.8, "Less than 80% of experiments passed")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
