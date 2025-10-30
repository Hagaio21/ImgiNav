"""
Comprehensive tests for all dataset classes and utilities.
Tests LayoutDataset, PovDataset, GraphDataset, UnifiedLayoutDataset,
and related utility functions using the test_dataset manifests.
"""
import os
import sys
import tempfile
import shutil
import unittest
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.datasets import (
    LayoutDataset, PovDataset, GraphDataset, UnifiedLayoutDataset,
    load_image, load_embedding, load_graph_text, valid_path, 
    compute_sample_weights, build_datasets, build_dataloaders, 
    save_split_csvs, collate_skip_none, collate_fn
)


class TestDatasetUtilities(unittest.TestCase):
    """Test utility functions for dataset operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.test_dir, "test_image.png")
        
        # Create a test image
        test_image = Image.new('RGB', (64, 64), color='red')
        test_image.save(self.test_image_path)
        
        # Create test embedding
        self.test_embedding_path = os.path.join(self.test_dir, "test_embedding.pt")
        test_embedding = torch.randn(4, 8, 8)
        torch.save(test_embedding, self.test_embedding_path)
        
        # Create test graph text
        self.test_graph_path = os.path.join(self.test_dir, "test_graph.json")
        test_graph = {"nodes": [{"id": 1, "type": "room"}], "edges": []}
        with open(self.test_graph_path, 'w') as f:
            import json
            json.dump(test_graph, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_load_image(self):
        """Test image loading with and without transforms."""
        # Test without transform
        img = load_image(self.test_image_path)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (64, 64))
        
        # Test with transform
        transform = T.ToTensor()
        img_tensor = load_image(self.test_image_path, transform)
        self.assertIsInstance(img_tensor, torch.Tensor)
        self.assertEqual(img_tensor.shape, (3, 64, 64))
        self.assertTrue(torch.all(img_tensor >= 0) and torch.all(img_tensor <= 1))
    
    def test_load_embedding(self):
        """Test embedding loading from .pt files."""
        embedding = load_embedding(self.test_embedding_path)
        self.assertIsInstance(embedding, torch.Tensor)
        self.assertEqual(embedding.shape, (4, 8, 8))
        self.assertEqual(embedding.dtype, torch.float32)
    
    def test_load_graph_text(self):
        """Test graph text loading."""
        graph_text = load_graph_text(self.test_graph_path)
        self.assertIsInstance(graph_text, str)
        self.assertIn("nodes", graph_text)
        self.assertIn("room", graph_text)
    
    def test_valid_path(self):
        """Test path validation function."""
        self.assertTrue(valid_path("valid_path.png"))
        self.assertTrue(valid_path("valid_path.pt"))
        self.assertFalse(valid_path(""))
        self.assertFalse(valid_path("false"))
        self.assertFalse(valid_path("0"))
        self.assertFalse(valid_path("none"))
        self.assertFalse(valid_path(None))
    
    def test_compute_sample_weights(self):
        """Test sample weight computation."""
        df = pd.DataFrame({
            'type': ['room', 'room', 'scene', 'scene', 'scene'],
            'room_id': ['room1', 'room2', 'scene', 'scene', 'scene']
        })
        weights = compute_sample_weights(df)
        
        self.assertIsInstance(weights, torch.Tensor)
        self.assertEqual(len(weights), 5)
        self.assertAlmostEqual(weights.sum().item(), 1.0, places=5)
        
        # Room samples should have higher weights (less frequent)
        room_weights = weights[:2]
        scene_weights = weights[2:]
        self.assertTrue(torch.all(room_weights > scene_weights))


class TestLayoutDataset(unittest.TestCase):
    """Test LayoutDataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manifest_path = "test_dataset/manifests/layouts_manifest.csv"
        self.taxonomy_path = "config/taxonomy.json"
        self.transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    
    def test_rgb_mode(self):
        """Test LayoutDataset in RGB mode."""
        dataset = LayoutDataset(
            manifest_path=self.manifest_path,
            transform=self.transform,
            mode="all",
            one_hot=False,
            skip_empty=True,
            return_embeddings=False
        )
        
        self.assertGreater(len(dataset), 0)
        
        # Test single sample
        sample = dataset[0]
        self.assertIn("layout", sample)
        self.assertIn("scene_id", sample)
        self.assertIn("room_id", sample)
        self.assertIn("type", sample)
        self.assertIn("is_empty", sample)
        self.assertIn("path", sample)
        
        # Check layout tensor
        layout = sample["layout"]
        self.assertIsInstance(layout, torch.Tensor)
        self.assertEqual(layout.shape, (3, 64, 64))
        self.assertTrue(torch.all(layout >= 0) and torch.all(layout <= 1))
    
    def test_one_hot_mode(self):
        """Test LayoutDataset in one-hot mode with taxonomy."""
        dataset = LayoutDataset(
            manifest_path=self.manifest_path,
            transform=self.transform,
            mode="all",
            one_hot=True,
            taxonomy_path=self.taxonomy_path,
            skip_empty=True,
            return_embeddings=False
        )
        
        self.assertGreater(len(dataset), 0)
        self.assertIsNotNone(dataset.COLOR_TO_CLASS)
        self.assertIsNotNone(dataset.NUM_CLASSES)
        
        # Test single sample
        sample = dataset[0]
        layout = sample["layout"]
        self.assertIsInstance(layout, torch.Tensor)
        self.assertEqual(layout.dtype, torch.long)
        self.assertEqual(len(layout.shape), 2)  # (H, W)
    
    def test_embedding_mode(self):
        """Test LayoutDataset with embeddings."""
        # Create a dummy embedding manifest
        temp_dir = tempfile.mkdtemp()
        try:
            # Read original manifest and add embedding paths
            df = pd.read_csv(self.manifest_path)
            df['layout_emb'] = df['layout_path'].str.replace('.png', '.pt')
            temp_manifest = os.path.join(temp_dir, "test_layouts_emb.csv")
            df.to_csv(temp_manifest, index=False)
            
            # Create dummy embeddings
            for _, row in df.head(5).iterrows():
                emb_path = row['layout_emb']
                os.makedirs(os.path.dirname(emb_path), exist_ok=True)
                dummy_emb = torch.randn(4, 8, 8)
                torch.save(dummy_emb, emb_path)
            
            dataset = LayoutDataset(
                manifest_path=temp_manifest,
                transform=None,
                mode="all",
                one_hot=False,
                skip_empty=True,
                return_embeddings=True
            )
            
            self.assertGreater(len(dataset), 0)
            sample = dataset[0]
            layout = sample["layout"]
            self.assertIsInstance(layout, torch.Tensor)
            self.assertEqual(layout.shape, (4, 8, 8))
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_filtering(self):
        """Test dataset filtering by mode and empty samples."""
        # Test scene-only mode
        scene_dataset = LayoutDataset(
            manifest_path=self.manifest_path,
            transform=self.transform,
            mode="scene",
            skip_empty=True
        )
        
        # Test room-only mode (if any exist)
        room_dataset = LayoutDataset(
            manifest_path=self.manifest_path,
            transform=self.transform,
            mode="room",
            skip_empty=True
        )
        
        # All samples should be scenes based on the manifest
        self.assertGreater(len(scene_dataset), 0)
        for sample in scene_dataset:
            self.assertEqual(sample["type"], "scene")
    
    def test_rgb_to_class_index(self):
        """Test RGB to class index conversion."""
        dataset = LayoutDataset(
            manifest_path=self.manifest_path,
            transform=self.transform,
            mode="all",
            one_hot=True,
            taxonomy_path=self.taxonomy_path,
            skip_empty=True
        )
        
        # Create a test RGB tensor
        test_rgb = torch.zeros(3, 64, 64)
        class_map = dataset.rgb_to_class_index(test_rgb)
        
        self.assertIsInstance(class_map, torch.Tensor)
        self.assertEqual(class_map.shape, (64, 64))
        self.assertEqual(class_map.dtype, torch.long)


class TestPovDataset(unittest.TestCase):
    """Test PovDataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manifest_path = "test_dataset/manifests/layouts_manifest.csv"  # Using layouts as proxy
        self.transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    
    def test_pov_dataset_creation(self):
        """Test PovDataset creation and basic functionality."""
        # Create a mock POV manifest
        temp_dir = tempfile.mkdtemp()
        try:
            # Create mock POV manifest
            pov_data = {
                'scene_id': ['test1', 'test2'],
                'type': ['seg', 'tex'],
                'room_id': ['room1', 'room2'],
                'is_empty': [0, 0],
                'pov_path': [os.path.join(temp_dir, 'pov1.png'), os.path.join(temp_dir, 'pov2.png')]
            }
            
            # Create test images
            for path in pov_data['pov_path']:
                img = Image.new('RGB', (64, 64), color='blue')
                img.save(path)
            
            pov_df = pd.DataFrame(pov_data)
            pov_manifest = os.path.join(temp_dir, "pov_manifest.csv")
            pov_df.to_csv(pov_manifest, index=False)
            
            dataset = PovDataset(
                manifest_path=pov_manifest,
                transform=self.transform,
                pov_type="seg",
                skip_empty=True,
                return_embeddings=False
            )
            
            self.assertEqual(len(dataset), 1)  # Only seg type
            sample = dataset[0]
            self.assertIn("pov", sample)
            self.assertIn("scene_id", sample)
            self.assertEqual(sample["type"], "seg")
            
        finally:
            shutil.rmtree(temp_dir)


class TestGraphDataset(unittest.TestCase):
    """Test GraphDataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manifest_path = "test_dataset/manifests/graphs_manifest.csv"
    
    def test_graph_dataset_creation(self):
        """Test GraphDataset creation and basic functionality."""
        dataset = GraphDataset(
            manifest_path=self.manifest_path,
            return_embeddings=False
        )
        
        self.assertGreater(len(dataset), 0)
        sample = dataset[0]
        self.assertIn("graph", sample)
        self.assertIn("scene_id", sample)
        self.assertIn("room_id", sample)
        self.assertIn("type", sample)
        self.assertIn("is_empty", sample)
        self.assertIn("path", sample)
        
        # Graph should be text
        graph = sample["graph"]
        self.assertIsInstance(graph, str)
        self.assertGreater(len(graph), 0)


class TestUnifiedLayoutDataset(unittest.TestCase):
    """Test UnifiedLayoutDataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manifest_path = "test_dataset/manifests/all_manifest.csv"
        self.transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    
    def test_unified_dataset_creation(self):
        """Test UnifiedLayoutDataset creation."""
        dataset = UnifiedLayoutDataset(
            manifest_path=self.manifest_path,
            use_embeddings=False,
            sample_type="both",
            pov_type=None,
            transform=self.transform,
            device=None
        )
        
        self.assertGreater(len(dataset), 0)
        sample = dataset[0]
        
        # Check required keys
        required_keys = ["sample_id", "scene_id", "sample_type", "pov", "graph", "layout"]
        for key in required_keys:
            self.assertIn(key, sample)
        
        # Check sample type
        self.assertIn(sample["sample_type"], ["room", "scene"])
        
        # Check data types
        if sample["sample_type"] == "room":
            self.assertIsNotNone(sample["pov"])
            self.assertIsNotNone(sample["room_id"])
        else:
            self.assertIsNone(sample["pov"])
            self.assertIsNone(sample["room_id"])
        
        # Graph and layout should always be present
        self.assertIsNotNone(sample["graph"])
        self.assertIsNotNone(sample["layout"])
    
    def test_sample_type_filtering(self):
        """Test filtering by sample type."""
        # Test scene-only
        scene_dataset = UnifiedLayoutDataset(
            manifest_path=self.manifest_path,
            use_embeddings=False,
            sample_type="scene",
            transform=self.transform
        )
        
        for sample in scene_dataset:
            self.assertEqual(sample["sample_type"], "scene")
            self.assertIsNone(sample["pov"])
    
    def test_pov_type_filtering(self):
        """Test filtering by POV type."""
        # Test with specific POV type
        dataset = UnifiedLayoutDataset(
            manifest_path=self.manifest_path,
            use_embeddings=False,
            sample_type="both",
            pov_type="seg",
            transform=self.transform
        )
        
        for sample in dataset:
            if sample["sample_type"] == "room":
                self.assertEqual(sample["pov_type"], "seg")


class TestCollateFunctions(unittest.TestCase):
    """Test collate functions."""
    
    def test_collate_skip_none(self):
        """Test collate_skip_none function."""
        # Test with valid batch
        batch = [
            {"layout": torch.randn(3, 64, 64), "scene_id": "test1"},
            {"layout": torch.randn(3, 64, 64), "scene_id": "test2"}
        ]
        collated = collate_skip_none(batch)
        self.assertIsNotNone(collated)
        self.assertIn("layout", collated)
        self.assertEqual(collated["layout"].shape, (2, 3, 64, 64))
        
        # Test with None values
        batch_with_none = [
            {"layout": torch.randn(3, 64, 64), "scene_id": "test1"},
            None,
            {"layout": torch.randn(3, 64, 64), "scene_id": "test2"}
        ]
        collated = collate_skip_none(batch_with_none)
        self.assertIsNotNone(collated)
        self.assertEqual(collated["layout"].shape, (2, 3, 64, 64))
        
        # Test with empty batch
        empty_batch = []
        collated = collate_skip_none(empty_batch)
        self.assertIsNone(collated)
    
    def test_collate_fn(self):
        """Test collate_fn function."""
        # Test with layout items
        batch = [
            {"layout": torch.randn(3, 64, 64)},
            {"layout": torch.randn(3, 64, 64)}
        ]
        collated = collate_fn(batch)
        self.assertIn("layout", collated)
        self.assertEqual(collated["layout"].shape, (2, 3, 64, 64))


class TestDatasetBuilders(unittest.TestCase):
    """Test dataset builder functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manifest_path = "test_dataset/manifests/layouts_manifest.csv"
        self.taxonomy_path = "config/taxonomy.json"
        self.transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_build_datasets(self):
        """Test build_datasets function."""
        dataset_cfg = {
            "manifest": self.manifest_path,
            "split_ratio": 0.8,
            "seed": 42,
            "one_hot": False,
            "taxonomy_path": self.taxonomy_path,
            "return_embeddings": False,
            "skip_empty": True
        }
        
        train_ds, val_ds = build_datasets(dataset_cfg, transform=self.transform)
        
        self.assertIsNotNone(train_ds)
        self.assertIsNotNone(val_ds)
        self.assertGreater(len(train_ds), 0)
        self.assertGreater(len(val_ds), 0)
        
        # Test train sample
        train_sample = train_ds[0]
        self.assertIn("layout", train_sample)
        self.assertIsInstance(train_sample["layout"], torch.Tensor)
    
    def test_build_dataloaders(self):
        """Test build_dataloaders function."""
        dataset_cfg = {
            "manifest": self.manifest_path,
            "batch_size": 4,
            "num_workers": 0,  # Use 0 for testing
            "shuffle": True,
            "pin_memory": False,
            "split_ratio": 0.8,
            "seed": 42,
            "return_embeddings": False,
            "skip_empty": True
        }
        
        train_ds, val_ds, train_loader, val_loader = build_dataloaders(
            dataset_cfg, transform=self.transform
        )
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        
        # Test batch loading
        train_batch = next(iter(train_loader))
        self.assertIn("layout", train_batch)
        self.assertEqual(train_batch["layout"].shape[0], 4)  # batch_size
    
    def test_save_split_csvs(self):
        """Test save_split_csvs function."""
        dataset_cfg = {
            "manifest": self.manifest_path,
            "split_ratio": 0.8,
            "seed": 42,
            "return_embeddings": False,
            "skip_empty": True
        }
        
        train_ds, val_ds = build_datasets(dataset_cfg, transform=self.transform)
        save_split_csvs(train_ds, val_ds, self.temp_dir)
        
        # Check if files were created
        train_csv = os.path.join(self.temp_dir, "trained_on.csv")
        val_csv = os.path.join(self.temp_dir, "evaluated_on.csv")
        
        self.assertTrue(os.path.exists(train_csv))
        self.assertTrue(os.path.exists(val_csv))
        
        # Check CSV contents
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        
        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(val_df), 0)
        self.assertIn("layout_path", train_df.columns)
        self.assertIn("layout_path", val_df.columns)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
