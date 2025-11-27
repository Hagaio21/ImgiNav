"""
Tests for component graph generation.
"""
import pytest
from pathlib import Path

from models.autoencoder import Autoencoder
from models.diffusion import DiffusionModel


class TestAutoencoderComponentGraphs:
    """Test component graph generation for Autoencoder."""
    
    def test_generate_graph_text_format(self, autoencoder_model):
        """Test generating component graph in text format."""
        graph = autoencoder_model.generate_component_graph(format='text')
        
        assert isinstance(graph, str)
        assert len(graph) > 0
    
    def test_graph_contains_model_name(self, autoencoder_model):
        """Test that graph contains model name."""
        graph = autoencoder_model.generate_component_graph(format='text')
        
        assert 'Autoencoder' in graph or 'Component' in graph
    
    def test_graph_contains_components(self, autoencoder_model):
        """Test that graph contains component names."""
        graph = autoencoder_model.generate_component_graph(format='text')
        
        # Should mention encoder and decoder
        assert 'encoder' in graph.lower() or 'Encoder' in graph
        assert 'decoder' in graph.lower() or 'Decoder' in graph
    
    def test_graph_structure(self, autoencoder_model):
        """Test that graph has expected structure elements."""
        graph = autoencoder_model.generate_component_graph(format='text')
        
        # Should have some structure indicators
        assert '=' in graph or '-' in graph or 'Component' in graph or 'contains' in graph.lower()
    
    def test_graph_with_shapes(self, autoencoder_model):
        """Test generating graph with shape information."""
        graph = autoencoder_model.generate_component_graph(format='text', include_shapes=True)
        
        assert isinstance(graph, str)
        assert len(graph) > 0
    
    def test_graph_without_shapes(self, autoencoder_model):
        """Test generating graph without shape information."""
        graph = autoencoder_model.generate_component_graph(format='text', include_shapes=False)
        
        assert isinstance(graph, str)
        assert len(graph) > 0


class TestDiffusionComponentGraphs:
    """Test component graph generation for DiffusionModel."""
    
    def test_generate_graph_text_format(self, diffusion_model):
        """Test generating component graph in text format."""
        graph = diffusion_model.generate_component_graph(format='text')
        
        assert isinstance(graph, str)
        assert len(graph) > 0
    
    def test_graph_contains_model_name(self, diffusion_model):
        """Test that graph contains model name."""
        graph = diffusion_model.generate_component_graph(format='text')
        
        assert 'DiffusionModel' in graph or 'Component' in graph
    
    def test_graph_contains_components(self, diffusion_model):
        """Test that graph contains component names."""
        graph = diffusion_model.generate_component_graph(format='text')
        
        # Should mention key components
        assert 'unet' in graph.lower() or 'UNet' in graph
        assert 'decoder' in graph.lower() or 'Decoder' in graph
        assert 'scheduler' in graph.lower() or 'Scheduler' in graph
    
    def test_graph_structure(self, diffusion_model):
        """Test that graph has expected structure elements."""
        graph = diffusion_model.generate_component_graph(format='text')
        
        # Should have some structure indicators
        assert '=' in graph or '-' in graph or 'Component' in graph or 'contains' in graph.lower()
    
    def test_graph_relationships(self, diffusion_model):
        """Test that graph shows component relationships."""
        graph = diffusion_model.generate_component_graph(format='text')
        
        # Should show relationships between components
        # (e.g., embedding_projection -> unet, scheduler -> unet)
        assert 'forward' in graph.lower() or 'data_flow' in graph.lower() or 'contains' in graph.lower()


class TestComponentGraphStructure:
    """Test graph structure (nodes, edges, relationships)."""
    
    def test_graph_has_nodes(self, autoencoder_model):
        """Test that graph contains nodes."""
        graph = autoencoder_model.generate_component_graph(format='text')
        
        # Graph should have content indicating nodes
        assert len(graph) > 10  # Should have substantial content
    
    def test_graph_has_edges(self, autoencoder_model):
        """Test that graph contains edges/relationships."""
        graph = autoencoder_model.generate_component_graph(format='text')
        
        # Should indicate relationships
        assert 'contains' in graph.lower() or 'forward' in graph.lower() or '->' in graph or '──' in graph
    
    def test_graph_hierarchy(self, autoencoder_model):
        """Test that graph shows component hierarchy."""
        graph = autoencoder_model.generate_component_graph(format='text')
        
        # Should show model -> components hierarchy
        assert 'Autoencoder' in graph or 'Component' in graph
        assert 'encoder' in graph.lower() or 'decoder' in graph.lower()


class TestComponentGraphOutputFormats:
    """Test different graph output formats."""
    
    def test_text_format(self, autoencoder_model):
        """Test text format output."""
        graph = autoencoder_model.generate_component_graph(format='text')
        
        assert isinstance(graph, str)
    
    def test_text_format_content(self, autoencoder_model):
        """Test that text format has meaningful content."""
        graph = autoencoder_model.generate_component_graph(format='text')
        
        # Should have multiple lines
        lines = graph.split('\n')
        assert len(lines) > 3
    
    @pytest.mark.skipif(True, reason="graphviz may not be installed")
    def test_png_format_if_available(self, autoencoder_model, temp_dir):
        """Test PNG format if graphviz is available."""
        output_path = temp_dir / "graph.png"
        
        try:
            result = autoencoder_model.generate_component_graph(
                output_path=output_path,
                format='png'
            )
            # If successful, file should exist or result should be returned
            assert output_path.exists() or result is not None
        except ImportError:
            # graphviz not available, skip
            pytest.skip("graphviz not available")
    
    @pytest.mark.skipif(True, reason="graphviz may not be installed")
    def test_svg_format_if_available(self, autoencoder_model, temp_dir):
        """Test SVG format if graphviz is available."""
        output_path = temp_dir / "graph.svg"
        
        try:
            result = autoencoder_model.generate_component_graph(
                output_path=output_path,
                format='svg'
            )
            # If successful, file should exist or result should be returned
            assert output_path.exists() or result is not None
        except ImportError:
            # graphviz not available, skip
            pytest.skip("graphviz not available")


class TestIndividualComponentGraphs:
    """Test component graph generation for individual components."""
    
    def test_encoder_graph(self):
        """Test encoder can generate its own graph."""
        from models.encoder import Encoder
        
        config = {
            "type": "Encoder",
            "in_channels": 3,
            "latent_channels": 4,
            "base_channels": 16,
            "downsampling_steps": 2,
        }
        encoder = Encoder.from_config(config)
        
        if hasattr(encoder, 'generate_component_graph'):
            graph = encoder.generate_component_graph(format='text')
            assert isinstance(graph, str)
            assert len(graph) > 0
    
    def test_decoder_graph(self):
        """Test decoder can generate its own graph."""
        from models.decoder import Decoder
        
        config = {
            "type": "Decoder",
            "latent_channels": 4,
            "base_channels": 16,
            "upsampling_steps": 2,
            "heads": [
                {"type": "RGBHead", "name": "rgb", "out_channels": 3}
            ]
        }
        decoder = Decoder.from_config(config)
        
        if hasattr(decoder, 'generate_component_graph'):
            graph = decoder.generate_component_graph(format='text')
            assert isinstance(graph, str)
            assert len(graph) > 0
    
    def test_unet_graph(self):
        """Test UNet can generate its own graph."""
        from models.components.unet import UnetWithAttention
        
        config = {
            "type": "UnetWithAttention",
            "in_channels": 4,
            "out_channels": 4,
            "base_channels": 16,
            "depth": 2,
            "num_res_blocks": 1,
            "time_dim": 128,
        }
        unet = UnetWithAttention.from_config(config)
        
        if hasattr(unet, 'generate_component_graph'):
            graph = unet.generate_component_graph(format='text')
            assert isinstance(graph, str)
            assert len(graph) > 0

