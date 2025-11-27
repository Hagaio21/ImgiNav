#!/usr/bin/env python3
"""
Visualize the dataflow/component graph for a model, showing actual data flow paths.
"""

import sys
import argparse
from pathlib import Path
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.diffusion import DiffusionModel
from common.utils import load_config_with_profile


def _get_component_shape_info(component, batch_size=1):
    """Get shape information for a component procedurally."""
    shape_info = None
    if hasattr(component, 'get_shape_info'):
        try:
            shape_info = component.get_shape_info(batch_size=batch_size)
        except:
            pass
    
    if shape_info is None:
        # Try get_input_shape and get_output_shape
        in_shape = None
        out_shape = None
        if hasattr(component, 'get_input_shape'):
            try:
                in_shape = component.get_input_shape(batch_size=batch_size)
            except:
                pass
        if hasattr(component, 'get_output_shape'):
            try:
                out_shape = component.get_output_shape(batch_size=batch_size)
            except:
                pass
        
        if in_shape or out_shape:
            shape_info = {'input': in_shape, 'output': out_shape}
    
    return shape_info


def _format_shape(shape_info):
    """Format shape information for display."""
    if shape_info is None:
        return None
    
    if isinstance(shape_info, dict):
        # Try to get output shape first, then input
        if 'output' in shape_info and shape_info['output']:
            return _format_shape_value(shape_info['output'])
        elif 'input' in shape_info and shape_info['input']:
            return _format_shape_value(shape_info['input'])
        else:
            return None
    else:
        return _format_shape_value(shape_info)


def _format_shape_value(shape):
    """Format a single shape value."""
    if shape is None:
        return None
    
    if isinstance(shape, dict):
        # If it's a dict with multiple keys, format the first one
        if shape:
            first_key = next(iter(shape.keys()))
            first_shape = shape[first_key]
            if isinstance(first_shape, tuple):
                # Replace None with '?' and use 'B' for batch dimension
                formatted = []
                for i, s in enumerate(first_shape):
                    if s is None:
                        formatted.append('?' if i > 0 else 'B')
                    else:
                        formatted.append(str(s))
                return f"[{', '.join(formatted)}]"
            return str(first_shape)
        return None
    elif isinstance(shape, tuple):
        # Replace None with '?' and use 'B' for batch dimension
        formatted = []
        for i, s in enumerate(shape):
            if s is None:
                formatted.append('?' if i > 0 else 'B')
            else:
                formatted.append(str(s))
        return f"[{', '.join(formatted)}]"
    elif isinstance(shape, (list, str)):
        return str(shape)
    else:
        return None


def generate_dataflow_graph(model, output_path=None, format='png', include_shapes=True):
    """
    Generate a dataflow graph showing actual data flow through the model.
    
    Procedurally analyzes the model structure and forward pass to generate the graph.
    """
    try:
        from graphviz import Digraph
        use_graphviz = True
    except ImportError:
        if format not in ['text']:
            raise ImportError("graphviz not installed. Install with: pip install graphviz")
        use_graphviz = False
    
    import inspect
    import torch
    
    # Build dataflow graph structure procedurally
    nodes = {}
    edges = []
    
    model_name = model.__class__.__name__
    
    # Inspect forward method signature to get inputs procedurally
    forward_sig = inspect.signature(model.forward)
    forward_params = list(forward_sig.parameters.keys())
    
    # Get batch size from decoder if available
    batch_size = 1
    if hasattr(model, 'decoder') and model.decoder is not None:
        decoder_shape_info = _get_component_shape_info(model.decoder, batch_size)
        if decoder_shape_info and 'input' in decoder_shape_info:
            input_shape = decoder_shape_info['input']
            if isinstance(input_shape, dict) and 'latent' in input_shape:
                latent_shape = input_shape['latent']
                if isinstance(latent_shape, tuple) and len(latent_shape) > 0:
                    batch_size = latent_shape[0] if latent_shape[0] is not None else 1
    
    # Add input nodes procedurally based on forward signature
    for param_name in forward_params:
        if param_name == 'self':
            continue
        
        # Infer shape from component configs or defaults
        shape = None
        if param_name == 'latents':
            if hasattr(model, 'decoder'):
                decoder_shape = _get_component_shape_info(model.decoder, batch_size)
                if decoder_shape and 'input' in decoder_shape:
                    shape = _format_shape(decoder_shape['input'])
            if not shape:
                shape = '[B, 4, H, W]'
        elif param_name == 'text_emb':
            shape = '[B, 384]'
        elif param_name == 'pov_emb':
            shape = '[B, 512]'
        elif param_name == 't':
            shape = '[B]'
        elif param_name == 'noise':
            shape = '[B, 4, H, W]'
        
        label = param_name.replace('_', ' ').title()
        nodes[f'input_{param_name}'] = {
            'label': f'{label}\n({param_name})',
            'type': 'input',
            'shape': shape
        }
    
    # Get components procedurally from model
    # _component_names maps component object -> name
    components = {}
    if hasattr(model, '_component_names'):
        # Reverse the mapping: name -> component
        components = {name: comp for comp, name in model._component_names.items() if comp is not None}
    # Also check for direct component attributes (decoder, unet, etc.)
    for attr_name in ['decoder', 'unet', 'scheduler', 'embedding_projection', 'encoder']:
        if hasattr(model, attr_name):
            comp = getattr(model, attr_name)
            if comp is not None:
                components[attr_name] = comp
    
    # Add component nodes procedurally
    for comp_name, component in components.items():
        comp_type = component.__class__.__name__
        
        # Get shape info procedurally
        shape_info = _get_component_shape_info(component, batch_size)
        output_shape = _format_shape(shape_info)
        
        nodes[comp_name] = {
            'label': f'{comp_name.replace("_", " ").title()}\n({comp_type})',
            'type': 'component',
            'shape': output_shape,
            'component': component
        }
    
    # Build dataflow edges based on model type
    if model_name == 'DiffusionModel':
        # Embedding projection flow
        if 'embedding_projection' in nodes:
            if 'input_text_emb' in nodes:
                edges.append(('input_text_emb', 'embedding_projection', 'text_emb', 
                            nodes['input_text_emb'].get('shape')))
            if 'input_pov_emb' in nodes:
                edges.append(('input_pov_emb', 'embedding_projection', 'pov_emb',
                            nodes['input_pov_emb'].get('shape')))
            
            # Get output shape from component
            ep_shape = nodes['embedding_projection'].get('shape')
            if not ep_shape or ep_shape in ['None', 'N/A', None]:
                # Try to get from component's output shape
                if 'embedding_projection' in components:
                    ep_comp = components['embedding_projection']
                    ep_shape_info = _get_component_shape_info(ep_comp, batch_size)
                    ep_shape = _format_shape(ep_shape_info)
            
            # Always create conditioning_signal node and edge, even if shape is unknown
            # Try to infer shape from embedding_projection config
            if not ep_shape or ep_shape in ['None', 'N/A', None]:
                # Try to get from component config
                if 'embedding_projection' in components:
                    ep_comp = components['embedding_projection']
                    if hasattr(ep_comp, '_init_kwargs'):
                        output_channels = ep_comp._init_kwargs.get('output_channels', 48)
                        spatial_size = ep_comp._init_kwargs.get('spatial_size', [32, 32])
                        if isinstance(spatial_size, list) and len(spatial_size) >= 2:
                            ep_shape = f'[B, {output_channels}, {spatial_size[0]}, {spatial_size[1]}]'
                        else:
                            ep_shape = f'[B, {output_channels}, H, W]'
            
            # Create conditioning signal node
            nodes['conditioning_signal'] = {
                'label': 'Conditioning Signal',
                'type': 'data',
                'shape': ep_shape if ep_shape and ep_shape not in ['None', 'N/A', None] else '[B, 48, 32, 32]'
            }
            # Always create edge from embedding_projection to conditioning_signal
            cond_shape = nodes['conditioning_signal'].get('shape')
            edges.append(('embedding_projection', 'conditioning_signal', 'output', cond_shape))
        else:
            nodes['conditioning_signal'] = {
                'label': 'Conditioning Signal\n(None)',
                'type': 'data',
                'shape': None
            }
        
        # Scheduler flow
        if 'scheduler' in nodes:
            if 'input_latents' in nodes:
                edges.append(('input_latents', 'scheduler', 'latents',
                            nodes['input_latents'].get('shape')))
            if 'input_t' in nodes:
                edges.append(('input_t', 'scheduler', 't',
                            nodes['input_t'].get('shape')))
            
            # Noise generation
            noise_shape = nodes.get('input_latents', {}).get('shape', '[B, 4, H, W]')
            nodes['noise'] = {
                'label': 'Noise',
                'type': 'data',
                'shape': noise_shape
            }
            edges.append(('scheduler', 'noise', 'randn_like', noise_shape))
            edges.append(('noise', 'scheduler', 'noise', noise_shape))
            
            # Noisy latents output
            nodes['noisy_latents'] = {
                'label': 'Noisy Latents',
                'type': 'data',
                'shape': noise_shape
            }
            edges.append(('scheduler', 'noisy_latents', 'add_noise', noise_shape))
        
        # UNet flow
        if 'unet' in nodes:
            if 'noisy_latents' in nodes:
                edges.append(('noisy_latents', 'unet', 'x_t',
                            nodes['noisy_latents'].get('shape')))
            if 'input_t' in nodes:
                edges.append(('input_t', 'unet', 't',
                            nodes['input_t'].get('shape')))
            # Always connect conditioning_signal to unet if it exists
            if 'conditioning_signal' in nodes:
                cond_shape = nodes['conditioning_signal'].get('shape')
                # Connect even if shape is None (for unconditional models)
                edges.append(('conditioning_signal', 'unet', 'conditioning', 
                            cond_shape if cond_shape and cond_shape not in ['None', 'N/A', None] else None))
            
            # Get UNet output shape - try to get from component
            unet_shape = nodes['unet'].get('shape')
            if not unet_shape or unet_shape in ['None', 'N/A', None]:
                # Try to get from component's output shape
                if 'unet' in components:
                    unet_comp = components['unet']
                    unet_shape_info = _get_component_shape_info(unet_comp, batch_size)
                    unet_shape = _format_shape(unet_shape_info)
                if not unet_shape or unet_shape in ['None', 'N/A', None]:
                    unet_shape = '[B, 4, H, W]'
            
            nodes['pred_noise'] = {
                'label': 'Predicted Noise',
                'type': 'output',
                'shape': unet_shape
            }
            edges.append(('unet', 'pred_noise', 'output', unet_shape))
            
            # Predicted latent (computed from scheduler + unet output)
            nodes['pred_latent'] = {
                'label': 'Predicted Latent',
                'type': 'output',
                'shape': unet_shape
            }
            if 'noisy_latents' in nodes:
                edges.append(('noisy_latents', 'pred_latent', 'input',
                            nodes['noisy_latents'].get('shape')))
            edges.append(('pred_noise', 'pred_latent', 'pred_noise', unet_shape))
            if 'input_t' in nodes:
                edges.append(('input_t', 'pred_latent', 't',
                            nodes['input_t'].get('shape')))
        
        # Decoder flow
        if 'decoder' in nodes:
            if 'pred_latent' in nodes:
                latent_shape = nodes['pred_latent'].get('shape')
                nodes['final_latents'] = {
                    'label': 'Final Latents',
                    'type': 'data',
                    'shape': latent_shape
                }
                edges.append(('pred_latent', 'final_latents', 'clamped', latent_shape))
                edges.append(('final_latents', 'decoder', 'latent', latent_shape))
            
            # Get decoder output shape - try to get from component
            decoder_shape = nodes['decoder'].get('shape')
            if not decoder_shape or decoder_shape in ['None', 'N/A', None]:
                # Try to get from component's output shape
                if 'decoder' in components:
                    decoder_comp = components['decoder']
                    decoder_shape_info = _get_component_shape_info(decoder_comp, batch_size)
                    decoder_shape = _format_shape(decoder_shape_info)
                if not decoder_shape or decoder_shape in ['None', 'N/A', None]:
                    decoder_shape = '[B, 3, 512, 512]'
            
            nodes['output_rgb'] = {
                'label': 'Output RGB',
                'type': 'output',
                'shape': decoder_shape
            }
            edges.append(('decoder', 'output_rgb', 'rgb', decoder_shape))
    
    # Generate visualization
    if use_graphviz and format != 'text':
        return _create_dataflow_graphviz(nodes, edges, output_path, format, include_shapes)
    else:
        return _create_dataflow_text(nodes, edges, include_shapes)


def _create_dataflow_graphviz(nodes, edges, output_path, format, include_shapes):
    """Create Graphviz visualization of dataflow."""
    from graphviz import Digraph
    
    graph = Digraph(comment='Dataflow Graph')
    # Use vertical layout for better readability, larger size
    graph.attr(rankdir='TB', size='20,30', ratio='auto', nodesep='1.0', ranksep='1.5')
    graph.attr('node', shape='box', style='rounded', fontsize='16', fontname='Arial')
    graph.attr('edge', fontsize='12', fontname='Arial')
    
    # Color scheme
    input_color = 'lightblue'
    component_color = 'lightgreen'
    data_color = 'lightyellow'
    output_color = 'lightcoral'
    
    # Helper function to create node with proper styling
    def _add_node(g, node_id, node_data, input_color, component_color, data_color, output_color, include_shapes, format):
        label = node_data['label']
        node_type = node_data['type']
        shape_info = node_data.get('shape', '')
        
        # For better readability, put shape on separate line or simplify
        # Skip N/A, None, or empty shapes
        if include_shapes and shape_info and shape_info not in ['N/A', 'None', None, '']:
            # Use HTML-like labels for better formatting
            if format in ['svg', 'pdf']:
                label = f"{label}\\n<font point-size='10'>{shape_info}</font>"
            else:
                label = f"{label}\\n{shape_info}"
        
        if node_type == 'input':
            g.node(node_id, label, shape='ellipse', style='filled', fillcolor=input_color)
        elif node_type == 'component':
            g.node(node_id, label, shape='box', style='filled', fillcolor=component_color)
        elif node_type == 'data':
            g.node(node_id, label, shape='diamond', style='filled', fillcolor=data_color)
        elif node_type == 'output':
            g.node(node_id, label, shape='ellipse', style='filled', fillcolor=output_color)
    
    # Identify conditioning path nodes to group in a cluster
    conditioning_nodes = set()
    if 'input_text_emb' in nodes:
        conditioning_nodes.add('input_text_emb')
    if 'input_pov_emb' in nodes:
        conditioning_nodes.add('input_pov_emb')
    if 'embedding_projection' in nodes:
        conditioning_nodes.add('embedding_projection')
    if 'conditioning_signal' in nodes:
        conditioning_nodes.add('conditioning_signal')
    
    # Create a cluster for conditioning path to keep it visually grouped
    if len(conditioning_nodes) > 1:
        with graph.subgraph(name='cluster_conditioning') as cond_cluster:
            cond_cluster.attr(label='Conditioning Path', style='rounded', color='lightgray', fontsize='14', fontname='Arial')
            for node_id in conditioning_nodes:
                if node_id in nodes:
                    _add_node(cond_cluster, node_id, nodes[node_id], input_color, component_color, data_color, output_color, include_shapes, format)
    
    # Add non-cluster nodes to main graph
    for node_id, node_data in nodes.items():
        if node_id not in conditioning_nodes:
            _add_node(graph, node_id, node_data, input_color, component_color, data_color, output_color, include_shapes, format)
    
    # Add edges with labels
    for edge in edges:
        if len(edge) < 3:
            continue
        source, target, edge_label = edge[0], edge[1], edge[2]
        shape_info = edge[3] if len(edge) > 3 else None
        
        if source not in nodes or target not in nodes:
            continue
        
        # Create edge label - simplify for readability
        # Skip N/A, None, or empty shapes
        edge_label_text = edge_label
        if include_shapes and shape_info and shape_info not in ['N/A', 'None', None, '']:
            if format != 'png':  # Only add shapes for SVG/PDF
                edge_label_text = f"{edge_label}\n{shape_info}"
            # For PNG, use shorter labels (just the edge label)
        
        # Different edge styles
        edge_style = 'solid'
        edge_color = 'black'
        if nodes[source]['type'] == 'input':
            edge_color = 'blue'
        elif nodes[target]['type'] == 'output':
            edge_color = 'red'
        
        graph.edge(source, target, label=edge_label_text, style=edge_style, color=edge_color, fontsize='12')
    
    # Render graph with high DPI for PNG
    if output_path:
        if format == 'png':
            # Use higher DPI for PNG to improve resolution
            graph.render(output_path, format=format, cleanup=True, engine='dot')
            # Try to re-render with explicit DPI if possible
            import subprocess
            dot_file = f"{output_path}.gv"
            if Path(dot_file).exists():
                try:
                    # Render with explicit DPI using dot command
                    subprocess.run(
                        ['dot', '-Tpng', f'-Gdpi=300', f'-o{output_path}.png', dot_file],
                        check=False,  # Don't fail if command not available
                        capture_output=True
                    )
                except:
                    pass  # Fallback to default rendering
        else:
            graph.render(output_path, format=format, cleanup=True)
        print(f"Dataflow graph saved to {output_path}.{format}")
    
    return graph


def _create_dataflow_text(nodes, edges, include_shapes):
    """Create text-based dataflow representation."""
    lines = []
    lines.append("=" * 80)
    lines.append("Dataflow Graph: DiffusionModel")
    lines.append("=" * 80)
    lines.append("")
    
    # Group by type
    inputs = {k: v for k, v in nodes.items() if v['type'] == 'input'}
    components = {k: v for k, v in nodes.items() if v['type'] == 'component'}
    data_nodes = {k: v for k, v in nodes.items() if v['type'] == 'data'}
    outputs = {k: v for k, v in nodes.items() if v['type'] == 'output'}
    
    lines.append("INPUTS:")
    lines.append("-" * 80)
    for name, data in inputs.items():
        shape = data.get('shape', '')
        lines.append(f"  {name}: {data['label']}")
        if include_shapes and shape:
            lines.append(f"    Shape: {shape}")
    
    lines.append("")
    lines.append("DATA FLOW:")
    lines.append("-" * 80)
    for edge in edges:
        if len(edge) < 3:
            continue
        source, target, label = edge[0], edge[1], edge[2]
        shape = edge[3] if len(edge) > 3 else ''
        
        arrow = "───" if nodes[source]['type'] == 'component' else "→"
        flow_line = f"  {source} {arrow} [{label}] {arrow} {target}"
        if include_shapes and shape:
            flow_line += f" ({shape})"
        lines.append(flow_line)
    
    lines.append("")
    lines.append("COMPONENTS:")
    lines.append("-" * 80)
    for name, data in components.items():
        lines.append(f"  {name}: {data['label']}")
        if include_shapes and data.get('shape'):
            lines.append(f"    Output shape: {data['shape']}")
    
    lines.append("")
    lines.append("OUTPUTS:")
    lines.append("-" * 80)
    for name, data in outputs.items():
        shape = data.get('shape', '')
        lines.append(f"  {name}: {data['label']}")
        if include_shapes and shape:
            lines.append(f"    Shape: {shape}")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataflow graph visualization showing actual data flow"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/diffusion/clip/regular_rooms/small_down_bottleneck_text_only.yaml",
        help="Path to experiment config"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (optional, for loading model state)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/dataflow_graph",
        help="Output path for graph (without extension)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "svg", "pdf", "dot", "text"],
        default="png",
        help="Output format"
    )
    parser.add_argument(
        "--no-shapes",
        action="store_true",
        help="Don't include shape information"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Generating Dataflow Graph")
    print("="*60)
    
    # Load config
    print(f"\nLoading config: {args.config}")
    config = load_config_with_profile(args.config)
    print("✓ Config loaded")
    
    # Build model
    print("\nBuilding model from config...")
    model = DiffusionModel.from_config(config)
    print("✓ Model built")
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        import torch
        payload = torch.load(args.checkpoint, map_location="cpu")
        state_dict = payload.get("state_dict", payload)
        model.load_state_dict(state_dict, strict=False)
        print("✓ Checkpoint loaded")
    
    # Generate dataflow graph
    print(f"\nGenerating dataflow graph...")
    print(f"  Format: {args.format}")
    print(f"  Include shapes: {not args.no_shapes}")
    
    try:
        graph = generate_dataflow_graph(
            model,
            output_path=args.output,
            format=args.format,
            include_shapes=not args.no_shapes
        )
        
        if args.format == "text":
            print("\n" + "="*60)
            print("Dataflow Graph Representation:")
            print("="*60)
            print(graph)
        else:
            print(f"\n✓ Dataflow graph saved to: {args.output}.{args.format}")
            
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo generate visual graphs, install graphviz:")
        print("  pip install graphviz")
        print("\nAlso install the graphviz system package:")
        print("  Windows: Download from https://graphviz.org/download/")
        print("  Linux: sudo apt-get install graphviz")
        print("  Mac: brew install graphviz")
        print("\nGenerating text representation instead...")
        graph = generate_dataflow_graph(
            model,
            output_path=None,
            format="text",
            include_shapes=not args.no_shapes
        )
        print("\n" + "="*60)
        print("Dataflow Graph Representation:")
        print("="*60)
        print(graph)
    
    print("\n" + "="*60)
    print("Dataflow Graph Generation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
