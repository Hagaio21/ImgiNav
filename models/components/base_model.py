"""
BaseModel - Subclass of BaseComponent for trainable full models.
Provides extended checkpointing capabilities for training state.
"""
import torch
from pathlib import Path
from .base_component import BaseComponent


class BaseModel(BaseComponent):
    """
    Base class for full trainable models (e.g., Autoencoder, DiffusionModel).
    
    Extends BaseComponent with model-specific functionality.
    All checkpoint methods from BaseComponent are inherited and work as-is.
    Subclasses can override for extended checkpointing (optimizer, epoch, etc.).
    """
    
    def __init__(self, **kwargs):
        # Initialize component tracking before calling super (which calls _build)
        # Use object.__setattr__ to avoid triggering __setattr__ during initialization
        object.__setattr__(self, '_component_names', {})  # component -> name mapping
        super().__init__(**kwargs)
    
    def save_checkpoint(self, path, include_config=True, **extra_state):
        """
        Save model checkpoint with optional extra training state.
        
        Args:
            path: Path to save checkpoint
            include_config: Whether to include model config
            **extra_state: Additional state to save (e.g., optimizer, epoch, etc.)
        """
        path = Path(path)
        payload = {"state_dict": self.state_dict()}
        if include_config:
            payload["config"] = self.to_config()
        payload.update(extra_state)
        torch.save(payload, path)
    
    @classmethod
    def load_checkpoint(cls, path, map_location="cpu", return_extra=False, config=None, strict=True):
        """
        Load model checkpoint, optionally returning extra state.
        
        Args:
            path: Path to checkpoint
            map_location: Device to load on
            return_extra: If True, return tuple (model, extra_state_dict)
            config: Optional config dict to use instead of saved config (useful when resuming training)
            strict: If True, require exact match. If False, filter out mismatched keys (default: True)
            
        Returns:
            If return_extra=False: just the model (backward compatible)
            If return_extra=True: (model, extra_state_dict) tuple where 
                extra_state_dict contains any additional state (optimizer, epoch, etc.)
        """
        path = Path(path)
        payload = torch.load(path, map_location=map_location)
        
        # Use provided config if available, otherwise use saved config
        model_config = config if config is not None else payload.get("config")
        model = cls.from_config(model_config) if model_config else cls()
        
        # Load state dict
        state_dict = payload["state_dict"]
        if strict:
            model.load_state_dict(state_dict)
        else:
            # Filter state dict to only include keys that match in shape
            # Also skip CLIP projection keys if they don't match (they're not needed for encoding)
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            skipped_keys = []
            clip_proj_keys = []
            
            for key, value in state_dict.items():
                # Skip CLIP projection keys if they don't match (optional for encoding)
                is_clip_proj = key.startswith("clip_projections.")
                
                if key in model_state_dict:
                    # Check if shapes match
                    if model_state_dict[key].shape == value.shape:
                        filtered_state_dict[key] = value
                    else:
                        if is_clip_proj:
                            clip_proj_keys.append(f"{key} (shape mismatch: {value.shape} vs {model_state_dict[key].shape})")
                        else:
                            skipped_keys.append(f"{key} (shape mismatch: {value.shape} vs {model_state_dict[key].shape})")
                else:
                    # Key doesn't exist in model
                    if is_clip_proj:
                        clip_proj_keys.append(f"{key} (not in model)")
                    else:
                        skipped_keys.append(f"{key} (not in model)")
            
            if clip_proj_keys:
                # CLIP projection mismatches are expected when loading for encoding (not needed)
                import warnings
                warnings.warn(
                    f"Skipping {len(clip_proj_keys)} CLIP projection keys (not needed for encoding). "
                    f"This is normal when loading spatial CLIP VAE checkpoints for embedding."
                )
            
            if skipped_keys:
                import warnings
                warnings.warn(
                    f"Skipping {len(skipped_keys)} keys when loading checkpoint (strict=False). "
                    f"First few: {skipped_keys[:5]}"
                )
            
            # Load filtered state dict
            model.load_state_dict(filtered_state_dict, strict=False)
        
        if return_extra:
            # Return model and any extra state (optimizer, epoch, etc.)
            extra_state = {k: v for k, v in payload.items() 
                          if k not in ["state_dict", "config"]}
            return model, extra_state
        
        return model
    
    def _write_model_statistics(self):
        """Write model statistics to file if save_path is available."""
        if not hasattr(self, 'save_path') or not self.save_path:
            return
        
        stats = self._get_component_statistics()
        if not stats:
            return
        
        save_path = Path(self.save_path)
        stats_file = save_path.parent / f"{save_path.stem}_statistics.txt"
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(stats_file, 'w') as f:
            f.write(f"Model Statistics: {self.__class__.__name__}\n")
            f.write("=" * 60 + "\n\n")
            
            totals = {"trainable": 0, "frozen": 0, "total": 0}
            for component_name, component_stats in stats.items():
                trainable = component_stats.get("trainable", 0)
                frozen = component_stats.get("frozen", 0)
                total = component_stats.get("total", 0)
                label = component_stats.get("label", component_name)
                shapes = component_stats.get("shapes")
                
                totals["trainable"] += trainable
                totals["frozen"] += frozen
                totals["total"] += total
                
                f.write(f"{label}:\n")
                f.write(f"  Trainable: {trainable:,}\n")
                f.write(f"  Frozen: {frozen:,}\n")
                f.write(f"  Total: {total:,}\n")
                
                # Add shape information if available
                if shapes:
                    f.write(f"  Input Shape: {shapes.get('input', 'N/A')}\n")
                    f.write(f"  Output Shape: {shapes.get('output', 'N/A')}\n")
                
                f.write("\n")
            
            f.write("=" * 60 + "\n")
            f.write(f"Total Trainable: {totals['trainable']:,}\n")
            f.write(f"Total Frozen: {totals['frozen']:,}\n")
            f.write(f"Total Parameters: {totals['total']:,}\n")
    
    def _get_component_statistics(self):
        """
        Get parameter statistics for model components.
        Must be implemented by subclasses.
        
        Returns:
            Dictionary mapping component names to statistics dicts with keys:
            - trainable: Number of trainable parameters
            - frozen: Number of frozen parameters
            - total: Total number of parameters
            - label: Display label for the component
            - shapes: Input/output shape information (if available)
        """
        raise NotImplementedError("Subclasses must implement _get_component_statistics()")
    
    def get_component_shapes(self, batch_size=1):
        """
        Get input/output shapes for all tracked components.
        
        Args:
            batch_size: Batch size for shape inference (default: 1)
        
        Returns:
            Dictionary mapping component names to shape information dicts
        """
        shapes = {}
        for component, name in self._component_names.items():
            if component is not None and hasattr(component, 'get_shape_info'):
                shapes[name] = component.get_shape_info(batch_size=batch_size)
        return shapes
    
    def _get_module_statistics(self, module, label=None, include_shapes=True):
        """
        Calculate parameter statistics for a single module.
        
        Args:
            module: The module to analyze (can be None)
            label: Optional display label (default: uses module name or "Unknown")
            include_shapes: If True, include input/output shape information
        
        Returns:
            Dictionary with keys: trainable, frozen, total, label, shapes (if include_shapes=True)
            Returns None if module is None
        """
        if module is None:
            return None
        
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in module.parameters())
        frozen = total - trainable
        
        if label is None:
            # Try to get a meaningful name
            if hasattr(module, '__class__'):
                label = module.__class__.__name__
            else:
                label = "Unknown"
        
        stats = {
            "trainable": trainable,
            "total": total,
            "frozen": frozen,
            "label": label
        }
        
        # Add shape information if requested and available
        if include_shapes and hasattr(module, 'get_shape_info'):
            shapes = module.get_shape_info(batch_size=1)
            stats["shapes"] = shapes
        
        return stats
    
    def _create_component_with_freeze(self, key, default_type=None, auto_freeze=False):
        """Create component from config and optionally freeze it."""
        component = self.create_component(key, default_type=default_type)
        if component is not None and auto_freeze:
            component_cfg = self._init_kwargs.get(key)
            if isinstance(component_cfg, dict) and component_cfg.get("frozen", False):
                component.freeze()
        return component
    
    def _ensure_component_type(self, key, default_type):
        """
        Ensure a component config has the specified type set.
        
        Args:
            key: Key in _init_kwargs for the component config
            default_type: Type to set if not already specified
        
        Returns:
            True if the config was modified, False otherwise
        """
        component_cfg = self._init_kwargs.get(key)
        if isinstance(component_cfg, dict) and "type" not in component_cfg:
            component_cfg = component_cfg.copy()
            component_cfg["type"] = default_type
            self._init_kwargs[key] = component_cfg
            return True
        return False
    
    def _validate_component_type(self, component, expected_class, component_name):
        """
        Validate that a component is an instance of the expected class.
        
        Args:
            component: The component to validate
            expected_class: The expected class (not a string, the actual class)
            component_name: Name of the component for error messages
        
        Raises:
            ValueError: If component is not an instance of expected_class
        """
        if not isinstance(component, expected_class):
            raise ValueError(
                f"{self.__class__.__name__} requires a {expected_class.__name__}. "
                f"Current {component_name} type: {component.__class__.__name__}"
            )
    
    
    def _add_component_to_config(self, cfg, component_name, config_key=None, condition=None):
        """Add component to config if it exists and condition is met."""
        config_key = config_key or component_name
        if not hasattr(self, component_name):
            return cfg
        
        component = getattr(self, component_name)
        should_include = (
            component is not None if condition is None
            else condition(component) if callable(condition)
            else bool(condition)
        )
        
        if should_include:
            cfg[config_key] = component.to_config() if hasattr(component, 'to_config') else component
        return cfg
    
    def add_component(self, name, component):
        """Add component to model and track it for config generation."""
        setattr(self, name, component)
        if component is not None:
            self._component_names[component] = name
    
    def _components_to_config(self, cfg):
        """Add all tracked components to config."""
        for component, name in self._component_names.items():
            if component is not None and hasattr(component, 'to_config'):
                cfg[name] = component.to_config()
        return cfg
    
    def _setup_projection(self, component_key, default_type=None):
        """Setup optional projection component."""
        component = self.create_component(component_key, default_type=default_type)
        if component is not None:
            self.add_component(component_key, component)
        return component
    
    def save_component_checkpoint(self, component_name, path, include_config=True):
        """
        Save a specific component as a separate checkpoint.
        
        Args:
            component_name: Name of the component attribute to save
            path: Path to save checkpoint
            include_config: Whether to include component config
        """
        if not hasattr(self, component_name):
            raise ValueError(f"Component '{component_name}' not found in {self.__class__.__name__}")
        
        component = getattr(self, component_name)
        if component is None:
            raise ValueError(f"Component '{component_name}' is None")
        
        if hasattr(component, 'save_checkpoint'):
            component.save_checkpoint(path, include_config=include_config)
        else:
            raise ValueError(f"Component '{component_name}' does not support checkpoint saving")
    
    def save_all_components(self, base_path, include_config=True):
        """
        Save all tracked components as separate checkpoints.
        
        Args:
            base_path: Base directory or path prefix for saving components
            include_config: Whether to include component configs
        
        Returns:
            Dict mapping component names to saved paths
        """
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        for component, name in self._component_names.items():
            if component is not None and hasattr(component, 'save_checkpoint'):
                component_path = base_path / f"{name}.pt"
                component.save_checkpoint(component_path, include_config=include_config)
                saved_paths[name] = str(component_path)
        
        return saved_paths
    
    @classmethod
    def load_component_checkpoint(cls, component_name, path, map_location="cpu"):
        """
        Load a component from a checkpoint.
        
        Args:
            component_name: Name/type of component to load (used to determine class)
            path: Path to component checkpoint
            map_location: Device to load on
        
        Returns:
            Loaded component instance
        """
        from .registry import create_component
        
        # Load checkpoint to get config
        payload = torch.load(path, map_location=map_location)
        config = payload.get("config")
        
        if config is None:
            raise ValueError(f"Component checkpoint at {path} has no config")
        
        # Create component from config
        component = create_component(config)
        
        # Load state dict
        state_dict = payload.get("state_dict", payload)
        component.load_state_dict(state_dict, strict=False)
        
        return component
    
    # -----------------------
    # Component Graph Visualization
    # -----------------------
    def generate_component_graph(self, output_path=None, format='png', include_shapes=True):
        """
        Generate a visual graph showing component interactions.
        
        Args:
            output_path: Path to save graph (if None, returns graph object)
            format: Output format ('png', 'svg', 'pdf', 'dot', or 'text')
            include_shapes: If True, include shape information on edges
        
        Returns:
            Graphviz object or text representation
        """
        try:
            from graphviz import Digraph
            use_graphviz = True
        except ImportError:
            if format not in ['text']:
                raise ImportError("graphviz not installed. Install with: pip install graphviz")
            use_graphviz = False
        
        # Build graph structure
        nodes = {}
        edges = []
        
        # Add model as root node
        model_name = self.__class__.__name__
        nodes[model_name] = {
            'label': model_name,
            'type': 'model',
            'components': []
        }
        
        # Add all tracked components as nodes
        for component, name in self._component_names.items():
            if component is None:
                continue
            
            comp_type = component.__class__.__name__
            label = f"{name}\n({comp_type})"
            
            nodes[name] = {
                'label': label,
                'type': 'component',
                'component': component
            }
            nodes[model_name]['components'].append(name)
            
            # Add edge from model to component
            edges.append((model_name, name, 'contains', None))
            
            # Try to detect sub-components
            if hasattr(component, '_component_names'):
                # Component has its own sub-components
                for sub_comp, sub_name in component._component_names.items():
                    if sub_comp is None:
                        continue
                    sub_full_name = f"{name}.{sub_name}"
                    sub_type = sub_comp.__class__.__name__
                    nodes[sub_full_name] = {
                        'label': f"{sub_name}\n({sub_type})",
                        'type': 'subcomponent',
                        'component': sub_comp
                    }
                    edges.append((name, sub_full_name, 'contains', None))
        
        # Generate visualization
        if use_graphviz and format != 'text':
            return self._create_graphviz_graph(nodes, edges, output_path, format, include_shapes)
        else:
            return self._create_text_graph(nodes, edges, include_shapes)
    
    def _create_graphviz_graph(self, nodes, edges, output_path, format, include_shapes):
        """Create Graphviz visualization."""
        from graphviz import Digraph
        
        graph = Digraph(comment='Component Interaction Graph')
        graph.attr(rankdir='TB', size='12,8')
        graph.attr('node', shape='box', style='rounded')
        
        # Add nodes with different styles
        for node_id, node_data in nodes.items():
            label = node_data['label']
            node_type = node_data['type']
            
            if node_type == 'model':
                graph.node(node_id, label, shape='ellipse', style='filled', fillcolor='lightblue')
            elif node_type == 'component':
                graph.node(node_id, label, shape='box', style='filled', fillcolor='lightgreen')
            else:  # subcomponent
                graph.node(node_id, label, shape='box', style='filled', fillcolor='lightyellow')
        
        # Add edges
        for edge in edges:
            if len(edge) < 3:
                continue
            source, target, interaction_type = edge[0], edge[1], edge[2]
            shape_info = edge[3] if len(edge) > 3 else None
            
            if source not in nodes or target not in nodes:
                continue
            
            # Create edge label
            edge_label = interaction_type
            
            # Different edge styles for different interaction types
            edge_style = 'solid'
            edge_color = 'black'
            if interaction_type == 'contains':
                edge_style = 'dashed'
                edge_color = 'gray'
            
            graph.edge(source, target, label=edge_label, style=edge_style, color=edge_color)
        
        # Render graph
        if output_path:
            graph.render(output_path, format=format, cleanup=True)
            print(f"Graph saved to {output_path}.{format}")
        
        return graph
    
    def _create_text_graph(self, nodes, edges, include_shapes):
        """Create text-based graph representation."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"Component Interaction Graph: {self.__class__.__name__}")
        lines.append("=" * 60)
        lines.append("")
        
        # Group edges by type
        contains_edges = [e for e in edges if len(e) > 2 and e[2] == 'contains']
        
        lines.append("Component Hierarchy:")
        lines.append("-" * 60)
        for edge in contains_edges:
            source, target = edge[0], edge[1]
            lines.append(f"  {source} ──contains──> {target}")
        
        # Add component details if available
        if include_shapes:
            lines.append("")
            lines.append("Component Details:")
            lines.append("-" * 60)
            for name, node_data in nodes.items():
                if node_data.get('type') == 'component':
                    component = node_data.get('component')
                    if component is not None:
                        if hasattr(component, 'get_input_shape'):
                            in_shape = component.get_input_shape(batch_size=1)
                            out_shape = component.get_output_shape(batch_size=1) if hasattr(component, 'get_output_shape') else None
                            lines.append(f"  {name}:")
                            lines.append(f"    Input:  {in_shape}")
                            if out_shape:
                                lines.append(f"    Output: {out_shape}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)

