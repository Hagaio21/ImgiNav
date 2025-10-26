#!/usr/bin/env python
# generate_embeddings.py
"""
Generates and saves embeddings from multiple autoencoder models.
This script loads models, encodes layouts, applies UMAP, and saves the results
as pickle files for later analysis.
"""
import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
import argparse
import json
import pickle
import umap.umap_ as umap
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as T
from tqdm import tqdm


def extract_model_info(model_name):
    """Extract model information from the model name.
    Expected format: VAE_512_NxNxC_LossType[_params]
    Example: VAE_512_64x64x4_MSE or VAE_512_32x32x2_SegLoss_HighSeg
    """
    parts = model_name.split('_')
    info = {
        'model_type': 'Unknown',
        'input_size': None,
        'latent_shape': None,
        'loss_type': 'Unknown',
        'params': ''
    }
    
    try:
        # Model type (VAE)
        if len(parts) > 0:
            info['model_type'] = parts[0]
        
        # Input size (512)
        if len(parts) > 1 and parts[1].isdigit():
            info['input_size'] = int(parts[1])
        
        # Latent dimensions (NxNxC)
        if len(parts) > 2 and 'x' in parts[2]:
            dims = parts[2].split('x')
            if len(dims) == 3 and all(d.isdigit() for d in dims):
                h, w, c = int(dims[0]), int(dims[1]), int(dims[2])
                info['latent_shape'] = (c, h, w)  # Convert to (C, H, W) format
                info['latent_dims'] = c * h * w
                info['compression_ratio'] = (info['input_size'] ** 2 * 3) / info['latent_dims'] if info['input_size'] else None
        
        # Loss type
        if len(parts) > 3:
            info['loss_type'] = parts[3]
        
        # Additional parameters
        if len(parts) > 4:
            info['params'] = '_'.join(parts[4:])
    
    except Exception as e:
        print(f"Warning: Could not parse model name '{model_name}': {e}")
    
    return info


def load_model_from_experiment(config_path, checkpoint_path, device="cuda"):
    """Loads a VAE model from its experiment config and checkpoint file."""
    print(f"\nLoading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if 'model' key exists, provide helpful error if not
    if 'model' not in config:
        raise KeyError(f"Error: 'model' key not found in config file: {config_path}. "
                       "Make sure you are using the 'experiment_config.yaml', not 'autoencoder_config.yaml'.")
    
    model_cfg = config['model']
    dataset_cfg = config['dataset']
    
    # Re-create model architecture from config
    model = AutoEncoder.from_shape(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        base_channels=model_cfg["base_channels"],
        latent_channels=model_cfg["latent_channels"],
        image_size=model_cfg["image_size"],
        latent_base=model_cfg["latent_base"],
        norm=model_cfg.get("norm"),
        act=model_cfg.get("act", "relu"),
        dropout=model_cfg.get("dropout", 0.0),
        num_classes=model_cfg.get("num_classes", None),
    )
    
    print(f"Loading weights from {checkpoint_path}")
    
    # Load state_dict into memory first
    state_dict = torch.load(checkpoint_path, map_location=device)

    # --- HACK for backward-compatibility ---
    if "decoder.final.weight" in state_dict:
        print("INFO: Detected old model checkpoint. Renaming 'decoder.final' -> 'decoder.final_rgb'.")
        state_dict["decoder.final_rgb.weight"] = state_dict.pop("decoder.final.weight")
        state_dict["decoder.final_rgb.bias"] = state_dict.pop("decoder.final.bias")
    # --- End of HACK ---
    
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"WARNING: Strict loading failed. Trying non-strict. Error: {e}")
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval() # Set model to evaluation mode
    
    # Try to get a meaningful name from the path
    try:
        # Assumes path like .../experiments/VAE/MODEL_NAME/config.yaml
        model_name = os.path.basename(os.path.dirname(config_path))

        model_name = os.path.basename(os.path.dirname(os.path.dirname(config_path)))

    except Exception:
         model_name = os.path.basename(config_path).replace(".yaml", "") # Fallback name
    print(f"Model '{model_name}' loaded successfully.")
    
    return model, dataset_cfg, model_name


def load_dataset_and_sample(manifest_path, dataset_cfg, num_points=None):
    """Loads the full dataset and (optionally) samples it."""
    print("\n--- Loading and Sampling Dataset ---")
    transform = T.ToTensor()
    # Ensure LayoutDataset is globally available
    from modules.datasets import LayoutDataset
    
    dataset = LayoutDataset(
        manifest_path=manifest_path,
        transform=transform,
        mode="all",
        one_hot=dataset_cfg.get("one_hot", False),
        taxonomy_path=dataset_cfg.get("taxonomy_path"),
    )
    
    # --- Load Taxonomy for ID translation ---
    taxonomy_path = dataset_cfg.get("taxonomy_path")
    if not taxonomy_path or not os.path.exists(taxonomy_path):
        print(f"Warning: Taxonomy path not found at {taxonomy_path}. Cannot translate room IDs.")
        id2room_map = {}
    else:
        with open(taxonomy_path, 'r') as f:
            id2room_map = json.load(f).get("id2room", {})

    # --- Sub-sample dataset if num_points is provided ---
    if num_points is not None and num_points > 0 and num_points < len(dataset):
        print(f"Sampling {num_points} random points from the total {len(dataset)} layouts.")
        # Set a random seed to make the sampling reproducible across runs
        np.random.seed(42) 
        indices = np.random.choice(len(dataset), num_points, replace=False)
        dataset = Subset(dataset, indices)
    
    print(f"Using {len(dataset)} layouts for all models.")
    return dataset, id2room_map


def encode_dataset(model, dataset, dataset_cfg, label_col, category_col, filter_empty, device="cuda"):
    """Encodes all layouts from a dataset using the model."""
    
    # Ensure collate_fn is available
    from modules.datasets import collate_skip_none
    
    loader = DataLoader(
        dataset, 
        batch_size=dataset_cfg.get("batch_size", 64),
        shuffle=False, 
        num_workers=dataset_cfg.get("num_workers", 4),
        collate_fn=collate_skip_none
    )
    
    all_latents = []
    all_latents_spatial = []  # Keep spatial version too
    all_labels = []
    all_categories = []
    latent_shape = None
    
    print(f"Encoding {len(dataset)} layouts (filter_empty={filter_empty})...")
    
    for batch in tqdm(loader, desc="Encoding batches"):
        if batch is None:
            continue
            
        if 'layout' not in batch:
            print("Warning: Skipping batch missing 'layout' key.")
            continue
            
        x = batch['layout'].to(device)
        is_empty = batch.get('is_empty', [False] * x.size(0)) 
        
        if filter_empty:
            non_empty_mask = torch.tensor([not e for e in is_empty], dtype=torch.bool).to(device)
            
            if not non_empty_mask.any():
                continue 
                
            x = x[non_empty_mask]
            
            current_labels = batch.get(label_col, [])
            current_categories = batch.get(category_col, [])

            labels = [label for label, empty in zip(current_labels, is_empty) if not empty]
            categories = [cat for cat, empty in zip(current_categories, is_empty) if not empty]

            if len(labels) != x.size(0) and x.size(0) > 0:
                 print(f"Warning: Mismatch after filtering. Layouts: {x.size(0)}, Labels: {len(labels)}. Using placeholders.")
                 labels = ['Unknown'] * x.size(0)
            if len(categories) != x.size(0) and x.size(0) > 0:
                 print(f"Warning: Mismatch after filtering. Layouts: {x.size(0)}, Categories: {len(categories)}. Using placeholders.")
                 categories = ['Unknown'] * x.size(0)

        else:
            labels = batch.get(label_col, ['Unknown'] * x.size(0))
            categories = batch.get(category_col, ['Unknown'] * x.size(0))
        
        with torch.no_grad():
            mean, _ = model.encode(x)
            batch_latents = mean.cpu().numpy()
            
            # Store spatial version
            all_latents_spatial.append(batch_latents)
            
            # Store the shape from first batch
            if latent_shape is None:
                latent_shape = batch_latents.shape[1:]  # (C, H, W)
                print(f"Latent shape detected: {latent_shape}")
            
            # Flatten for UMAP and clustering metrics
            if batch_latents.ndim == 4:  # (B, C, H, W)
                B, C, H, W = batch_latents.shape
                batch_latents = batch_latents.reshape(B, -1)  # Flatten to (B, C*H*W)
            elif batch_latents.ndim == 3:  # (B, H, W)
                B, H, W = batch_latents.shape
                batch_latents = batch_latents.reshape(B, -1)  # Flatten to (B, H*W)
            elif batch_latents.ndim > 2:  # Any other high-dimensional case
                batch_latents = batch_latents.reshape(batch_latents.shape[0], -1)
            all_latents.append(batch_latents)
        
        all_labels.extend(labels)
        all_categories.extend(categories)
    
    if not all_latents:
        print("Error: No valid layouts found to encode.")
        return None, None, None, None
    
    all_latents = np.vstack(all_latents)
    all_latents_spatial = np.vstack(all_latents_spatial)
    print(f"Latent vectors shape: {all_latents.shape}")
    print(f"Spatial latent shape: {all_latents_spatial.shape}")
    
    return all_latents, all_latents_spatial, all_labels, all_categories


def apply_umap_transform(latents, n_neighbors=15, min_dist=0.1, metric='euclidean', 
                        random_state=42, suppress_warnings=False):
    """Apply UMAP dimensionality reduction to latent vectors."""
    print(f"Running UMAP on {latents.shape[0]} latent vectors of dimension {latents.shape[1]}...")
    
    # Suppress warnings if requested
    if suppress_warnings:
        import warnings
        warnings.filterwarnings("ignore", message="Graph is not fully connected")
        warnings.filterwarnings("ignore", message="n_jobs value .* overridden to 1 by setting random_state")
    
    # Verify the input shape
    if latents.ndim != 2:
        raise ValueError(f"Expected 2D array for UMAP, got shape {latents.shape}")
    
    # Adjust n_neighbors if we have fewer samples
    n_samples = latents.shape[0]
    n_neighbors_adjusted = min(n_neighbors, n_samples - 1)
    
    if n_neighbors_adjusted < n_neighbors:
        print(f"Adjusted n_neighbors from {n_neighbors} to {n_neighbors_adjusted} due to small sample size")
    
    reducer = umap.UMAP(
        n_components=2, 
        n_neighbors=n_neighbors_adjusted, 
        min_dist=min_dist,
        random_state=random_state,
        metric=metric,
        # Add these parameters to handle connectivity issues
        set_op_mix_ratio=1.0,
        local_connectivity=1,
        # Increase iterations for better convergence
        n_epochs=500
    )
    
    try:
        umap_embeddings = reducer.fit_transform(latents)
        print(f"UMAP completed successfully. Shape: {umap_embeddings.shape}")
    except Exception as e:
        print(f"Warning: UMAP failed with error: {e}")
        print("Attempting with more robust parameters...")
        
        # Fallback with more conservative parameters
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(5, n_samples - 1),
            min_dist=0.3,
            random_state=random_state,
            metric='euclidean',
            init='random',
            n_epochs=200
        )
        umap_embeddings = reducer.fit_transform(latents)
        print(f"UMAP completed with fallback parameters. Shape: {umap_embeddings.shape}")
    
    return umap_embeddings, reducer


def translate_room_ids(labels, id2room_map):
    """Translate room IDs to room names using the taxonomy mapping."""
    translated_labels = []
    for label in labels:
        if str(label) in id2room_map:
            translated_labels.append(id2room_map[str(label)])
        else:
            translated_labels.append(f"room_{label}")
    return translated_labels


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for multiple autoencoder models")
    parser.add_argument('--project_root', type=str, required=True, help="Path to the ImgiNav project root directory")
    parser.add_argument('--manifest', type=str, required=True, help="Path to the full manifest.csv file")
    parser.add_argument('--output_dir', type=str, default="embeddings", help="Directory to save the embeddings")
    
    # Metadata
    parser.add_argument('--label_col', type=str, required=True, help="Column name for room type (e.g., 'room_id')")
    parser.add_argument('--category_col', type=str, required=True, help="Column name for category (e.g., 'type')")
    parser.add_argument('--no_filter_empty', action='store_true', help="Set this flag to *disable* filtering of empty layouts")
    
    # Sampling
    parser.add_argument('--num_points', type=int, default=None, help="Number of random points to sample for embedding (default: all)")
    
    # Models
    parser.add_argument('--configs', type=str, nargs='+', required=True, help="List of paths to experiment_config.yaml files")
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True, help="List of paths to .pt checkpoint files (must match order of --configs)")
    
    # UMAP parameters
    parser.add_argument('--umap_neighbors', type=int, default=15, help="UMAP n_neighbors parameter")
    parser.add_argument('--umap_min_dist', type=float, default=0.1, help="UMAP min_dist parameter")
    parser.add_argument('--umap_metric', type=str, default='euclidean', help="UMAP distance metric")
    parser.add_argument('--suppress_warnings', action='store_true', help="Suppress UMAP connectivity warnings")

    args = parser.parse_args()

    # Validate Input
    if len(args.configs) != len(args.checkpoints):
        print("Error: Number of config files must match number of checkpoint files.")
        sys.exit(1)
    n_models = len(args.configs)
    if n_models == 0:
        print("Error: No models specified.")
        sys.exit(1)
    print(f"Received {n_models} models to process.")

    # Add project root to path
    sys.path.append(args.project_root)
    try:
        global AutoEncoder
        from modules.autoencoder import AutoEncoder
        global LayoutDataset, collate_skip_none
        from modules.datasets import LayoutDataset, collate_skip_none
    except ImportError:
        print(f"Error: Could not import modules from {args.project_root}")
        print("Please ensure --project_root points to the correct directory")
        sys.exit(1)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    filter_empty_layouts = not args.no_filter_empty

    # Load all models first
    models = []
    names = []
    dset_cfgs = []
    for i in range(n_models):
        config_path = args.configs[i]
        ckpt_path = args.checkpoints[i]
        model, dset_cfg, name = load_model_from_experiment(config_path, ckpt_path, DEVICE)
        models.append(model)
        names.append(name)
        dset_cfgs.append(dset_cfg)

    # Load dataset once
    first_dataset_cfg = dset_cfgs[0] if dset_cfgs else None
    if first_dataset_cfg is None:
        print("Error: Could not load dataset config from any model.")
        sys.exit(1)
        
    dataset, id2room_map = load_dataset_and_sample(
        args.manifest, 
        first_dataset_cfg,
        args.num_points
    )

    # Process each model
    for i in range(n_models):
        name = names[i]
        model = models[i]
        print(f"\n--- Processing Model: {name} ---")
        
        # Extract model information from name
        model_info = extract_model_info(name)
        print(f"  Model type: {model_info['model_type']}")
        print(f"  Input size: {model_info['input_size']}")
        print(f"  Latent shape: {model_info['latent_shape']}")
        print(f"  Loss type: {model_info['loss_type']}")
        if model_info.get('compression_ratio'):
            print(f"  Compression ratio: {model_info['compression_ratio']:.1f}:1")
        
        # Encode dataset
        latents, latents_spatial, labels, categories = encode_dataset(
            model, 
            dataset,
            dset_cfgs[i],
            args.label_col, 
            args.category_col, 
            filter_empty_layouts, 
            DEVICE
        )
        
        if latents is None:
            print(f"Skipping model {name} due to encoding error.")
            continue
        
        # Apply UMAP
        umap_embeddings, umap_model = apply_umap_transform(
            latents, 
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            random_state=42,
            suppress_warnings=args.suppress_warnings
        )
        
        # Translate room IDs
        room_names = translate_room_ids(labels, id2room_map)
        
        # Create DataFrame
        df = pd.DataFrame({
            'x': umap_embeddings[:, 0],
            'y': umap_embeddings[:, 1],
            args.label_col: labels,
            'room_name': room_names,
            args.category_col: categories
        })
        
        # Save results
        output_path = os.path.join(args.output_dir, f"{name}_embeddings.pkl")
        save_data = {
            'dataframe': df,
            'latents': latents,
            'latents_spatial': latents_spatial,  # Add spatial version
            'umap_embeddings': umap_embeddings,
            'umap_model': umap_model,
            'model_name': name,
            'model_info': model_info,  # Add model information
            'metadata': {
                'num_points': len(df),
                'filter_empty': filter_empty_layouts,
                'label_col': args.label_col,
                'category_col': args.category_col,
                'umap_params': {
                    'n_neighbors': args.umap_neighbors,
                    'min_dist': args.umap_min_dist
                }
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Saved embeddings to: {output_path}")
        print(f"  - Latent shape: {latents.shape}")
        print(f"  - Spatial latent shape: {latents_spatial.shape}")
        print(f"  - UMAP shape: {umap_embeddings.shape}")
        print(f"  - Unique room types: {df['room_name'].nunique()}")

    print(f"\nAll embeddings saved to: {args.output_dir}")


if __name__ == "__main__":
    main()