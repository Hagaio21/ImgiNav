#!/usr/bin/env python
# visualize_latents.py
import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
import argparse
import json
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend, essential for HPC
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap # Assuming CPU UMAP
# If you installed cuml: import cuml.manifold as umap
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as T
from tqdm import tqdm

# (Make sure 'modules.autoencoder' and 'modules.datasets' are importable
#  by adding the project root to sys.path, as done in main())


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
    # Ensure AutoEncoder is globally available after sys.path append
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
        # Assumes path like .../experiments/VAE/MODEL_NAME/output/config.yaml
        model_name = os.path.basename(os.path.dirname(os.path.dirname(config_path)))
        if model_name == "cross_enth_loss": # Handle potential extra directory level
             model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(config_path))))
    except Exception:
        model_name = os.path.basename(config_path).replace(".yaml", "") # Fallback name
    print(f"Model '{model_name}' loaded successfully.")
    
    return model, dataset_cfg, model_name

# --- NEW FUNCTION: Loads and samples the dataset ONCE ---
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

# --- MODIFIED: Takes a pre-loaded dataset ---
def get_embedding(model, dataset, id2room_map, dataset_cfg, label_col, category_col, filter_empty, device="cuda"):
    """Encodes all layouts from a *pre-loaded* dataset and computes their UMAP embedding."""
    
    # Ensure collate_fn is available
    from modules.datasets import collate_skip_none
    
    loader = DataLoader(
        dataset, 
        batch_size=dataset_cfg.get("batch_size", 64), # Use model-specific batch size
        shuffle=False, 
        num_workers=dataset_cfg.get("num_workers", 4),
        collate_fn=collate_skip_none
    )
    
    all_latents = []
    all_labels = []
    all_categories = []
    
    print(f"Encoding {len(dataset)} layouts (filter_empty={filter_empty})...")
    
    # --- 2. Encode All Layouts ---
    for batch in tqdm(loader, desc="Encoding batches"):
        # Check if batch is None (due to collate_fn skipping)
        if batch is None:
            continue
            
        # Ensure 'layout' key exists
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
            
            # Ensure columns exist before accessing
            current_labels = batch.get(label_col, [])
            current_categories = batch.get(category_col, [])

            labels = [label for label, empty in zip(current_labels, is_empty) if not empty]
            categories = [cat for cat, empty in zip(current_categories, is_empty) if not empty]

            # Handle case where columns might be missing in a batch after filtering/collating
            if len(labels) != x.size(0) and x.size(0) > 0:
                 print(f"Warning: Mismatch after filtering. Layouts: {x.size(0)}, Labels: {len(labels)}. Using placeholders.")
                 labels = ['Unknown'] * x.size(0) # Assign placeholder
            if len(categories) != x.size(0) and x.size(0) > 0:
                 print(f"Warning: Mismatch after filtering. Layouts: {x.size(0)}, Categories: {len(categories)}. Using placeholders.")
                 categories = ['Unknown'] * x.size(0) # Assign placeholder

        else:
            # Provide defaults if keys are missing even when not filtering
            labels = batch.get(label_col, ['Unknown'] * x.size(0))
            categories = batch.get(category_col, ['Unknown'] * x.size(0))
            
        # --- Translate Room IDs to Names ---
        translated_labels = [str(id2room_map.get(str(l), l)) for l in labels] # Ensure string keys
        
        with torch.no_grad():
            z = model.encode_latent(x, deterministic=True)
        
        z_flat = z.view(z.size(0), -1)
        
        all_latents.append(z_flat.cpu().numpy())
        all_labels.extend(translated_labels) 
        all_categories.extend(categories)
        
    if not all_latents:
        print("Error: No non-empty layouts found or encoded. Cannot proceed.")
        return None 

    X = np.concatenate(all_latents, axis=0)
    print(f"Finished encoding. Total non-empty layouts: {X.shape[0]}")
    
    # --- 3. Run UMAP ---
    print("Running UMAP (this may take a minute or more)...")
    reducer = umap.UMAP(
        n_neighbors=50,
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=42
    )
    embedding = reducer.fit_transform(X)
    print("UMAP complete.")
    
    # --- 4. Create DataFrame ---
    df = pd.DataFrame(embedding, columns=['x', 'y'])
    df[label_col] = all_labels      
    df[category_col] = all_categories 
    
    return df

def plot_comparison_n(list_of_dfs, list_of_names, plot_by, filename, category_col, 
                      filter_category=None, palette=None, grid_layout=None):
    """
    Generates a comparison plot for N models and saves it.
    Can specify a grid_layout (e.g., (2, 2)) for the subplots.
    """
    
    n_models = len(list_of_dfs)
    if n_models == 0:
        print("Error: No dataframes provided for plotting.")
        return
        
    print(f"Generating plot for {n_models} models: {filename}")

    # --- DYNAMIC PALETTE GENERATION ---
    final_palette = palette
    n_hues = 0

    if final_palette is None:
        # If no palette is provided, create one.
        # We must find all unique values for 'plot_by' across *all* dataframes
        # in this function call to ensure the palette is consistent.
        
        try:
            # Combine all DFs that will be plotted
            all_dfs_combined = pd.concat(list_of_dfs)
            
            # Apply the *same filter* that will be used in the loop
            if filter_category:
                if category_col in all_dfs_combined.columns:
                     all_dfs_combined = all_dfs_combined[all_dfs_combined[category_col] == filter_category]
                # (Warning will be printed in loop if col is missing)
            
            if plot_by in all_dfs_combined.columns:
                unique_hues = all_dfs_combined[plot_by].unique()
                unique_hues.sort() # Ensure consistent order!
                n_hues = len(unique_hues)
                
                # Check if we need more colors than the default 'tab20'
                if n_hues > 20:
                    print(f"  > Info: Found {n_hues} unique categories for '{plot_by}'. Generating 'hls' palette.")
                    # Generate a list of n_hues distinct colors
                    final_palette_list = sns.color_palette("hls", n_hues)
                    # Create a dictionary to map each unique value to a color
                    # This ensures consistency across all subplots
                    final_palette = {hue: color for hue, color in zip(unique_hues, final_palette_list)}
                else:
                    # --- THIS IS THE FIX ---
                    # Always create a dictionary, even for <= 20 hues, to ensure
                    # the *same* dictionary is passed to *both* subplots.
                    print(f"  > Info: Found {n_hues} unique categories for '{plot_by}'. Generating 'tab20' palette dict.")
                    final_palette_list = sns.color_palette("tab20", n_hues)
                    final_palette = {hue: color for hue, color in zip(unique_hues, final_palette_list)}
                    # --- END OF FIX ---
            else:
                # Column is missing, plotting will fail later, but set a default
                final_palette = 'tab20'
        except Exception as e:
            print(f"Warning: Could not auto-generate palette. Defaulting to 'tab20'. Error: {e}")
            final_palette = 'tab20'
            
    # --- END OF DYNAMIC PALETTE ---


    # --- NEW: DYNAMIC GRID LAYOUT ---
    if grid_layout:
        rows, cols = grid_layout
        # Check if layout is valid for the number of models
        if rows * cols < n_models:
            print(f"Warning: Grid layout {grid_layout} is too small for {n_models} models. Defaulting to 1x{n_models}.")
            rows, cols = 1, n_models
        figsize = (8 * cols, 8 * rows) # Adjust figsize based on grid
    else:
        # Default to 1xN layout
        rows, cols = 1, n_models
        figsize = (8 * n_models, 8) # Original logic
    
    # Dynamically create subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False) 
    axes = axes.flatten() # Ensure axes is always a 1D array
    # --- END OF DYNAMIC GRID ---

    for i, (df, model_name) in enumerate(zip(list_of_dfs, list_of_names)):
        ax = axes[i]
        plot_df = df.copy()
        
        if filter_category:
             # Ensure the category column exists before filtering
            if category_col not in plot_df.columns:
                 print(f"Warning: Category column '{category_col}' not found in dataframe for model {model_name}. Cannot filter.")
            else:
                 plot_df = plot_df[plot_df[category_col] == filter_category]

        # Ensure the plot_by column exists
        if plot_by not in plot_df.columns:
             print(f"Warning: Plotting column '{plot_by}' not found in dataframe for model {model_name}. Skipping plot.")
             ax.set_title(f"{model_name}\n(Error: Column '{plot_by}' missing)", fontsize=14)
             continue
        
        sns.scatterplot(
            data=plot_df, x='x', y='y', hue=plot_by,
            s=4, alpha=0.7, # Adjusted point size and alpha for density
            legend= 'auto' if i == 0 else False, # Show legend only on the first plot
            palette=final_palette, # Use the new dynamic palette
            ax=ax
        )
        ax.set_title(f"{model_name}\n(Plotted by {plot_by})", fontsize=14) # Slightly smaller font
        
        # Add legend only to the first plot
        if i == 0:
            legend_title = 'Type' if plot_by == category_col else 'Room Type'
            try:
                # --- MODIFIED: Always show legend ---
                ax.legend(title=legend_title, markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
                
            except Exception as e:
                 print(f"Warning: Could not create/modify legend for {model_name}. Error: {e}")

    # --- Clean up extra subplots if grid is larger than n_models ---
    if grid_layout:
        for j in range(n_models, rows * cols):
            fig.delaxes(axes[j]) # Remove unused axes

    # Add a main title above all subplots
    plot_title = f"Latent Space Comparison ({plot_by})"
    if filter_category:
        plot_title += f" (Filtered for '{filter_category}')"
    plt.suptitle(plot_title, fontsize=20, y=1.03)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout for suptitle
    
    # Save the figure. Format is inferred from the filename extension (.svg)
    plt.savefig(filename, bbox_inches='tight') 
    print(f"Saved plot to {filename}")
    plt.close(fig) # Close the figure to free up memory

def main():
    parser = argparse.ArgumentParser(description="Visualize and compare N VAE latent spaces.")
    
    # --- Paths ---
    parser.add_argument('--project_root', type=str, required=True, help="Path to the ImgiNav project root directory")
    parser.add_argument('--manifest', type=str, required=True, help="Path to the full manifest.csv file")
    parser.add_argument('--output_dir', type=str, default=".", help="Directory to save the output plots")
    
    # --- Metadata ---
    parser.add_argument('--label_col', type=str, required=True, help="Column name for room type (e.g., 'room_id')")
    parser.add_argument('--category_col', type=str, required=True, help="Column name for category (e.g., 'type')")
    parser.add_argument('--no_filter_empty', action='store_true', help="Set this flag to *disable* filtering of empty layouts")
    
    # --- NEW: Sampling ---
    parser.add_argument('--num_points', type=int, default=None, help="Number of random points to sample for embedding (default: all)")
    
    # --- Models (Accepts multiple) ---
    parser.add_argument('--configs', type=str, nargs='+', required=True, help="List of paths to experiment_config.yaml files")
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True, help="List of paths to .pt checkpoint files (must match order of --configs)")

    args = parser.parse_args()

    # --- Validate Input ---
    if len(args.configs) != len(args.checkpoints):
        print("Error: Number of config files must match number of checkpoint files.")
        sys.exit(1)
    n_models = len(args.configs)
    if n_models == 0:
        print("Error: No models specified.")
        sys.exit(1)
    print(f"Received {n_models} models to compare.")

    # --- Add project root to path ---
    sys.path.append(args.project_root)
    try:
        global AutoEncoder
        from modules.autoencoder import AutoEncoder
        global LayoutDataset, collate_skip_none
        from modules.datasets import LayoutDataset, collate_skip_none
    except ImportError:
        print(f"Error: Could not import modules from {args.project_root}")
        print("Please ensure --project_root points to the correct directory (e.g., /work3/s233249/ImgiNav/ImgiNav)")
        sys.exit(1)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    filter_empty_layouts = not args.no_filter_empty

    # --- Load Models and Generate Embeddings ---
    all_dfs = {}
    all_names = {}
    
    # Assume the first dataset config is representative
    first_dataset_cfg = None 
    
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
        if first_dataset_cfg is None:
            first_dataset_cfg = dset_cfg # Store the first one

    # --- NEW: Load and sample dataset ONCE ---
    if first_dataset_cfg is None:
        print("Error: Could not load dataset config from any model.")
        sys.exit(1)
        
    dataset, id2room_map = load_dataset_and_sample(
        args.manifest, 
        first_dataset_cfg, # Use first config for dataset params
        args.num_points
    )
    # --- END NEW ---

    # Generate embeddings sequentially (NOW using the SAME dataset)
    for i in range(n_models):
        name = names[i]
        model = models[i]
        print(f"\n--- Generating Embedding for Model ({name}) ---")
        
        # --- MODIFIED: Pass the pre-loaded dataset ---
        df = get_embedding(
            model, 
            dataset,         # Pass the single, sampled dataset
            id2room_map,     # Pass the single map
            dset_cfgs[i],    # Pass the model-specific config (for batch_size)
            args.label_col, 
            args.category_col, 
            filter_empty_layouts, 
            DEVICE
        )
        
        if df is not None:
             all_dfs[name] = df
             all_names[name] = name # Store name with df
        else:
             print(f"Skipping model {name} due to embedding error.")
             
    if not all_dfs:
        print("Error: Failed to generate embeddings for any models. Exiting.")
        sys.exit(1)

    print("\n--- Embeddings generated. Starting plots. ---")

    # --- Identify Models for Comparison ---
    # We rely on the order provided in the command line
    # Figure 1: MSE (model 0) vs. SegLoss 64x64x4 (model 1)
    if n_models >= 2:
        model_indices_fig1 = [0, 1]
        dfs_fig1 = [all_dfs[names[i]] for i in model_indices_fig1 if names[i] in all_dfs]
        names_fig1 = [all_names[names[i]] for i in model_indices_fig1 if names[i] in all_dfs]
        
        if len(dfs_fig1) == 2:
            # --- Plot 1a: Continents ---
            category_palette = {"room": "orange", "scene": "blue", "Unknown": "grey"} 
            plot_comparison_n(
                dfs_fig1, names_fig1,
                plot_by=args.category_col,
                # *** CHANGED to .svg ***
                filename=os.path.join(args.output_dir, "latent_comparison_continents_mse_vs_seg64.svg"),
                category_col=args.category_col,
                palette=category_palette
                # No grid_layout specified, will default to 1x2
            )
            # --- Plot 1b: Countries ---
            plot_comparison_n(
                dfs_fig1, names_fig1,
                plot_by=args.label_col,
                # *** CHANGED to .svg ***
                filename=os.path.join(args.output_dir, "latent_comparison_countries_mse_vs_seg64.svg"),
                category_col=args.category_col,
                filter_category='room' 
                # No grid_layout specified, will default to 1x2
                # Palette will be auto-generated consistently by plot_comparison_n
            )
        else:
            print("Warning: Could not create MSE vs Seg64 comparison plots (missing one or both models).")

    # Figure 2: All SegLoss Models (models 1, 2, 3, 4)
    if n_models >= 5: # Assumes the order is [MSE64, Seg64, Seg32, SegHigh32, SegNarrow32]
        model_indices_fig2 = [1, 2, 3, 4]
        dfs_fig2 = [all_dfs[names[i]] for i in model_indices_fig2 if names[i] in all_dfs]
        names_fig2 = [all_names[names[i]] for i in model_indices_fig2 if names[i] in all_dfs]
        
        if len(dfs_fig2) > 1: # Need at least 2 models to compare
            # --- Plot 2a: Continents ---
            category_palette = {"room": "orange", "scene": "blue", "Unknown": "grey"} 
            plot_comparison_n(
                dfs_fig2, names_fig2,
                plot_by=args.category_col,
                # *** CHANGED to .svg ***
                filename=os.path.join(args.output_dir, "latent_comparison_continents_segloss_sizes.svg"),
                category_col=args.category_col,
                palette=category_palette,
                grid_layout=(2, 2) # *** ADDED grid_layout ***
            )
            # --- Plot 2b: Countries ---
            plot_comparison_n(
                dfs_fig2, names_fig2,
                plot_by=args.label_col,
                # *** CHANGED to .svg ***
                filename=os.path.join(args.output_dir, "latent_comparison_countries_segloss_sizes.svg"),
                category_col=args.category_col,
                filter_category='room',
                grid_layout=(2, 2) #
                # Palette will be auto-generated consistently by plot_comparison_n
            )
        else:
             print("Warning: Could not create SegLoss comparison plots (fewer than 2 SegLoss models found/processed).")
    elif n_models >= 2:
         print("Info: Fewer than 5 models provided. Skipping SegLoss size comparison plot.")


    print("\nAll done. Plots saved to " + args.output_dir)

if __name__ == "__main__":
    main()