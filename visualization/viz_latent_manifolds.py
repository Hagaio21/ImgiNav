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
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from tqdm import tqdm

# (Keep load_model_from_experiment, get_embedding, and plot_comparison_n functions exactly as they were in the previous version)

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
        # Assumes path like .../experiments/VAE/MODEL_NAME/output/config.yaml
        model_name = os.path.basename(os.path.dirname(os.path.dirname(config_path)))
        if model_name == "cross_enth_loss": # Handle potential extra directory level
             model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(config_path))))
    except Exception:
        model_name = os.path.basename(config_path).replace(".yaml", "") # Fallback name
    print(f"Model '{model_name}' loaded successfully.")
    
    return model, dataset_cfg, model_name

def get_embedding(model, dataset_cfg, manifest_path, label_col, category_col, filter_empty, device="cuda"):
    """Encodes all layouts from a manifest and computes their UMAP embedding."""
    
    # --- 1. Load Dataset ---
    transform = T.ToTensor()
    from modules.datasets import LayoutDataset, collate_skip_none #
    
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
    
    loader = DataLoader(
        dataset, 
        batch_size=dataset_cfg.get("batch_size", 64), # Increased batch size
        shuffle=False, 
        num_workers=dataset_cfg.get("num_workers", 4),
        collate_fn=collate_skip_none
    )
    
    all_latents = []
    all_labels = []
    all_categories = []
    
    print(f"Loaded {len(dataset)} total layouts. filter_empty={filter_empty}")
    print(f"Encoding non-empty layouts...")
    
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
            z = model.encode_latent(x, deterministic=True) #
        
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

def plot_comparison_n(list_of_dfs, list_of_names, plot_by, filename, category_col, filter_category=None, palette=None):
    """Generates a side-by-side comparison plot for N models and saves it."""
    
    n_models = len(list_of_dfs)
    if n_models == 0:
        print("Error: No dataframes provided for plotting.")
        return
        
    print(f"Generating plot for {n_models} models: {filename}")
    
    # Dynamically create N subplots in one row
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 8), squeeze=False) 
    axes = axes.flatten() # Ensure axes is always a 1D array

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
            s=3, alpha=0.5, # Adjusted point size and alpha for density
            legend= 'auto' if i == 0 else False, # Show legend only on the first plot
            palette=palette or 'tab20', 
            ax=ax
        )
        ax.set_title(f"{model_name}\n(Plotted by {plot_by})", fontsize=14) # Slightly smaller font
        
        # Add legend only to the first plot
        if i == 0:
            legend_title = 'Type' if plot_by == category_col else 'Room Type'
            try:
                # Place legend outside for potentially many room types/categories
                ax.legend(title=legend_title, markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
            except ValueError:
                 print(f"Warning: Could not create legend for {model_name}. Possibly no data points to plot.")


    # Add a main title above all subplots
    plot_title = f"Latent Space Comparison ({plot_by})"
    if filter_category:
        plot_title += f" (Filtered for '{filter_category}')"
    plt.suptitle(plot_title, fontsize=20, y=1.03)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout for suptitle
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
        from modules.autoencoder import AutoEncoder #
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

    # Generate embeddings sequentially
    for i in range(n_models):
        name = names[i]
        model = models[i]
        print(f"\n--- Generating Embedding for Model ({name}) ---")
        # Use the dataset_cfg corresponding to THIS model for loading params like batch_size
        df = get_embedding(model, dset_cfgs[i], args.manifest, args.label_col, args.category_col, filter_empty_layouts, DEVICE)
        
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
                filename=os.path.join(args.output_dir, "latent_comparison_continents_mse_vs_seg64.png"),
                category_col=args.category_col,
                palette=category_palette 
            )
            # --- Plot 1b: Countries ---
            plot_comparison_n(
                dfs_fig1, names_fig1,
                plot_by=args.label_col,
                filename=os.path.join(args.output_dir, "latent_comparison_countries_mse_vs_seg64.png"),
                category_col=args.category_col,
                filter_category='room' 
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
                filename=os.path.join(args.output_dir, "latent_comparison_continents_segloss_sizes.png"),
                category_col=args.category_col,
                palette=category_palette 
            )
            # --- Plot 2b: Countries ---
            plot_comparison_n(
                dfs_fig2, names_fig2,
                plot_by=args.label_col,
                filename=os.path.join(args.output_dir, "latent_comparison_countries_segloss_sizes.png"),
                category_col=args.category_col,
                filter_category='room' 
            )
        else:
             print("Warning: Could not create SegLoss comparison plots (fewer than 2 SegLoss models found/processed).")
    elif n_models >= 2:
         print("Info: Fewer than 5 models provided. Skipping SegLoss size comparison plot.")


    print("\nAll done. Plots saved to " + args.output_dir)

if __name__ == "__main__":
    main()