# visualize_latents.py (v3 - Fix Room Names & Colors)
import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from tqdm import tqdm

# (Keep load_model_from_experiment function exactly the same as previous version)
def load_model_from_experiment(config_path, checkpoint_path, device="cuda"):
    """Loads a VAE model from its experiment config and checkpoint file."""
    print(f"\nLoading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'model' not in config:
        raise KeyError(f"Error: 'model' key not found in config file: {config_path}. "
                       "Make sure you are using the 'experiment_config.yaml', not 'autoencoder_config.yaml'.")

    model_cfg = config['model']
    dataset_cfg = config['dataset']

    # Add project root to path *before* importing AutoEncoder if necessary
    # Assuming AutoEncoder import works because project root is added in main()
    from modules.autoencoder import AutoEncoder # Assuming this is correct relative to project_root

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
    state_dict = torch.load(checkpoint_path, map_location=device)

    if "decoder.final.weight" in state_dict:
        print("INFO: Detected old model checkpoint. Renaming 'decoder.final' -> 'decoder.final_rgb'.")
        state_dict["decoder.final_rgb.weight"] = state_dict.pop("decoder.final.weight")
        state_dict["decoder.final_rgb.bias"] = state_dict.pop("decoder.final.bias")

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"WARNING: Strict loading failed. Trying non-strict. Error: {e}")
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    try:
        model_name = os.path.basename(os.path.dirname(os.path.dirname(config_path)))
        if model_name == "output" or model_name == "checkpoints": # Go up one more level if needed
             model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(config_path))))
    except Exception:
        model_name = os.path.basename(config_path).replace(".yaml", "")
    print(f"Model '{model_name}' loaded successfully.")

    return model, dataset_cfg, model_name


def get_embedding(model, dataset_cfg, manifest_path, label_col, category_col, filter_empty, device="cuda"):
    """Encodes all layouts from a manifest and computes their UMAP embedding."""

    # --- 1. Load Dataset ---
    transform = T.ToTensor()
    from modules.datasets import LayoutDataset, collate_skip_none

    dataset = LayoutDataset(
        manifest_path=manifest_path,
        transform=transform,
        mode="all",
        one_hot=dataset_cfg.get("one_hot", False),
        taxonomy_path=dataset_cfg.get("taxonomy_path"),
    )

    # --- Load Taxonomy for ID translation ---
    taxonomy_path = dataset_cfg.get("taxonomy_path")
    id2room_map = {} # Default empty map
    if taxonomy_path and os.path.exists(taxonomy_path):
        try:
            with open(taxonomy_path, 'r') as f:
                taxonomy_data = json.load(f)
                # Ensure id2room exists and convert keys to string for reliable lookup
                id2room_map = {str(k): v for k, v in taxonomy_data.get("id2room", {}).items()}
                print(f"Loaded id2room map with {len(id2room_map)} entries.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from taxonomy file: {taxonomy_path}")
        except Exception as e:
            print(f"Error loading taxonomy: {e}")
    else:
        print(f"Warning: Taxonomy path '{taxonomy_path}' not found or invalid. Cannot translate room IDs.")


    loader = DataLoader(
        dataset,
        batch_size=dataset_cfg.get("batch_size", 64),
        shuffle=False,
        num_workers=dataset_cfg.get("num_workers", 4),
        collate_fn=collate_skip_none
    )

    all_latents = []
    all_labels = [] # Will store translated names
    all_categories = []

    print(f"Loaded {len(dataset)} total layouts. filter_empty={filter_empty}")
    print(f"Encoding non-empty layouts...")

    # --- 2. Encode All Layouts ---
    processed_count = 0
    for batch in tqdm(loader, desc="Encoding batches"):
        if batch is None: continue
        if 'layout' not in batch: continue

        x = batch['layout'].to(device)
        is_empty = batch.get('is_empty', [False] * x.size(0))

        # --- Filter ---
        if filter_empty:
            non_empty_mask = torch.tensor([not e for e in is_empty], dtype=torch.bool) # No need for .to(device) here
            if not non_empty_mask.any(): continue

            x = x[non_empty_mask]
            # Get original IDs/types corresponding to non-empty items
            original_labels = [label for label, empty in zip(batch.get(label_col, []), is_empty) if not empty]
            original_categories = [cat for cat, empty in zip(batch.get(category_col, []), is_empty) if not empty]

             # Handle potential missing metadata more robustly
            if len(original_labels) != x.size(0) and x.size(0) > 0:
                 print(f"Warning: Label mismatch after filtering. Layouts: {x.size(0)}, Labels: {len(original_labels)}. Using placeholders.")
                 original_labels = ['Unknown'] * x.size(0)
            if len(original_categories) != x.size(0) and x.size(0) > 0:
                 print(f"Warning: Category mismatch after filtering. Layouts: {x.size(0)}, Categories: {len(original_categories)}. Using placeholders.")
                 original_categories = ['Unknown'] * x.size(0)
        else:
            original_labels = batch.get(label_col, ['Unknown'] * x.size(0))
            original_categories = batch.get(category_col, ['Unknown'] * x.size(0))
        # --- End Filter ---

        # --- Translate Room IDs to Names (using string keys) ---
        # Translate only if id2room_map is not empty
        translated_labels = [id2room_map.get(str(l), str(l)) for l in original_labels] if id2room_map else [str(l) for l in original_labels]

        with torch.no_grad():
            z = model.encode_latent(x, deterministic=True)

        z_flat = z.view(z.size(0), -1)

        all_latents.append(z_flat.cpu().numpy())
        all_labels.extend(translated_labels)
        all_categories.extend(original_categories)
        processed_count += x.size(0)

    if not all_latents:
        print("Error: No non-empty layouts found or encoded. Cannot proceed.")
        return None

    X = np.concatenate(all_latents, axis=0)
    print(f"Finished encoding. Total non-empty layouts processed: {processed_count}. Latent shape: {X.shape}")

    # --- 3. Run UMAP ---
    print("Running UMAP (this may take a minute or more)...")
    reducer = umap.UMAP(
        n_neighbors=50,
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=42,
        # verbose=True # Uncomment for UMAP progress
    )
    embedding = reducer.fit_transform(X)
    print("UMAP complete.")

    # --- 4. Create DataFrame ---
    df = pd.DataFrame(embedding, columns=['x', 'y'])
    # Use the actual column names passed as arguments
    df[label_col] = all_labels      # Now contains names like "Kitchen"
    df[category_col] = all_categories # Contains "room" and "scene"

    return df

def plot_comparison_n(list_of_dfs, list_of_names, plot_by, filename, category_col, label_col, filter_category=None, palette=None):
    """Generates a side-by-side comparison plot for N models and saves it."""

    n_models = len(list_of_dfs)
    if n_models == 0:
        print("Error: No dataframes provided for plotting.")
        return

    print(f"Generating plot for {n_models} models: {filename}")
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 8), squeeze=False)
    axes = axes.flatten()

    # Determine a suitable palette if not provided
    if palette is None:
        # Get unique values for hue across all dataframes to determine palette size
        all_hue_values = pd.concat([df[plot_by] for df in list_of_dfs]).unique()
        n_colors = len(all_hue_values)
        if plot_by == category_col:
             # Use Set2 for categories (room/scene)
             palette = sns.color_palette("Set2", n_colors=max(n_colors, 2))
        else:
             # Use husl for room types
             palette = sns.color_palette("husl", n_colors=n_colors)

    for i, (df, model_name) in enumerate(zip(list_of_dfs, list_of_names)):
        ax = axes[i]
        plot_df = df.copy()

        if filter_category:
            if category_col not in plot_df.columns:
                 print(f"Warning: Category column '{category_col}' not found in df for model {model_name}. Cannot filter.")
            else:
                 plot_df = plot_df[plot_df[category_col] == filter_category]

        if plot_by not in plot_df.columns:
             print(f"Warning: Plotting column '{plot_by}' not found in df for model {model_name}. Skipping plot.")
             ax.set_title(f"{model_name}\n(Error: Column '{plot_by}' missing)", fontsize=14)
             continue

        if plot_df.empty:
             print(f"Warning: No data to plot for model {model_name} after filtering. Skipping plot.")
             ax.set_title(f"{model_name}\n(No data after filtering)", fontsize=14)
             continue

        sns.scatterplot(
            data=plot_df, x='x', y='y', hue=plot_by,
            s=3, alpha=0.5,
            legend= 'auto' if i == 0 else False,
            palette=palette,
            ax=ax
        )
        ax.set_title(f"{model_name}\n(Plotted by {plot_by})", fontsize=14)

        # Add legend only to the first plot and adjust title
        if i == 0:
            legend_title = 'Type' if plot_by == category_col else 'Room Type'
            try:
                # Get current handles and labels
                handles, labels = ax.get_legend_handles_labels()
                # Sort labels (and handles accordingly) for better readability, especially room types
                if labels and handles:
                    sorted_indices = np.argsort(labels)
                    handles = [handles[j] for j in sorted_indices]
                    labels = [labels[j] for j in sorted_indices]

                # Place legend outside, adjust title
                ax.legend(handles, labels, title=legend_title, markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small') # Smaller font
            except Exception as e:
                 print(f"Warning: Could not create legend for {model_name}. Error: {e}")
                 if ax.get_legend() is not None: ax.get_legend().remove() # Remove if partially created

    plot_title = f"Latent Space Comparison ({plot_by})"
    if filter_category:
        plot_title += f" (Filtered for '{filter_category}')"
    plt.suptitle(plot_title, fontsize=20, y=1.03)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved plot to {filename}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Visualize and compare N VAE latent spaces.")

    # --- Paths ---
    parser.add_argument('--project_root', type=str, required=True, help="Path to the ImgiNav project root directory")
    parser.add_argument('--manifest', type=str, required=True, help="Path to the full manifest.csv file")
    parser.add_argument('--output_dir', type=str, default=".", help="Directory to save the output plots")

    # --- Metadata ---
    parser.add_argument('--label_col', type=str, default="room_id", help="Column name for room type (default: 'room_id')") # Default added
    parser.add_argument('--category_col', type=str, default="type", help="Column name for category (default: 'type')") # Default added
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
    # Make sure this path exists and is correct relative to where you run the script
    if not os.path.isdir(args.project_root):
        print(f"Error: Project root directory not found: {args.project_root}")
        sys.exit(1)
    sys.path.insert(0, args.project_root) # Use insert instead of append for priority
    try:
        global AutoEncoder
        # Now try importing AFTER adding to path
        from modules.autoencoder import AutoEncoder
    except ImportError as e:
        print(f"Error: Could not import AutoEncoder from modules.autoencoder. Error: {e}")
        print(f"Searched sys.path includes: {sys.path}")
        print("Please ensure --project_root points to the correct directory containing the 'modules' folder (e.g., /path/to/ImgiNav/ImgiNav)")
        sys.exit(1)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    os.makedirs(args.output_dir, exist_ok=True)

    filter_empty_layouts = not args.no_filter_empty

    # --- Load Models and Generate Embeddings ---
    all_dfs = {}
    all_names = {}
    first_dataset_cfg = None
    models = []
    names = []
    dset_cfgs = []

    for i in range(n_models):
        config_path = args.configs[i]
        ckpt_path = args.checkpoints[i]
        try:
            model, dset_cfg, name = load_model_from_experiment(config_path, ckpt_path, DEVICE)
            models.append(model)
            names.append(name)
            dset_cfgs.append(dset_cfg)
            if first_dataset_cfg is None:
                first_dataset_cfg = dset_cfg
        except FileNotFoundError:
             print(f"Error: Config or Checkpoint not found for model {i+1}. Paths:\nConfig: {config_path}\nCheckpoint: {ckpt_path}")
             print("Skipping this model.")
        except KeyError as e:
             print(f"Error loading model {i+1} due to missing key: {e}. Config path: {config_path}")
             print("Skipping this model.")
        except Exception as e:
             print(f"Unexpected error loading model {i+1} ({config_path}): {e}")
             print("Skipping this model.")


    # Generate embeddings sequentially
    for i in range(len(models)): # Use length of successfully loaded models
        name = names[i]
        model = models[i]
        print(f"\n--- Generating Embedding for Model ({name}) ---")
        try:
            # Pass the specific dataset_cfg for this model
            df = get_embedding(model, dset_cfgs[i], args.manifest, args.label_col, args.category_col, filter_empty_layouts, DEVICE)
            if df is not None:
                 all_dfs[name] = df
                 all_names[name] = name # Use the loaded name as key
            else:
                 print(f"Skipping model {name} due to embedding error.")
        except FileNotFoundError:
             print(f"Error: Manifest file not found during embedding for model {name}: {args.manifest}")
             print("Skipping embedding for this model.")
        except KeyError as e:
             print(f"Error during embedding for model {name}: Missing key '{e}' in batch data or manifest.")
             print("Skipping embedding for this model.")
        except Exception as e:
             print(f"Unexpected error during embedding for model {name}: {e}")
             print("Skipping embedding for this model.")

    if not all_dfs:
        print("Error: Failed to generate embeddings for any models. Exiting.")
        sys.exit(1)

    print("\n--- Embeddings generated. Starting plots. ---")
    active_names = list(all_dfs.keys()) # Get names of models that were successfully processed


    # --- Identify Models for Comparison ---
    # Figure 1: MSE (model 0) vs. SegLoss 64x64x4 (model 1)
    if len(active_names) >= 2:
        model_indices_fig1 = [0, 1]
        # Get names based on original order, check if they were processed
        names_fig1 = [names[i] for i in model_indices_fig1 if i < len(names) and names[i] in all_dfs]
        dfs_fig1 = [all_dfs[name] for name in names_fig1]


        if len(dfs_fig1) == 2:
            print("\n--- Plotting Figure 1: MSE vs SegLoss 64x64 ---")
            # --- Plot 1a: Continents ---
            plot_comparison_n(
                dfs_fig1, names_fig1,
                plot_by=args.category_col,
                filename=os.path.join(args.output_dir, "latent_comparison_continents_mse_vs_seg64.png"),
                category_col=args.category_col,
                label_col=args.label_col, # Pass label_col
                palette=sns.color_palette("Set2", n_colors=len(pd.concat(dfs_fig1)[args.category_col].unique())) # Use Set2
            )
            # --- Plot 1b: Countries ---
            plot_comparison_n(
                dfs_fig1, names_fig1,
                plot_by=args.label_col,
                filename=os.path.join(args.output_dir, "latent_comparison_countries_mse_vs_seg64.png"),
                category_col=args.category_col,
                label_col=args.label_col, # Pass label_col
                filter_category='room' # Assumes 'room' is the category string
            )
        else:
            print("Warning: Could not create MSE vs Seg64 comparison plots (missing one or both models' embeddings).")

    # Figure 2: All SegLoss Models (models 1, 2, 3, 4 based on input order)
    # Assumes the order is [MSE64, Seg64, Seg32, SegHigh32, SegNarrow32]
    model_indices_fig2 = list(range(1, len(names))) # Get indices for all models *except* the first (MSE) one
    names_fig2 = [names[i] for i in model_indices_fig2 if i < len(names) and names[i] in all_dfs]
    dfs_fig2 = [all_dfs[name] for name in names_fig2]

    if len(dfs_fig2) >= 2: # Need at least 2 SegLoss models to compare
        print("\n--- Plotting Figure 2: SegLoss Models Comparison ---")
        # --- Plot 2a: Continents ---
        plot_comparison_n(
            dfs_fig2, names_fig2,
            plot_by=args.category_col,
            filename=os.path.join(args.output_dir, "latent_comparison_continents_segloss_sizes.png"),
            category_col=args.category_col,
            label_col=args.label_col, # Pass label_col
            palette=sns.color_palette("Set2", n_colors=len(pd.concat(dfs_fig2)[args.category_col].unique())) # Use Set2
        )
        # --- Plot 2b: Countries ---
        plot_comparison_n(
            dfs_fig2, names_fig2,
            plot_by=args.label_col,
            filename=os.path.join(args.output_dir, "latent_comparison_countries_segloss_sizes.png"),
            category_col=args.category_col,
            label_col=args.label_col, # Pass label_col
            filter_category='room' # Assumes 'room' is the category string
        )
    elif len(dfs_fig2) == 1:
        print("Info: Only one SegLoss model embedding was generated. Skipping SegLoss comparison plot.")
    else:
        print("Warning: No SegLoss model embeddings were generated. Skipping SegLoss comparison plot.")


    print("\nAll done. Plots saved to " + args.output_dir)

if __name__ == "__main__":
    main()