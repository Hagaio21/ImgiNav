#!/usr/bin/env python
# analyze_embeddings.py
"""
Analyzes pre-computed embeddings from autoencoder models.
Creates visualizations with distinct colors and computes clustering metrics.
"""
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from matplotlib.patches import Patch
import matplotlib.cm as cm


def analyze_spatial_structure(latents_spatial, labels, room_names):
    """Analyze the spatial structure of latent representations.
    
    Args:
        latents_spatial: numpy array of shape (N, C, H, W)
        labels: list of room IDs
        room_names: list of room names
    
    Returns:
        dict containing various spatial analyses
    """
    N, C, H, W = latents_spatial.shape
    
    # Get unique room types
    unique_rooms = sorted(list(set(room_names)))
    
    # 1. Channel-wise activation by room type
    channel_activations = {}
    for room in unique_rooms:
        mask = np.array([rn == room for rn in room_names])
        if mask.sum() > 0:
            # Average activation per channel for this room type
            room_latents = latents_spatial[mask]
            channel_activations[room] = room_latents.mean(axis=(0, 2, 3))  # Average over samples and spatial dims
    
    # 2. Spatial attention maps by room type
    spatial_attention = {}
    for room in unique_rooms:
        mask = np.array([rn == room for rn in room_names])
        if mask.sum() > 0:
            room_latents = latents_spatial[mask]
            # Average over samples and channels to get spatial importance
            spatial_attention[room] = room_latents.mean(axis=(0, 1))  # Shape: (H, W)
    
    # 3. Channel importance (variance across samples)
    channel_variance = latents_spatial.var(axis=(0, 2, 3))  # Variance per channel
    channel_importance = channel_variance / channel_variance.sum()
    
    # 4. Spatial variance (which regions vary most)
    spatial_variance = latents_spatial.var(axis=0).mean(axis=0)  # Average variance across channels
    
    return {
        'channel_activations': channel_activations,
        'spatial_attention': spatial_attention,
        'channel_importance': channel_importance,
        'spatial_variance': spatial_variance,
        'shape': (C, H, W)
    }


def create_spatial_visualization(spatial_analysis, model_name, output_dir):
    """Create visualizations of spatial structure analysis."""
    C, H, W = spatial_analysis['shape']
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Channel importance bar plot
    ax1 = plt.subplot(2, 3, 1)
    channels = np.arange(C)
    ax1.bar(channels, spatial_analysis['channel_importance'])
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Importance (normalized variance)')
    ax1.set_title(f'{model_name} - Channel Importance')
    ax1.set_xticks(channels)
    
    # 2. Spatial variance heatmap
    ax2 = plt.subplot(2, 3, 2)
    im = ax2.imshow(spatial_analysis['spatial_variance'], cmap='hot', aspect='equal')
    ax2.set_title(f'{model_name} - Spatial Variance\n(Higher = more variable across samples)')
    plt.colorbar(im, ax=ax2)
    
    # 3. Channel activations by room type (top 5 rooms)
    ax3 = plt.subplot(2, 3, 3)
    room_names = list(spatial_analysis['channel_activations'].keys())[:5]  # Top 5 rooms
    channel_acts = np.array([spatial_analysis['channel_activations'][room] for room in room_names])
    
    im = ax3.imshow(channel_acts, aspect='auto', cmap='viridis')
    ax3.set_yticks(range(len(room_names)))
    ax3.set_yticklabels(room_names)
    ax3.set_xticks(range(C))
    ax3.set_xlabel('Channel')
    ax3.set_title('Channel Activations by Room Type')
    plt.colorbar(im, ax=ax3)
    
    # 4-6. Spatial attention for top 3 room types
    for i, (idx, room) in enumerate([(4, 0), (5, 1), (6, 2)]):
        if i < len(room_names):
            ax = plt.subplot(2, 3, idx)
            room_name = room_names[room]
            spatial_att = spatial_analysis['spatial_attention'][room_name]
            im = ax.imshow(spatial_att, cmap='viridis', aspect='equal')
            ax.set_title(f'Spatial Attention: {room_name}')
            plt.colorbar(im, ax=ax)
        else:
            ax = plt.subplot(2, 3, idx)
            ax.axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{model_name}_spatial_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_31_distinct_colors():
    """Generate 31 visually distinct colors using multiple strategies."""
    colors = []
    
    # 1. Start with some hand-picked distinct colors (12 colors)
    base_colors = [
        '#FF0000',  # Red
        '#00FF00',  # Green
        '#0000FF',  # Blue
        '#FFD700',  # Gold
        '#FF00FF',  # Magenta
        '#00FFFF',  # Cyan
        '#FF8C00',  # Dark Orange
        '#800080',  # Purple
        '#008000',  # Dark Green
        '#FFC0CB',  # Pink
        '#A52A2A',  # Brown
        '#808080',  # Gray
    ]
    colors.extend(base_colors)
    
    # 2. Use HSV color space for maximum separation (19 more colors)
    # Start from a different offset to avoid overlap with base colors
    for i in range(19):
        hue = (i * 360 / 19 + 15) % 360  # Offset by 15 degrees
        # Vary saturation and value for more distinction
        if i % 3 == 0:
            sat, val = 1.0, 0.8
        elif i % 3 == 1:
            sat, val = 0.7, 1.0
        else:
            sat, val = 0.85, 0.65
        
        # Convert HSV to RGB
        import colorsys
        rgb = colorsys.hsv_to_rgb(hue/360, sat, val)
        hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in rgb)
        colors.append(hex_color)
    
    return colors[:31]  # Return exactly 31 colors


def load_embeddings(filepath):
    """Load embeddings from pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def compute_clustering_metrics(embeddings, labels, label_name):
    """Compute clustering quality metrics."""
    # Convert labels to numeric if they're strings
    if isinstance(labels[0], str):
        unique_labels = list(set(labels))
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = [label_to_int[label] for label in labels]
    else:
        numeric_labels = labels
    
    # Skip if only one cluster
    n_clusters = len(set(numeric_labels))
    if n_clusters < 2:
        return {
            'silhouette': np.nan,
            'calinski_harabasz': np.nan,
            'davies_bouldin': np.nan,
            'n_clusters': n_clusters
        }
    
    # Compute metrics
    try:
        silhouette = silhouette_score(embeddings, numeric_labels)
    except:
        silhouette = np.nan
    
    try:
        ch_score = calinski_harabasz_score(embeddings, numeric_labels)
    except:
        ch_score = np.nan
    
    try:
        db_score = davies_bouldin_score(embeddings, numeric_labels)
    except:
        db_score = np.nan
    
    return {
        'silhouette': silhouette,
        'calinski_harabasz': ch_score,
        'davies_bouldin': db_score,
        'n_clusters': n_clusters
    }


def plot_single_embedding(df, plot_by, title, ax, palette=None, show_legend=True, 
                         category_col=None, filter_category=None):
    """Plot a single embedding with specified coloring."""
    # Filter if needed
    if filter_category and category_col:
        df = df[df[category_col] == filter_category].copy()
    
    # Get unique values for coloring
    unique_values = df[plot_by].unique()
    n_colors = len(unique_values)
    
    # Generate colors if palette not provided
    if palette is None:
        if n_colors <= 31:
            colors_list = generate_31_distinct_colors()
            palette = {val: colors_list[i % 31] for i, val in enumerate(sorted(unique_values))}
        else:
            # Fallback to colormap if more than 31 categories
            cmap = cm.get_cmap('tab20', n_colors)
            palette = {val: cmap(i) for i, val in enumerate(sorted(unique_values))}
    
    # Plot each category
    for value in sorted(unique_values):
        mask = df[plot_by] == value
        ax.scatter(df[mask]['x'], df[mask]['y'], 
                  c=palette[value], 
                  label=value if show_legend else "",
                  s=10, alpha=0.5)
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(title, fontsize=12, pad=10)
    ax.grid(True, alpha=0.3)
    
    # Add legend if requested
    if show_legend and n_colors <= 15:  # Only show legend if not too many categories
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                 markerscale=1.5, fontsize=8)
    
    return palette


def plot_comparison(embeddings_list, names, plot_by, filename, 
                   category_col=None, filter_category=None, 
                   palette=None, figsize=None, grid_layout=None):
    """Create comparison plots for multiple embeddings."""
    n_models = len(embeddings_list)
    
    # Determine layout
    if grid_layout:
        n_rows, n_cols = grid_layout
    else:
        n_cols = min(n_models, 3)
        n_rows = (n_models + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    # Get consistent palette across all plots
    all_values = set()
    for emb_data in embeddings_list:
        df = emb_data['dataframe']
        if filter_category and category_col:
            df = df[df[category_col] == filter_category]
        all_values.update(df[plot_by].unique())
    
    if palette is None and len(all_values) <= 31:
        colors_list = generate_31_distinct_colors()
        palette = {val: colors_list[i % 31] for i, val in enumerate(sorted(all_values))}
    
    # Plot each model
    for i, (emb_data, name) in enumerate(zip(embeddings_list, names)):
        df = emb_data['dataframe']
        plot_single_embedding(df, plot_by, name, axes[i], 
                            palette=palette,
                            show_legend=(i == 0),  # Only show legend on first plot
                            category_col=category_col,
                            filter_category=filter_category)
    
    # Hide extra axes
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    # Add a single shared legend if we have room
    if len(all_values) <= 31 and palette:
        # Create legend elements
        legend_elements = [Patch(facecolor=palette[val], label=val) 
                          for val in sorted(all_values)]
        
        # Add legend to the side
        fig.legend(handles=legend_elements, 
                  bbox_to_anchor=(1.02, 0.5), 
                  loc='center left',
                  ncol=1 if len(all_values) <= 15 else 2,
                  fontsize=8)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")


def create_summary_tables(metrics_df, all_embeddings, output_dir):
    """Create formatted summary tables for thesis/papers with model dimensions."""
    
    # Add model dimensions to the dataframe
    for idx, row in metrics_df.iterrows():
        model_name = row['model']
        if model_name in all_embeddings and 'model_info' in all_embeddings[model_name]:
            info = all_embeddings[model_name]['model_info']
            metrics_df.at[idx, 'latent_shape'] = f"{info['latent_shape'][1]}×{info['latent_shape'][2]}×{info['latent_shape'][0]}" if info['latent_shape'] else "Unknown"
            metrics_df.at[idx, 'latent_dims'] = info['latent_dims'] if 'latent_dims' in info else None
            metrics_df.at[idx, 'compression'] = f"{info['compression_ratio']:.1f}:1" if info.get('compression_ratio') else "Unknown"
    
    # Round values for cleaner presentation
    summary_df = metrics_df.copy()
    summary_df['silhouette'] = summary_df['silhouette'].round(4)
    summary_df['calinski_harabasz'] = summary_df['calinski_harabasz'].round(2)
    summary_df['davies_bouldin'] = summary_df['davies_bouldin'].round(4)
    
    # Create LaTeX table
    latex_columns = ['model', 'latent_shape', 'compression', 'silhouette', 'calinski_harabasz', 'davies_bouldin']
    latex_table = summary_df[latex_columns].to_latex(
        index=False,
        caption="Clustering metrics for autoencoder latent spaces",
        label="tab:clustering_metrics",
        column_format='l|ccccc',
        float_format=lambda x: f'{x:.4f}' if isinstance(x, float) and abs(x) < 10 else f'{x:.2f}' if isinstance(x, float) else str(x)
    )
    
    # Save LaTeX table
    with open(os.path.join(output_dir, 'metrics_table.tex'), 'w') as f:
        f.write(latex_table)
    
    # Create Markdown table
    markdown_table = "| Model | Latent Shape | Compression | Silhouette Score | Calinski-Harabasz | Davies-Bouldin |\n"
    markdown_table += "|-------|--------------|-------------|-----------------|-------------------|----------------|\n"
    for _, row in summary_df.iterrows():
        markdown_table += f"| {row['model']} | {row.get('latent_shape', 'N/A')} | {row.get('compression', 'N/A')} | {row['silhouette']:.4f} | {row['calinski_harabasz']:.2f} | {row['davies_bouldin']:.4f} |\n"
    
    # Save Markdown table
    with open(os.path.join(output_dir, 'metrics_table.md'), 'w') as f:
        f.write(markdown_table)
    
    # Create summary statistics
    summary_stats = f"""
# Clustering Metrics Summary

## Model Dimensions

"""
    for _, row in summary_df.iterrows():
        summary_stats += f"**{row['model']}**: {row.get('latent_shape', 'N/A')} latent, {row.get('compression', 'N/A')} compression\n"
    
    summary_stats += f"""

## Best Performing Models

**Silhouette Score (higher is better):**
- Best: {summary_df.loc[summary_df['silhouette'].idxmax(), 'model']} ({summary_df['silhouette'].max():.4f})
- Worst: {summary_df.loc[summary_df['silhouette'].idxmin(), 'model']} ({summary_df['silhouette'].min():.4f})
- Mean: {summary_df['silhouette'].mean():.4f} ± {summary_df['silhouette'].std():.4f}

**Calinski-Harabasz Score (higher is better):**
- Best: {summary_df.loc[summary_df['calinski_harabasz'].idxmax(), 'model']} ({summary_df['calinski_harabasz'].max():.2f})
- Worst: {summary_df.loc[summary_df['calinski_harabasz'].idxmin(), 'model']} ({summary_df['calinski_harabasz'].min():.2f})
- Mean: {summary_df['calinski_harabasz'].mean():.2f} ± {summary_df['calinski_harabasz'].std():.2f}

**Davies-Bouldin Score (lower is better):**
- Best: {summary_df.loc[summary_df['davies_bouldin'].idxmin(), 'model']} ({summary_df['davies_bouldin'].min():.4f})
- Worst: {summary_df.loc[summary_df['davies_bouldin'].idxmax(), 'model']} ({summary_df['davies_bouldin'].max():.4f})
- Mean: {summary_df['davies_bouldin'].mean():.4f} ± {summary_df['davies_bouldin'].std():.4f}

## Interpretation for Diffusion Models

Based on the metrics, the best model for diffusion generation would be:
"""
    
    # Score each model for diffusion suitability
    summary_df['diffusion_score'] = 0
    
    # Silhouette: optimal range 0.3-0.6
    summary_df.loc[(summary_df['silhouette'] >= 0.3) & (summary_df['silhouette'] <= 0.6), 'diffusion_score'] += 1
    
    # Calinski-Harabasz: optimal range 100-500
    summary_df.loc[(summary_df['calinski_harabasz'] >= 100) & (summary_df['calinski_harabasz'] <= 500), 'diffusion_score'] += 1
    
    # Davies-Bouldin: optimal range 0.7-1.3
    summary_df.loc[(summary_df['davies_bouldin'] >= 0.7) & (summary_df['davies_bouldin'] <= 1.3), 'diffusion_score'] += 1
    
    best_for_diffusion = summary_df.loc[summary_df['diffusion_score'].idxmax()]
    
    summary_stats += f"""
**{best_for_diffusion['model']}** (Diffusion suitability score: {best_for_diffusion['diffusion_score']}/3)
- Silhouette: {best_for_diffusion['silhouette']:.4f} {'✓' if 0.3 <= best_for_diffusion['silhouette'] <= 0.6 else '✗'}
- Calinski-Harabasz: {best_for_diffusion['calinski_harabasz']:.2f} {'✓' if 100 <= best_for_diffusion['calinski_harabasz'] <= 500 else '✗'}
- Davies-Bouldin: {best_for_diffusion['davies_bouldin']:.4f} {'✓' if 0.7 <= best_for_diffusion['davies_bouldin'] <= 1.3 else '✗'}
"""
    
    # Save summary
    with open(os.path.join(output_dir, 'metrics_summary.txt'), 'w') as f:
        f.write(summary_stats)
    
    print(f"\nSaved summary tables:")
    print(f"  - LaTeX table: {os.path.join(output_dir, 'metrics_table.tex')}")
    print(f"  - Markdown table: {os.path.join(output_dir, 'metrics_table.md')}")
    print(f"  - Summary statistics: {os.path.join(output_dir, 'metrics_summary.txt')}")
    
    return summary_df


def create_spatial_comparison(spatial_analysis_results, output_dir):
    """Create a comparison of spatial structures across all models."""
    if len(spatial_analysis_results) < 2:
        return
    
    models = sorted(spatial_analysis_results.keys())
    n_models = len(models)
    
    # Create figure for channel importance comparison
    fig, axes = plt.subplots(1, n_models, figsize=(4*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    max_channels = max(spatial_analysis_results[m]['shape'][0] for m in models)
    
    for i, model in enumerate(models):
        ax = axes[i]
        C, H, W = spatial_analysis_results[model]['shape']
        importance = spatial_analysis_results[model]['channel_importance']
        
        # Pad if needed for alignment
        if len(importance) < max_channels:
            importance = np.pad(importance, (0, max_channels - len(importance)))
        
        ax.bar(range(len(importance)), importance[:C], color='steelblue')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Importance' if i == 0 else '')
        ax.set_title(f'{model}\n({H}×{W}×{C})')
        ax.set_ylim(0, max(importance) * 1.1)
    
    plt.suptitle('Channel Importance Comparison Across Models', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'channel_importance_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create spatial variance comparison
    fig, axes = plt.subplots(1, n_models, figsize=(4*n_models, 4))
    if n_models == 1:
        axes = [axes]
    
    vmin = min(spatial_analysis_results[m]['spatial_variance'].min() for m in models)
    vmax = max(spatial_analysis_results[m]['spatial_variance'].max() for m in models)
    
    for i, model in enumerate(models):
        ax = axes[i]
        spatial_var = spatial_analysis_results[model]['spatial_variance']
        im = ax.imshow(spatial_var, cmap='hot', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(f'{model}')
        ax.axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    plt.suptitle('Spatial Variance Comparison', fontsize=14)
    plt.savefig(os.path.join(output_dir, 'spatial_variance_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_metrics_comparison_plot(metrics_df, output_path):
    """Create a comparison plot of clustering metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Silhouette Score (higher is better)
    ax = axes[0]
    bars = ax.bar(range(len(metrics_df)), metrics_df['silhouette'])
    ax.set_xlabel('Model')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score\n(Higher is Better)')
    ax.set_xticks(range(len(metrics_df)))
    ax.set_xticklabels(metrics_df['model'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Color best performer
    best_idx = metrics_df['silhouette'].idxmax()
    bars[best_idx].set_color('green')
    
    # Calinski-Harabasz Score (higher is better)
    ax = axes[1]
    bars = ax.bar(range(len(metrics_df)), metrics_df['calinski_harabasz'])
    ax.set_xlabel('Model')
    ax.set_ylabel('Calinski-Harabasz Score')
    ax.set_title('Calinski-Harabasz Score\n(Higher is Better)')
    ax.set_xticks(range(len(metrics_df)))
    ax.set_xticklabels(metrics_df['model'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Color best performer
    best_idx = metrics_df['calinski_harabasz'].idxmax()
    bars[best_idx].set_color('green')
    
    # Davies-Bouldin Score (lower is better)
    ax = axes[2]
    bars = ax.bar(range(len(metrics_df)), metrics_df['davies_bouldin'])
    ax.set_xlabel('Model')
    ax.set_ylabel('Davies-Bouldin Score')
    ax.set_title('Davies-Bouldin Score\n(Lower is Better)')
    ax.set_xticks(range(len(metrics_df)))
    ax.set_xticklabels(metrics_df['model'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Color best performer (lowest value)
    best_idx = metrics_df['davies_bouldin'].idxmin()
    bars[best_idx].set_color('green')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze pre-computed embeddings")
    parser.add_argument('--embeddings_dir', type=str, required=True, 
                       help="Directory containing embedding pickle files")
    parser.add_argument('--output_dir', type=str, default="analysis_results", 
                       help="Directory to save analysis results")
    
    # Optional: specify which models to analyze
    parser.add_argument('--models', type=str, nargs='*', 
                       help="Specific model names to analyze (default: all)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    sns.set_style("darkgrid")
    # Load all embedding files
    embedding_files = [f for f in os.listdir(args.embeddings_dir) 
                      if f.endswith('_embeddings.pkl')]
    
    if not embedding_files:
        print(f"No embedding files found in {args.embeddings_dir}")
        sys.exit(1)
    
    # Load embeddings
    all_embeddings = {}
    for filename in sorted(embedding_files):
        filepath = os.path.join(args.embeddings_dir, filename)
        data = load_embeddings(filepath)
        model_name = data['model_name']
        
        # Filter by model names if specified
        if args.models and model_name not in args.models:
            continue
            
        all_embeddings[model_name] = data
        print(f"Loaded embeddings for: {model_name}")
    
    if not all_embeddings:
        print("No embeddings matched the criteria")
        sys.exit(1)
    
    # Get metadata from first embedding
    first_data = next(iter(all_embeddings.values()))
    label_col = first_data['metadata']['label_col']
    category_col = first_data['metadata']['category_col']
    
    # Prepare for metrics calculation and spatial analysis
    metrics_results = []
    spatial_analysis_results = {}
    
    # Compute metrics for each model
    for model_name, emb_data in all_embeddings.items():
        print(f"\n--- Computing metrics for {model_name} ---")
        
        df = emb_data['dataframe']
        umap_embeddings = emb_data['umap_embeddings']
        
        # Check if spatial latents are available
        if 'latents_spatial' in emb_data and emb_data['latents_spatial'] is not None:
            print(f"Performing spatial analysis...")
            spatial_analysis = analyze_spatial_structure(
                emb_data['latents_spatial'],
                df[label_col].values,
                df['room_name'].values
            )
            spatial_analysis_results[model_name] = spatial_analysis
            
            # Create spatial visualization
            create_spatial_visualization(spatial_analysis, model_name, args.output_dir)
        
        # Compute metrics for room_name clustering
        room_metrics = compute_clustering_metrics(
            umap_embeddings, 
            df['room_name'].values,
            'room_name'
        )
        
        # Compute metrics for category clustering
        category_metrics = compute_clustering_metrics(
            umap_embeddings,
            df[category_col].values,
            category_col
        )
        
        metrics_results.append({
            'model': model_name,
            'silhouette': room_metrics['silhouette'],
            'calinski_harabasz': room_metrics['calinski_harabasz'],
            'davies_bouldin': room_metrics['davies_bouldin'],
            'n_clusters': room_metrics['n_clusters'],
            'category_silhouette': category_metrics['silhouette'],
            'category_ch': category_metrics['calinski_harabasz'],
            'category_db': category_metrics['davies_bouldin']
        })
        
        print(f"  Room clustering (n={room_metrics['n_clusters']} clusters):")
        print(f"    Silhouette: {room_metrics['silhouette']:.4f}")
        print(f"    Calinski-Harabasz: {room_metrics['calinski_harabasz']:.2f}")
        print(f"    Davies-Bouldin: {room_metrics['davies_bouldin']:.4f}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_results)
    metrics_df.to_csv(os.path.join(args.output_dir, 'clustering_metrics.csv'), index=False)
    print(f"\nSaved metrics to: {os.path.join(args.output_dir, 'clustering_metrics.csv')}")
    
    # Create summary tables for thesis (now with model dimensions)
    summary_df = create_summary_tables(metrics_df, all_embeddings, args.output_dir)
    
    # Create metrics comparison plot
    create_metrics_comparison_plot(
        metrics_df, 
        os.path.join(args.output_dir, 'metrics_comparison.png')
    )
    
    # Generate visualization plots
    embeddings_list = list(all_embeddings.values())
    names_list = list(all_embeddings.keys())
    
    # Plot 1: All models colored by category
    print("\nGenerating visualizations...")
    category_palette = {"room": "orange", "scene": "blue", "Unknown": "grey"}
    plot_comparison(
        embeddings_list, names_list,
        plot_by=category_col,
        filename=os.path.join(args.output_dir, "embeddings_by_category.svg"),
        palette=category_palette,
        grid_layout=(2, 3) if len(embeddings_list) > 4 else None
    )
    
    # Plot 2: All models colored by room type (filtered to rooms only)
    plot_comparison(
        embeddings_list, names_list,
        plot_by='room_name',
        filename=os.path.join(args.output_dir, "embeddings_by_room_type.svg"),
        category_col=category_col,
        filter_category='room',
        grid_layout=(2, 3) if len(embeddings_list) > 4 else None
    )
    
    # Plot 3: Individual high-quality plots for each model
    os.makedirs(os.path.join(args.output_dir, 'individual_plots'), exist_ok=True)
    for model_name, emb_data in all_embeddings.items():
        # Room type plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        df = emb_data['dataframe']
        room_df = df[df[category_col] == 'room'].copy()
        
        plot_single_embedding(
            room_df, 'room_name', 
            f'{model_name} - Room Types',
            ax, show_legend=True
        )
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output_dir, 'individual_plots', f'{model_name}_rooms.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
    
    # Create spatial structure comparison if available
    if spatial_analysis_results:
        print("\nCreating spatial structure comparisons...")
        create_spatial_comparison(spatial_analysis_results, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")
    
    # Print metrics table to console
    print("\n" + "="*70)
    print("CLUSTERING METRICS SUMMARY")
    print("="*70)
    
    # Read and print the markdown table
    with open(os.path.join(args.output_dir, 'metrics_table.md'), 'r') as f:
        print(f.read())
    
    # Print summary of best models
    print("\n=== Best Models by Metric ===")
    print(f"Best Silhouette Score: {metrics_df.loc[metrics_df['silhouette'].idxmax(), 'model']} "
          f"({metrics_df['silhouette'].max():.4f})")
    print(f"Best Calinski-Harabasz: {metrics_df.loc[metrics_df['calinski_harabasz'].idxmax(), 'model']} "
          f"({metrics_df['calinski_harabasz'].max():.2f})")
    print(f"Best Davies-Bouldin: {metrics_df.loc[metrics_df['davies_bouldin'].idxmin(), 'model']} "
          f"({metrics_df['davies_bouldin'].min():.4f})")


if __name__ == "__main__":
    main()