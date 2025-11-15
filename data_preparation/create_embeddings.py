#!/usr/bin/env python3
"""
Unified embedding creation for layouts, POVs, and graphs.

This script can create embeddings for:
- Layout images (RGB → latent using autoencoder) - for diffusion training
- POV images (using ResNet18)
- Graph data (using SentenceTransformer)

Layout embeddings support two workflows:
1. Manifest-based: Read from manifest CSV, encode RGB images, add latent_path column
2. Directory-based: Scan directory for layouts, create embeddings

Usage:
    # Layout embeddings (manifest-based, for diffusion)
    python create_embeddings.py --type layout \
        --manifest datasets/manifest.csv \
        --output-manifest datasets/manifest_with_latents.csv \
        --autoencoder-config config.yaml \
        --autoencoder-checkpoint checkpoint.pt
    
    # Layout embeddings (directory-based, legacy)
    python create_embeddings.py --type layout \
        --data-root datasets/ \
        --autoencoder-config config.yaml \
        --autoencoder-checkpoint checkpoint.pt
    
    # POV embeddings
    python create_embeddings.py --type pov \
        --manifest datasets/manifest.csv \
        --output datasets/manifest_with_pov_emb.csv
    
    # Graph embeddings
    python create_embeddings.py --type graph \
        --manifest datasets/manifest.csv \
        --output datasets/manifest_with_graph_emb.csv \
        --taxonomy config/taxonomy.json
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterator
from itertools import islice

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import yaml
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

# Add parent directory to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.autoencoder import Autoencoder
from models.datasets.datasets import ManifestDataset
from training.utils import load_config, build_dataset
from utils.text_utils import graph2text
from common.taxonomy import Taxonomy
from common.file_io import read_manifest, create_manifest


# =============================================================================
# Utility Functions
# =============================================================================

def batched(iterable: Iterator, n: int):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def _process_batch_with_model(batch_items, model, transform, device, process_item_fn, encode_fn):
    batch_imgs = []
    valid_items = []
    
    for item in batch_items:
        try:
            img, metadata = process_item_fn(item)
            if img is not None:
                batch_imgs.append(transform(img))
                valid_items.append(metadata)
        except Exception as e:
            print(f"Error processing item: {e}")
            continue
    
    if not valid_items:
        return []
    
    # Process batch
    imgs_tensor = torch.stack(batch_imgs).to(device)
    with torch.no_grad():
        embeddings = encode_fn(imgs_tensor)
    
    # Return results
    results = []
    for metadata, emb in zip(valid_items, embeddings):
        results.append((metadata, emb.cpu()))
    
    return results


# =============================================================================
# Model Loading
# =============================================================================

def load_autoencoder_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """Load autoencoder using the newer Autoencoder API."""
    print(f"[INFO] Loading autoencoder from {checkpoint_path}")
    
    # Load config
    config = load_config(config_path)
    ae_cfg = config.get("autoencoder", config)
    
    # Load model using newer API
    model = Autoencoder.load_checkpoint(checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()
    
    print(f"[INFO] Autoencoder loaded successfully")
    return model


def load_resnet_model(device: str = "cuda"):
    from torchvision import models
    resnet = models.resnet18(weights="IMAGENET1K_V1").to(device)
    resnet.fc = torch.nn.Identity()  # remove classifier head
    resnet.eval()
    return resnet


def load_sentence_transformer_model(model_name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)


# =============================================================================
# Layout Embeddings (Manifest-based workflow for diffusion)
# =============================================================================

def create_layout_embeddings_from_manifest(
    encoder, manifest_path, output_manifest_path, batch_size=32, 
    num_workers=8, overwrite=False, device="cuda", autoencoder_config_path=None,
    output_latent_dir=None, diffusion_config_path=None
):
    """
    Create layout embeddings from manifest (manifest-based workflow).
    This is the preferred workflow for diffusion training.
    """
    manifest_path = Path(manifest_path)
    output_manifest_path = Path(output_manifest_path)
    manifest_dir = manifest_path.parent
    
    # Load manifest
    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest with {len(df)} samples")
    
    # Apply the same filters that ManifestDataset will use
    # Filter out rows with NaN values in required columns
    required_cols = ["layout_path"]
    df = df.dropna(subset=required_cols)
    
    # Apply filters (same as ManifestDataset)
    # Default: filter empty layouts
    filters = {"is_empty": [False]}
    
    # Try to load filters from config
    # Priority: 1) diffusion config filters (for whiteness filtering), 2) autoencoder config filters, 3) default
    if diffusion_config_path:
        try:
            from training.utils import load_config
            config = load_config(diffusion_config_path)
            dataset_filters = config.get("dataset", {}).get("filters", None)
            if dataset_filters:
                filters = dataset_filters
                print(f"[INFO] Using filters from diffusion config: {filters}")
        except Exception as e:
            print(f"[INFO] Could not load filters from diffusion config: {e}")
    elif autoencoder_config_path:
        try:
            from training.utils import load_config
            config = load_config(autoencoder_config_path)
            dataset_filters = config.get("dataset", {}).get("filters", None)
            if dataset_filters:
                filters = dataset_filters
                print(f"[INFO] Using filters from autoencoder config: {filters}")
        except Exception as e:
            print(f"[INFO] Could not load filters from autoencoder config: {e}, using default filters")
    
    # Apply filters using ManifestDataset's filter logic
    for key, value in filters.items():
        if "__lt" in key:
            col = key.replace("__lt", "")
            if col in df.columns:
                df = df[df[col] < value]
        elif "__gt" in key:
            col = key.replace("__gt", "")
            if col in df.columns:
                df = df[df[col] > value]
        elif "__le" in key:
            col = key.replace("__le", "")
            if col in df.columns:
                df = df[df[col] <= value]
        elif "__ge" in key:
            col = key.replace("__ge", "")
            if col in df.columns:
                df = df[df[col] >= value]
        elif "__ne" in key:
            col = key.replace("__ne", "")
            if col in df.columns:
                df = df[df[col] != value]
        else:
            if key in df.columns:
                if isinstance(value, (list, tuple, set)):
                    df = df[df[key].isin(value)]
                else:
                    df = df[df[key] == value]
    df = df.reset_index(drop=True)
    
    print(f"After filtering: {len(df)} samples")
    
    # Create dataset for loading images (using filtered manifest)
    # Load transform from autoencoder config if available (for consistent preprocessing)
    # This ensures embedding uses same transform as training (e.g., 256×256 resize)
    transform = None
    if autoencoder_config_path:
        try:
            from training.utils import load_config
            config = load_config(autoencoder_config_path)
            transform = config.get("dataset", {}).get("transform", None)
            if transform:
                print(f"[INFO] Using transform from autoencoder config: {transform.get('type', 'Compose')}")
            else:
                print("[INFO] No transform found in config, using default image loading")
        except Exception as e:
            print(f"Warning: Could not load transform from config: {e}")
            print("Using default image loading (no transform)")
    else:
        print("[INFO] No autoencoder config path provided, using default image loading")
    
    # Use the same filters for ManifestDataset
    dataset_filters = filters if filters else None
    dataset = ManifestDataset(
        manifest=str(manifest_path),
        outputs={"rgb": "layout_path"},
        filters=dataset_filters,
        return_path=False,
        transform=transform  # Pass transform if available
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, pin_memory=True)
    
    # Verify that filtered dataframe matches dataset length
    if len(df) != len(dataset):
        raise ValueError(
            f"Mismatch between filtered dataframe length ({len(df)}) "
            f"and dataset length ({len(dataset)}). "
            "They should match after applying the same filters."
        )
    
    device_obj = torch.device(device)
    encoder = encoder.to(device_obj)
    
    # Process all samples
    latent_paths = []
    processed = 0
    skipped = 0
    failed = 0
    
    # Collect latents for statistics computation
    all_latents = []
    
    output_manifest_dir = output_manifest_path.parent
    output_manifest_dir.mkdir(parents=True, exist_ok=True)
    
    print("Encoding images to latents...")
    print(f"[INFO] Using transform: {dataset.transform}")
    print(f"[INFO] Sample RGB range check (first batch):")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Encoding")):
            rgb_images = batch["rgb"].to(device_obj, non_blocking=True)
            
            # Log RGB range for first batch to verify normalization
            if batch_idx == 0:
                rgb_min = rgb_images.min().item()
                rgb_max = rgb_images.max().item()
                rgb_mean = rgb_images.mean().item()
                print(f"[INFO] First batch RGB stats: min={rgb_min:.3f}, max={rgb_max:.3f}, mean={rgb_mean:.3f}")
                print(f"[INFO] Expected range: [-1, 1] (if using Normalize with mean=0.5, std=0.5)")
                if rgb_min < -1.1 or rgb_max > 1.1:
                    print(f"[WARNING] RGB values outside expected [-1, 1] range! Transform may be incorrect.")
                elif rgb_min > -0.1 and rgb_max < 1.1:
                    print(f"[WARNING] RGB values in [0, 1] range - transform may be missing Normalize step!")
            
            # Encode to latents
            encoder_out = encoder(rgb_images)  # Returns dict
            
            # Extract latent from dict
            if "latent" in encoder_out:
                latents = encoder_out["latent"]
            elif "mu" in encoder_out:
                latents = encoder_out["mu"]  # Use mu for VAE
            else:
                raise ValueError(f"Encoder output must contain 'latent' or 'mu'. Got: {list(encoder_out.keys())}")
            
            # Collect latents for statistics (detach and move to CPU to save memory)
            all_latents.append(latents.detach().cpu())
            
            # Save latents for each sample in batch
            batch_start_idx = batch_idx * batch_size
            for i in range(len(rgb_images)):
                sample_idx = batch_start_idx + i
                if sample_idx >= len(df):
                    break
                
                row = df.iloc[sample_idx]
                
                # Determine output path for latent
                layout_path_str = row.get("layout_path", "")
                if not layout_path_str:
                    latent_paths.append("")
                    failed += 1
                    continue
                
                layout_path = Path(layout_path_str)
                
                # Convert to absolute path if relative
                if not layout_path.is_absolute():
                    # If relative, resolve relative to manifest directory
                    layout_path = (manifest_dir / layout_path).resolve()
                
                # Use output_latent_dir if provided (experiment folder), otherwise use dataset structure
                if output_latent_dir is not None:
                    # Save in experiment folder (preferred for diffusion training)
                    latent_dir = Path(output_latent_dir)
                else:
                    # Legacy: For augmented dataset: create latents in /work3/s233249/ImgiNav/datasets/augmented/latents/
                    # Structure: images/name.png -> latents/name.pt
                    # Find the augmented directory in the path
                    latent_base_dir = None
                    if "augmented" in layout_path.parts:
                        # Find the augmented directory and use it as base
                        parts = list(layout_path.parts)
                        for j, part in enumerate(parts):
                            if part == "augmented":
                                # Use everything up to and including "augmented" as base
                                latent_base_dir = Path(*parts[:j+1])
                                break
                    
                    if latent_base_dir is None:
                        # Fallback: use manifest directory's parent (assuming it's in datasets/augmented/)
                        latent_base_dir = manifest_dir.parent if manifest_dir.name != "augmented" else manifest_dir
                    
                    # Create latent path: augmented/latents/filename.pt
                    latent_dir = latent_base_dir / "latents"
                
                # Get filename from layout_path and change extension
                if layout_path.name:
                    latent_filename = Path(layout_path.name).with_suffix('.pt')
                    latent_path_full = latent_dir / latent_filename
                else:
                    # Fallback: use hash of path or index
                    latent_filename = f"latent_{sample_idx}.pt"
                    latent_path_full = latent_dir / latent_filename
                
                # Ensure path is absolute (for cross-environment compatibility)
                if not latent_path_full.is_absolute():
                    latent_path_full = latent_path_full.resolve()
                
                # Skip if exists and not overwriting
                if latent_path_full.exists() and not overwrite:
                    latent_paths.append(str(latent_path_full.resolve()))
                    skipped += 1
                    continue
                
                # Save latent tensor
                try:
                    latent_path_full.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(latents[i].cpu(), latent_path_full)
                    latent_paths.append(str(latent_path_full.resolve()))
                    processed += 1
                except Exception as e:
                    print(f"Error saving latent for {layout_path}: {e}")
                    latent_paths.append("")
                    failed += 1
    
    # Create output manifest with latent_path column
    # Ensure all paths are absolute for cross-environment compatibility
    df_output = df.copy()
    df_output["latent_path"] = ""
    
    for idx in range(len(df_output)):
        if idx < len(latent_paths) and latent_paths[idx]:
            path_obj = Path(latent_paths[idx])
            # Convert to absolute if not already
            if not path_obj.is_absolute():
                path_obj = (manifest_dir / path_obj).resolve()
            else:
                path_obj = path_obj.resolve()
            df_output.loc[idx, "latent_path"] = str(path_obj)
    
    # Ensure layout_path column is also absolute for consistency
    if "layout_path" in df_output.columns:
        df_output["layout_path"] = df_output["layout_path"].apply(
            lambda p: str((manifest_dir / Path(p)).resolve()) if p and not Path(p).is_absolute() else str(p) if p else ""
        )
    
    # Save new manifest
    df_output.to_csv(output_manifest_path, index=False)
    
    # Compute latent statistics
    print("\nComputing latent statistics...")
    if all_latents:
        # Concatenate all latents
        all_latents_tensor = torch.cat(all_latents, dim=0)
        
        # Flatten for global statistics
        latent_flat = all_latents_tensor.reshape(all_latents_tensor.shape[0], -1)
        
        # Compute global statistics
        latent_mean = latent_flat.mean().item()
        latent_std = latent_flat.std().item()
        latent_min = latent_flat.min().item()
        latent_max = latent_flat.max().item()
        
        # Compute per-channel statistics (if spatial dimensions exist)
        if all_latents_tensor.ndim == 4:  # [B, C, H, W]
            per_channel_mean = all_latents_tensor.mean(dim=(0, 2, 3)).cpu().numpy()  # [C]
            per_channel_std = all_latents_tensor.std(dim=(0, 2, 3)).cpu().numpy()  # [C]
            # min/max need to be called with a single dim, so we need to reshape first
            # Reshape to [B*H*W, C] then take min/max along first dim
            B, C, H, W = all_latents_tensor.shape
            latents_reshaped = all_latents_tensor.permute(1, 0, 2, 3).reshape(C, -1)  # [C, B*H*W]
            per_channel_min = latents_reshaped.min(dim=1)[0].cpu().numpy()  # [C]
            per_channel_max = latents_reshaped.max(dim=1)[0].cpu().numpy()  # [C]
            
            print(f"\n{'='*60}")
            print(f"Latent Distribution Statistics (After Encoding)")
            print(f"{'='*60}")
            print(f"Global Statistics:")
            print(f"  Mean: {latent_mean:.6f} (target: 0.0, ideal for N(0,1))")
            print(f"  Std:  {latent_std:.6f} (target: 1.0, ideal for N(0,1))")
            print(f"  Min:  {latent_min:.6f}")
            print(f"  Max:  {latent_max:.6f}")
            print(f"  Range: [{latent_min:.6f}, {latent_max:.6f}]")
            
            mean_deviation = abs(latent_mean)
            std_deviation = abs(latent_std - 1.0)
            print(f"\nDistribution Quality:")
            if mean_deviation < 0.1 and std_deviation < 0.2:
                print(f"  ✓ Good: Mean deviation {mean_deviation:.6f} < 0.1, Std deviation {std_deviation:.6f} < 0.2")
            elif mean_deviation < 0.2 and std_deviation < 0.5:
                print(f"  ⚠ Moderate: Mean deviation {mean_deviation:.6f}, Std deviation {std_deviation:.6f}")
            else:
                print(f"  ✗ Poor: Mean deviation {mean_deviation:.6f}, Std deviation {std_deviation:.6f}")
                print(f"    Consider adjusting LatentStandardizationLoss or KLDLoss weight")
            
            print(f"\nPer-Channel Statistics:")
            print(f"  Channel | Mean      | Std       | Min       | Max       | Status")
            print(f"  {'-'*70}")
            for ch_idx in range(len(per_channel_mean)):
                ch_mean = per_channel_mean[ch_idx]
                ch_std = per_channel_std[ch_idx]
                ch_min = per_channel_min[ch_idx]
                ch_max = per_channel_max[ch_idx]
                mean_dev = abs(ch_mean)
                std_dev = abs(ch_std - 1.0)
                status = "✓" if mean_dev < 0.1 and std_dev < 0.2 else "⚠" if mean_dev < 0.2 and std_dev < 0.5 else "✗"
                print(f"  {ch_idx:7d} | {ch_mean:9.6f} | {ch_std:9.6f} | {ch_min:9.6f} | {ch_max:9.6f} | {status}")
            
            # Check bounds for VAE vs AE
            if abs(latent_max) > 2.0 or abs(latent_min) > 2.0:
                print(f"\n⚠ Warning: Latents exceed [-2, 2] range (VAE bounds)")
                print(f"   Consider using wider bounds in diffusion sampling or adjusting VAE KL weight")
            elif abs(latent_max) > 1.0 or abs(latent_min) > 1.0:
                print(f"\nℹ Info: Latents exceed [-1, 1] range (AE bounds)")
                print(f"   This is expected for VAE. Diffusion should use [-2, 2] clamping.")
            
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"Latent Distribution Statistics (After Encoding)")
            print(f"{'='*60}")
            print(f"Global Statistics:")
            print(f"  Mean: {latent_mean:.6f} (target: 0.0)")
            print(f"  Std:  {latent_std:.6f} (target: 1.0)")
            print(f"  Min: {latent_min:.6f}, Max: {latent_max:.6f}")
            print(f"{'='*60}")
    else:
        print("⚠ No latents collected for statistics (all skipped?)")
    
    print(f"\n✓ Encoding complete!")
    print(f"  Processed: {processed}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Output manifest: {output_manifest_path}")


# =============================================================================
# Layout Embeddings (Directory-based workflow, legacy)
# =============================================================================

def parse_layout_path(path: Path, data_root: Path) -> Tuple[str, str, str]:
    relative_parts = path.relative_to(data_root).parts
    
    if len(relative_parts) == 3:
        # e.g., ('scene_000', 'room_0', 'layout.png')
        scene_id = relative_parts[0]
        type = "room"
        room_id = relative_parts[1]
    elif len(relative_parts) == 2:
        # e.g., ('scene_000', 'scene_layout.png')
        scene_id = relative_parts[0]
        type = "scene"
        room_id = "none"
    else:
        raise ValueError(f"Unexpected path structure: {path.relative_to(data_root)}")
    return scene_id, type, room_id


def create_layout_embeddings_from_directory(model, data_root: Path, device: str = "cuda", 
                            batch_size: int = 32, overwrite: bool = False) -> List[Dict[str, Any]]:
    """
    Create layout embeddings by scanning directory (legacy workflow).
    """
    layout_paths = sorted(list(data_root.rglob("*layout.png")))
    
    if not layout_paths:
        raise RuntimeError(f"No layout images found under {data_root}")
    
    # Get image size from encoder config (need to check model structure)
    # For newer Autoencoder, encoder might have image_size attribute or we need to infer
    try:
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'image_size'):
            image_size = model.encoder.image_size
        else:
            image_size = 256  # Default fallback
    except:
        image_size = 256
    
    # Prepare transform
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    
    # Scan paths and build manifest/job list
    manifest_data = []
    paths_to_encode = []
    skipped_existing = 0
    skipped_parsing = 0
    
    print(f"[INFO] Scanning {len(layout_paths)} layout paths...")
    for path in tqdm(layout_paths, desc="Scanning paths", unit="file"):
        try:
            scene_id, type, room_id = parse_layout_path(path, data_root)
        except ValueError as e:
            print(f"[WARN] Skipping file with unexpected path: {e}")
            skipped_parsing += 1
            continue
        
        out_path = path.with_name(path.stem + "_emb.pt")
        entry = {
            "scene": scene_id,
            "type": type,
            "room_id": room_id,
            "layout_path": str(path.relative_to(data_root)),
            "layout_emb_path": str(out_path.relative_to(data_root))
        }
        manifest_data.append(entry)
        
        if not out_path.exists() or overwrite:
            paths_to_encode.append((path, out_path))
        else:
            skipped_existing += 1
    
    # Encode images in batches
    total_to_encode = len(paths_to_encode)
    success = 0
    failed_load = 0
    
    print(f"[INFO] Found {total_to_encode} layouts to encode.")
    
    def process_layout_item(item):
        in_path, out_path = item
        try:
            img = Image.open(in_path).convert("RGB")
            return img, out_path
        except Exception as e:
            print(f"[WARN] Failed to load {in_path}: {e}")
            return None, None
    
    def encode_layouts(imgs_tensor):
        # Use encoder from autoencoder
        encoder_out = model.encode(imgs_tensor)
        if "latent" in encoder_out:
            return encoder_out["latent"]
        elif "mu" in encoder_out:
            return encoder_out["mu"]
        else:
            raise ValueError(f"Encoder output must contain 'latent' or 'mu'. Got: {list(encoder_out.keys())}")
    
    for i in tqdm(range(0, total_to_encode, batch_size), desc="Encoding layouts", unit="batch"):
        batch_job_paths = paths_to_encode[i:i + batch_size]
        results = _process_batch_with_model(
            batch_job_paths, model, transform, device, 
            process_layout_item, encode_layouts
        )
        
        for out_path, emb in results:
            torch.save(emb, out_path)
            success += 1
    
    print("\n[SUMMARY]")
    print(f"  Total layouts found: {len(layout_paths)}")
    print(f"  Skipped (bad path):  {skipped_parsing}")
    print(f"  Skipped (existing):  {skipped_existing}")
    print(f"  Total to encode:     {total_to_encode}")
    print(f"    Encoded new:     {success}")
    print(f"    Failed to load:  {failed_load}")
    
    return manifest_data


# =============================================================================
# POV Embeddings
# =============================================================================

def create_pov_embeddings(manifest_path: Path, output_manifest: Path,
                         save_format: str = "pt", batch_size: int = 32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading ResNet18 model...")
    model = load_resnet_model(device)
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # Read manifest
    rows = read_manifest(manifest_path)
    
    print(f"Found {len(rows)} POV images to process")
    
    output_rows = []
    skipped = 0
    processed = 0
    
    def process_pov_item(row):
        pov_path = row.get('pov_path') or row.get('pov_image', '')
        if not pov_path:
            return None, None
        
        # Check if empty flag exists
        is_empty = int(row.get('is_empty', 0))
        if is_empty or not Path(pov_path).exists():
            return None, None
            
        try:
            img = Image.open(pov_path).convert("RGB")
            return img, row
        except Exception as e:
            print(f"Error reading {pov_path}: {e}")
            return None, None
    
    def encode_povs(imgs_tensor):
        with torch.cuda.amp.autocast():
            return model(imgs_tensor)
    
    for batch_rows in tqdm(batched(rows, batch_size),
                           total=len(rows)//batch_size + 1,
                           desc="Processing POV images"):
        # Process items that don't need encoding
        for row in batch_rows:
            pov_path = row.get('pov_path') or row.get('pov_image', '')
            if not pov_path or int(row.get('is_empty', 0)) or not Path(pov_path).exists():
                out = row.copy()
                out['embedding_path'] = ''
                output_rows.append(out)
                skipped += 1
        
        # Process batch with model
        results = _process_batch_with_model(
            batch_rows, model, transform, device,
            process_pov_item, encode_povs
        )
        
        for row, emb in results:
            pov_path_obj = Path(row.get('pov_path') or row.get('pov_image', ''))
            embedding_path = pov_path_obj.with_suffix('.pt')
            
            if save_format == "pt":
                torch.save(emb, embedding_path)
            else:  # npy
                np.save(embedding_path.with_suffix('.npy'), emb.numpy())
            
            out = row.copy()
            out['embedding_path'] = str(embedding_path)
            output_rows.append(out)
            processed += 1
    
    # Write output manifest
    fieldnames = list(rows[0].keys()) + ['embedding_path']
    create_manifest(output_rows, output_manifest, fieldnames)
    
    print(f"\n✓ Processed {processed}/{len(rows)} POV images successfully")
    print(f"✓ Skipped {skipped} images (empty or errors)")
    print(f"✓ Output manifest: {output_manifest}")


# =============================================================================
# Graph Embeddings
# =============================================================================

def create_graph_embeddings(manifest_path: Path, taxonomy_path: Path, 
                           output_manifest: Path, model_name: str = "all-MiniLM-L6-v2",
                           save_format: str = "pt"):
    # Load taxonomy
    print(f"Loading taxonomy from {taxonomy_path}")
    taxonomy = Taxonomy(taxonomy_path)
    
    # Load embedding model
    print(f"Loading SentenceTransformer model: {model_name}")
    embedder = load_sentence_transformer_model(model_name)
    
    # Read manifest
    print(f"Reading manifest: {manifest_path}")
    rows = read_manifest(manifest_path)
    
    print(f"Found {len(rows)} graphs to process")
    
    # Process each graph
    output_rows = []
    skipped = 0
    
    for row in tqdm(rows, desc="Processing graphs"):
        graph_path = row.get('graph_path', '')
        if not graph_path:
            skipped += 1
            output_row = row.copy()
            output_row['embedding_path'] = ''
            output_rows.append(output_row)
            continue
        
        try:
            text = graph2text(graph_path, taxonomy)
            
            if not text:
                print(f"Warning: Empty text for {graph_path}")
                skipped += 1
                output_row = row.copy()
                output_row['embedding_path'] = ''
                output_rows.append(output_row)
                continue
            
            # Generate embedding
            embedding = embedder.encode(text, normalize_embeddings=True)
            
            # Determine save path
            graph_path_obj = Path(graph_path)
            if save_format == "pt":
                embedding_path = graph_path_obj.with_suffix('.pt')
                torch.save(torch.from_numpy(embedding), embedding_path)
            else:  # npy
                embedding_path = graph_path_obj.with_suffix('.npy')
                np.save(embedding_path, embedding)
            
            # Add to output manifest
            output_row = row.copy()
            output_row['embedding_path'] = str(embedding_path)
            output_rows.append(output_row)
            
        except Exception as e:
            print(f"Error processing {graph_path}: {e}")
            skipped += 1
            output_row = row.copy()
            output_row['embedding_path'] = ''
            output_rows.append(output_row)
    
    # Write output manifest
    fieldnames = list(rows[0].keys()) + ['embedding_path']
    create_manifest(output_rows, output_manifest, fieldnames)
    
    print(f"\n✓ Processed {len(rows) - skipped}/{len(rows)} graphs successfully")
    print(f"✓ Skipped {skipped} graphs")
    print(f"✓ Output manifest: {output_manifest}")


def create_graph_text_files(manifest_path: Path, taxonomy_path: Path):
    print(f"Loading taxonomy from {taxonomy_path}")
    taxonomy = Taxonomy(taxonomy_path)
    
    print(f"Reading manifest: {manifest_path}")
    rows = read_manifest(manifest_path)
    
    print(f"Found {len(rows)} graphs to process")
    
    skipped = 0
    for row in tqdm(rows, desc="Creating text files"):
        graph_path = row['graph_path']
        
        try:
            text = graph2text(graph_path, taxonomy)
            
            if not text:
                print(f"Warning: Empty text for {graph_path}")
                skipped += 1
                continue
            
            txt_path = Path(graph_path).with_suffix('.txt')
            txt_path.write_text(text, encoding='utf-8')
            
        except Exception as e:
            print(f"Error processing {graph_path}: {e}")
            skipped += 1
    
    print(f"\n✓ Processed {len(rows) - skipped}/{len(rows)} graphs successfully")
    print(f"✓ Skipped {skipped} graphs")


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified embedding creation for layouts, POVs, and graphs"
    )
    
    # Embedding type
    parser.add_argument(
        "--type",
        required=True,
        choices=["layout", "pov", "graph", "graph_text"],
        help="Type of embeddings to create (or graph_text for text files only)"
    )
    
    # Common arguments
    parser.add_argument(
        "--manifest",
        help="Path to input manifest CSV (required for layout manifest-based, pov and graph types)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for output manifest or directory"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--format",
        choices=["pt", "npy"],
        default="pt",
        help="Embedding save format: pt (PyTorch) or npy (NumPy)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing embedding files"
    )
    
    # Layout-specific arguments
    parser.add_argument(
        "--autoencoder-config",
        help="Path to Autoencoder config YAML (required for layout type)"
    )
    parser.add_argument(
        "--autoencoder-checkpoint",
        help="Path to Autoencoder checkpoint (required for layout type)"
    )
    parser.add_argument(
        "--data-root",
        help="Root folder containing scenes/rooms (for layout directory-based workflow)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of DataLoader workers (for layout manifest-based workflow)"
    )
    parser.add_argument(
        "--output-manifest",
        help="Output manifest path (for layout manifest-based workflow)"
    )
    parser.add_argument(
        "--output-latent-dir",
        default=None,
        help="Directory to save latent files (default: uses dataset structure, set to experiment/embeddings/latents for diffusion)"
    )
    parser.add_argument(
        "--diffusion-config",
        default=None,
        help="Path to diffusion config YAML (optional, used to get filters for whiteness filtering during embedding)"
    )
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="Output manifest filename (for layout directory-based workflow, saved in data-root)"
    )
    
    # Graph-specific arguments
    parser.add_argument(
        "--taxonomy",
        help="Path to taxonomy.json (required for graph and graph_text types)"
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name (for graph type, default: all-MiniLM-L6-v2)"
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    
    if args.type == "layout":
        if not args.autoencoder_config or not args.autoencoder_checkpoint:
            parser.error("--autoencoder-config and --autoencoder-checkpoint are required for layout type")
        
        print(f"[INFO] Using device: {device}")
        model = load_autoencoder_model(args.autoencoder_config, args.autoencoder_checkpoint, device=device)
        
        # Determine workflow: manifest-based or directory-based
        if args.manifest and args.output_manifest:
            # Manifest-based workflow (preferred for diffusion)
            print("[INFO] Using manifest-based workflow")
            # Determine output latent directory
            output_latent_dir = args.output_latent_dir
            if output_latent_dir is None:
                # Default: use embeddings/latents subdirectory in output manifest directory
                output_latent_dir = Path(args.output_manifest).parent / "latents"
            else:
                output_latent_dir = Path(output_latent_dir)
            
            create_layout_embeddings_from_manifest(
                encoder=model.encoder,
                manifest_path=args.manifest,
                output_manifest_path=args.output_manifest,
                batch_size=args.batch_size,
                autoencoder_config_path=args.autoencoder_config,
                num_workers=args.num_workers,
                overwrite=args.overwrite,
                device=str(device),
                output_latent_dir=output_latent_dir,
                diffusion_config_path=args.diffusion_config
            )
        elif args.data_root:
            # Directory-based workflow (legacy)
            print("[INFO] Using directory-based workflow")
            manifest_data = create_layout_embeddings_from_directory(
                model,
                Path(args.data_root),
                device=str(device),
                batch_size=args.batch_size,
                overwrite=args.overwrite
            )
            
            if manifest_data and args.manifest_out:
                df = pd.DataFrame(manifest_data)
                cols = ["scene", "type", "room_id", "layout_path", "layout_emb_path"]
                df = df[cols]
                output_csv_path = Path(args.data_root) / args.manifest_out
                df.to_csv(output_csv_path, index=False, sep="|")
                print(f"\n[INFO] Manifest saved to {output_csv_path}")
        else:
            parser.error("For layout type, either (--manifest + --output-manifest) or --data-root must be provided")
    
    elif args.type == "pov":
        if not args.manifest:
            parser.error("--manifest is required for pov type")
        
        create_pov_embeddings(
            manifest_path=Path(args.manifest),
            output_manifest=Path(args.output),
            save_format=args.format,
            batch_size=args.batch_size
        )
    
    elif args.type == "graph":
        if not args.manifest or not args.taxonomy:
            parser.error("--manifest and --taxonomy are required for graph type")
        
        create_graph_embeddings(
            manifest_path=Path(args.manifest),
            taxonomy_path=Path(args.taxonomy),
            output_manifest=Path(args.output),
            model_name=args.model,
            save_format=args.format
        )
    
    elif args.type == "graph_text":
        if not args.manifest or not args.taxonomy:
            parser.error("--manifest and --taxonomy are required for graph_text type")
        
        create_graph_text_files(
            Path(args.manifest),
            Path(args.taxonomy)
        )


if __name__ == "__main__":
    main()
