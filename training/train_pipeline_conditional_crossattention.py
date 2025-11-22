#!/usr/bin/env python3
"""
Complete training pipeline for conditional cross-attention diffusion.

This script:
1. Trains the 32x32 VAE autoencoder
2. Embeds the ControlNet dataset with the new VAE (preserves existing embeddings)
3. Calculates scale_factor from embedded latents
4. Updates diffusion config with VAE checkpoint, embedded manifest, and scale_factor
5. Trains the conditional cross-attention diffusion model

Usage:
    python training/train_pipeline_conditional_crossattention.py \
        --ae-config <autoencoder_config> \
        --diffusion-config <diffusion_config> \
        [--skip-ae] [--skip-embedding] [--skip-training]
"""

import torch
import sys
import os
import yaml
import argparse
import pandas as pd
import subprocess
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from itertools import islice

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.utils import (
    set_deterministic,
    load_config,
    get_device,
)
from training.train import main as train_ae_main
from training.train_diffusion import main as train_diffusion_main, calculate_scale_factor_from_dataset
from models.datasets.datasets import ManifestDataset
from models.autoencoder import Autoencoder
from data_preparation.create_embeddings import (
    create_layout_embeddings_from_manifest,
    load_resnet_model,
    load_sentence_transformer_model,
)
from common.file_io import read_manifest, create_manifest


def find_ae_checkpoint(ae_config):
    """
    Find the best checkpoint path from autoencoder config.
    
    Args:
        ae_config: Autoencoder config dictionary
        
    Returns:
        Path to best checkpoint (absolute), or None if not found
    """
    exp_name = ae_config.get("experiment", {}).get("name", "unnamed")
    save_path = ae_config.get("experiment", {}).get("save_path")
    
    if save_path is None:
        save_path = Path("outputs") / exp_name
    else:
        save_path = Path(save_path)
    
    # Resolve to absolute path
    save_path = save_path.resolve()
    
    # Check for best checkpoint
    best_checkpoint = save_path / f"{exp_name}_checkpoint_best.pt"
    
    if best_checkpoint.exists():
        return best_checkpoint.resolve()
    
    # Also check in checkpoints subdirectory
    checkpoint_dir = save_path / "checkpoints"
    best_checkpoint = checkpoint_dir / f"{exp_name}_checkpoint_best.pt"
    
    if best_checkpoint.exists():
        return best_checkpoint.resolve()
    
    # Also check for latest checkpoint as fallback
    latest_checkpoint = save_path / f"{exp_name}_checkpoint_latest.pt"
    if latest_checkpoint.exists():
        print(f"Warning: Best checkpoint not found, using latest: {latest_checkpoint}")
        return latest_checkpoint.resolve()
    
    latest_checkpoint = checkpoint_dir / f"{exp_name}_checkpoint_latest.pt"
    if latest_checkpoint.exists():
        print(f"Warning: Best checkpoint not found, using latest: {latest_checkpoint}")
        return latest_checkpoint.resolve()
    
    return None


def train_autoencoder(ae_config_path):
    """
    Train autoencoder using the training script, or use existing checkpoint if available.
    
    Args:
        ae_config_path: Path to autoencoder config YAML
        
    Returns:
        Path to best checkpoint, or None if training failed
    """
    print(f"\n{'='*60}")
    print("PHASE 1: Training 32x32 VAE Autoencoder")
    print(f"{'='*60}")
    print(f"Config: {ae_config_path}")
    print(f"{'='*60}\n")
    
    try:
        # First, check if best checkpoint already exists
        ae_config = load_config(ae_config_path)
        existing_checkpoint = find_ae_checkpoint(ae_config)
        
        if existing_checkpoint is not None:
            print(f"Found existing best checkpoint: {existing_checkpoint}")
            print("Skipping VAE training - using existing checkpoint")
            print(f"{'='*60}\n")
            return existing_checkpoint
        
        # No checkpoint found, proceed with training
        print("No existing checkpoint found. Starting VAE training...")
        print()
        
        base_dir = Path(__file__).parent.parent
        ae_config_abs = Path(ae_config_path).resolve()
        train_script = base_dir / "training" / "train.py"
        
        result = subprocess.run(
            [sys.executable, str(train_script), str(ae_config_abs)],
            check=True,
            cwd=base_dir
        )
        
        print(f"\n{'='*60}")
        print("Autoencoder training completed successfully!")
        print(f"{'='*60}\n")
        
        # Find the checkpoint after training
        checkpoint_path = find_ae_checkpoint(ae_config)
        
        if checkpoint_path is None:
            print("WARNING: Could not find autoencoder checkpoint after training!")
            return None
        
        print(f"Found autoencoder checkpoint: {checkpoint_path}")
        return checkpoint_path
        
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Autoencoder training failed with exit code {e.returncode}")
        print(f"{'='*60}\n")
        return None
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Autoencoder training failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        return None


def embed_controlnet_dataset_with_vae(
    ae_checkpoint_path,
    ae_config_path,
    input_manifest_path,
    output_manifest_path,
    batch_size=32,
    num_workers=8
):
    """
    Embed ControlNet dataset from scratch: layouts, POVs, and graphs.
    
    This function:
    1. Embeds layouts using the VAE to create latent_path
    2. Embeds POVs using ResNet18 to create pov_embedding_path
    3. Embeds graphs using SentenceTransformer to create graph_embedding_path
    4. Creates a new manifest with all three: latent_path, graph_embedding_path, pov_embedding_path
    
    Args:
        ae_checkpoint_path: Path to VAE checkpoint
        ae_config_path: Path to VAE config
        input_manifest_path: Path to ControlNet manifest (with layout_path, pov_path, graph_text_path)
        output_manifest_path: Path to output manifest (with all embeddings)
        batch_size: Batch size for encoding
        num_workers: Number of workers
        
    Returns:
        True if embedding succeeded, False otherwise
    """
    print(f"\n{'='*60}")
    print("PHASE 2: Embedding ControlNet Dataset (Layouts + POVs + Graphs)")
    print(f"{'='*60}")
    print(f"VAE checkpoint: {ae_checkpoint_path}")
    print(f"Input manifest: {input_manifest_path}")
    print(f"Output manifest: {output_manifest_path}")
    print(f"{'='*60}\n")
    
    try:
        ae_checkpoint_abs = Path(ae_checkpoint_path).resolve()
        ae_config_abs = Path(ae_config_path).resolve()
        input_manifest_abs = Path(input_manifest_path).resolve()
        output_manifest_abs = Path(output_manifest_path).resolve()
        
        # Ensure output directory exists
        output_manifest_abs.parent.mkdir(parents=True, exist_ok=True)
        latents_dir = output_manifest_abs.parent / "latents"
        pov_embeddings_dir = output_manifest_abs.parent / "embeddings" / "povs"
        graph_embeddings_dir = output_manifest_abs.parent / "embeddings" / "graphs"
        
        latents_dir.mkdir(parents=True, exist_ok=True)
        pov_embeddings_dir.mkdir(parents=True, exist_ok=True)
        graph_embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directories:")
        print(f"  Latents: {latents_dir}")
        print(f"  POV embeddings: {pov_embeddings_dir}")
        print(f"  Graph embeddings: {graph_embeddings_dir}")
        print()
        
        # Read input manifest
        rows = read_manifest(input_manifest_abs)
        print(f"Loaded manifest with {len(rows)} samples")
        
        # Step 1: Embed layouts using VAE
        print(f"\n{'='*60}")
        print("Step 1/3: Embedding layouts with VAE")
        print(f"{'='*60}")
        
        # Load autoencoder
        print(f"Loading autoencoder from: {ae_checkpoint_abs}")
        autoencoder = Autoencoder.load_checkpoint(ae_checkpoint_abs, map_location="cpu")
        autoencoder.eval()
        
        # Get autoencoder config path
        checkpoint_path = Path(ae_checkpoint_abs)
        possible_configs = [
            checkpoint_path.parent / f"{checkpoint_path.stem.replace('_checkpoint_best', '').replace('_checkpoint_latest', '')}.yaml",
            checkpoint_path.parent.parent / "experiment_config.yaml",
            ae_config_abs,
        ]
        autoencoder_config_path = None
        for pc in possible_configs:
            if pc.exists():
                autoencoder_config_path = str(pc)
                break
        
        if not autoencoder_config_path:
            print("[WARNING] Could not find autoencoder config, using defaults for transform")
        
        # Create temporary manifest for layouts
        layout_temp_manifest = output_manifest_abs.parent / "temp_layout_manifest.csv"
        layout_output_manifest = output_manifest_abs.parent / "temp_layout_output_manifest.csv"
        
        layout_rows = [
            {"layout_path": row.get("layout_path", ""), "is_empty": 0}
            for row in rows
            if row.get("layout_path")
        ]
        
        layout_emb_mapping = {}
        if layout_rows:
            create_manifest(layout_rows, layout_temp_manifest, ["layout_path", "is_empty"])
            
            create_layout_embeddings_from_manifest(
                encoder=autoencoder.encoder,
                manifest_path=layout_temp_manifest,
                output_manifest_path=layout_output_manifest,
                batch_size=batch_size,
                num_workers=num_workers,
                overwrite=False,
                device="cuda",
                autoencoder_config_path=autoencoder_config_path,
                output_latent_dir=str(latents_dir),
                diffusion_config_path=None
            )
            
            # Read layout embeddings mapping
            layout_emb_df = pd.read_csv(layout_output_manifest)
            layout_emb_mapping = dict(zip(layout_emb_df["layout_path"], layout_emb_df["latent_path"]))
            
            # Clean up autoencoder from GPU
            print("Cleaning up autoencoder from GPU...")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                autoencoder.encoder = autoencoder.encoder.cpu()
                torch.cuda.synchronize()
            del autoencoder
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # Step 2: Embed POVs using ResNet18
        print(f"\n{'='*60}")
        print("Step 2/3: Embedding POVs with ResNet18")
        print(f"{'='*60}")
        
        pov_rows = [
            row for row in rows
            if row.get("pov_path") and row.get("pov_path") != "0"
        ]
        
        pov_emb_mapping = {}
        if pov_rows:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading ResNet18 model on {device}...")
            model = load_resnet_model(device)
            transform = T.Compose([
                T.Resize((512, 512)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            
            def process_pov_item(row):
                pov_path = row.get("pov_path", "")
                if not pov_path or not Path(pov_path).exists():
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
            
            def batched(iterable, n):
                it = iter(iterable)
                while True:
                    batch = list(islice(it, n))
                    if not batch:
                        break
                    yield batch
            
            processed = 0
            for batch_rows in tqdm(batched(pov_rows, batch_size), desc="Embedding POVs"):
                batch_imgs = []
                valid_rows = []
                
                for row in batch_rows:
                    img, row_data = process_pov_item(row)
                    if img is not None:
                        batch_imgs.append(transform(img))
                        valid_rows.append(row_data)
                
                if not valid_rows:
                    continue
                
                imgs_tensor = torch.stack(batch_imgs).to(device)
                with torch.no_grad():
                    embeddings = encode_povs(imgs_tensor)
                
                for row_data, emb in zip(valid_rows, embeddings):
                    pov_path = row_data["pov_path"]
                    pov_filename = Path(pov_path).name
                    emb_filename = pov_filename.replace(".png", ".pt")
                    emb_path = pov_embeddings_dir / emb_filename
                    
                    emb_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(emb.cpu(), emb_path)
                    pov_emb_mapping[pov_path] = str(emb_path.resolve())
                    processed += 1
            
            print(f"✓ Embedded {processed}/{len(pov_rows)} POV images")
            
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Step 3: Embed graphs using SentenceTransformer
        print(f"\n{'='*60}")
        print("Step 3/3: Embedding graphs with SentenceTransformer")
        print(f"{'='*60}")
        
        graph_rows = [
            row for row in rows
            if row.get("graph_text_path")
        ]
        
        graph_emb_mapping = {}
        if graph_rows:
            print("Loading SentenceTransformer model...")
            embedder = load_sentence_transformer_model("all-MiniLM-L6-v2")
            
            processed = 0
            skipped = 0
            
            for row in tqdm(graph_rows, desc="Embedding graphs"):
                graph_text_path = row.get("graph_text_path", "")
                if not graph_text_path:
                    skipped += 1
                    continue
                
                try:
                    text_path = Path(graph_text_path)
                    if not text_path.exists():
                        print(f"Warning: Text file not found: {text_path}")
                        skipped += 1
                        continue
                    
                    text = text_path.read_text(encoding="utf-8")
                    if not text:
                        print(f"Warning: Empty text for {text_path}")
                        skipped += 1
                        continue
                    
                    embedding = embedder.encode(text, normalize_embeddings=True)
                    
                    graph_filename = Path(graph_text_path).name
                    emb_filename = graph_filename.replace(".txt", ".pt")
                    emb_path = graph_embeddings_dir / emb_filename
                    
                    emb_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(torch.from_numpy(embedding), emb_path)
                    graph_emb_mapping[graph_text_path] = str(emb_path.resolve())
                    processed += 1
                    
                except Exception as e:
                    print(f"Error processing {graph_text_path}: {e}")
                    skipped += 1
            
            print(f"✓ Embedded {processed}/{len(graph_rows)} graphs (skipped {skipped})")
            
            del embedder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Step 4: Create zero embeddings for scenes (pov_path == "0")
        print(f"\n{'='*60}")
        print("Step 4/4: Creating zero embeddings for scenes")
        print(f"{'='*60}")
        
        pov_emb_dim = 512  # ResNet18 output dimension
        if pov_emb_mapping:
            sample_emb_path = list(pov_emb_mapping.values())[0]
            sample_emb = torch.load(sample_emb_path)
            pov_emb_dim = sample_emb.shape[0] if sample_emb.dim() == 1 else sample_emb.numel()
        
        zero_emb_path = pov_embeddings_dir / "zero_embedding.pt"
        zero_emb = torch.zeros(pov_emb_dim)
        torch.save(zero_emb, zero_emb_path)
        print(f"Created zero embedding: {zero_emb_path} (shape: {zero_emb.shape})")
        
        # Step 5: Create final manifest with all embeddings
        print(f"\n{'='*60}")
        print("Step 5/5: Creating final manifest")
        print(f"{'='*60}")
        
        output_rows = []
        for row in rows:
            output_row = row.copy()
            
            # Add layout latent path
            layout_path = row.get("layout_path", "")
            if layout_path:
                output_row["latent_path"] = layout_emb_mapping.get(layout_path, "")
            else:
                output_row["latent_path"] = ""
            
            # Add POV embedding path
            pov_path = row.get("pov_path", "")
            if pov_path and pov_path != "0":
                output_row["pov_embedding_path"] = pov_emb_mapping.get(pov_path, "")
            elif pov_path == "0":
                output_row["pov_embedding_path"] = str(zero_emb_path.resolve())
            else:
                output_row["pov_embedding_path"] = ""
            
            # Add graph embedding path
            graph_text_path = row.get("graph_text_path", "")
            if graph_text_path:
                output_row["graph_embedding_path"] = graph_emb_mapping.get(graph_text_path, "")
            else:
                output_row["graph_embedding_path"] = ""
            
            output_rows.append(output_row)
        
        # Write final manifest
        output_manifest_abs.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys()) + ["latent_path", "pov_embedding_path", "graph_embedding_path"]
        create_manifest(output_rows, output_manifest_abs, fieldnames)
        
        # Clean up temporary files
        for temp_file in [layout_temp_manifest, layout_output_manifest]:
            if temp_file.exists():
                temp_file.unlink()
        
        print(f"\n{'='*60}")
        print("Dataset embedding completed successfully!")
        print(f"Output manifest: {output_manifest_abs}")
        print(f"  - latent_path: Created from 32x32 VAE")
        print(f"  - graph_embedding_path: Created from SentenceTransformer")
        print(f"  - pov_embedding_path: Created from ResNet18")
        print(f"{'='*60}\n")
        
        # Final GPU cleanup
        if torch.cuda.is_available():
            print("Performing final GPU cleanup...")
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print("Waiting 5 seconds for GPU to fully release resources...")
            time.sleep(5)
            try:
                torch.cuda.synchronize()
            except:
                pass
            print("✓ GPU memory fully cleared")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Dataset embedding failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        return False


def calculate_and_update_scale_factor(diffusion_config_path, embedded_manifest_path):
    """
    Calculate scale_factor from embedded latents and update config.
    
    Args:
        diffusion_config_path: Path to diffusion config
        embedded_manifest_path: Path to manifest with embedded latents
        
    Returns:
        Calculated scale_factor, or None if failed
    """
    print(f"\n{'='*60}")
    print("PHASE 3: Calculating Scale Factor")
    print(f"{'='*60}")
    
    try:
        # Build dataset to calculate scale factor
        from models.datasets.datasets import ManifestDataset
        
        dataset_config = {
            "manifest": str(embedded_manifest_path),
            "outputs": {"latent": "latent_path"},
            "filters": {"is_empty": [False]},
            "return_path": False
        }
        
        dataset = ManifestDataset.from_config(dataset_config)
        
        # Calculate scale factor
        scale_factor = calculate_scale_factor_from_dataset(
            dataset,
            num_samples=100,
            seed=42
        )
        
        # Update config
        with open(diffusion_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config["scale_factor"] = scale_factor
        
        with open(diffusion_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        print(f"Calculated scale_factor: {scale_factor:.6f}")
        print(f"Updated config: {diffusion_config_path}")
        print(f"{'='*60}\n")
        
        return scale_factor
        
    except Exception as e:
        print(f"WARNING: Failed to calculate scale_factor: {e}")
        print("  Will use default scale_factor=1.0 or value from config")
        print(f"{'='*60}\n")
        return None


class NumpySafeLoader(yaml.SafeLoader):
    """Custom YAML loader that handles numpy scalar types."""
    pass


def numpy_scalar_constructor(loader, node):
    """Convert numpy scalar tags to Python native types."""
    # For python/object/apply tags, the node contains a sequence with the function and args
    # numpy.core.multiarray.scalar is called with the value as argument
    try:
        # Construct the sequence which contains [numpy.core.multiarray.scalar, value]
        sequence = loader.construct_sequence(node)
        if len(sequence) >= 2:
            # The value is the second element (first is the function/class)
            value = sequence[1]
            # Convert numpy types to Python native types
            if isinstance(value, (np.integer, np.floating, np.ndarray)):
                return value.item() if hasattr(value, 'item') else float(value)
            return value
        return sequence[0] if sequence else None
    except Exception:
        # Fallback: try to construct as sequence and take first value
        try:
            sequence = loader.construct_sequence(node)
            return sequence[0] if sequence else None
        except Exception:
            return None


# Register the constructor for numpy scalar types
NumpySafeLoader.add_constructor(
    'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar',
    numpy_scalar_constructor
)


class NumpyFullLoader(yaml.FullLoader):
    """Custom YAML FullLoader that handles numpy scalar types."""
    pass


# Register the constructor for numpy scalar types in FullLoader too
NumpyFullLoader.add_constructor(
    'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar',
    numpy_scalar_constructor
)


def read_flags(flags_file):
    """Read pipeline flags from flags.txt file."""
    flags = {}
    if flags_file.exists():
        with open(flags_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    flags[key.strip()] = value.strip()
    return flags


def write_flag(flags_file, key, value):
    """Write a single flag to flags.txt file."""
    flags = read_flags(flags_file)
    flags[key] = value
    
    flags_file.parent.mkdir(parents=True, exist_ok=True)
    with open(flags_file, 'w') as f:
        f.write("# Pipeline progress flags\n")
        f.write("# Format: KEY=value\n")
        f.write("# Auto-generated - do not edit manually\n\n")
        for k, v in sorted(flags.items()):
            f.write(f"{k}={v}\n")


def check_flag(flags_file, key):
    """Check if a flag is set and return its value, or None if not set."""
    flags = read_flags(flags_file)
    return flags.get(key, None)


def update_diffusion_config(
    diffusion_config_path,
    ae_checkpoint_path,
    embedded_manifest_path,
    scale_factor=None
):
    """
    Update diffusion config with VAE checkpoint, embedded manifest, and scale_factor.
    
    Args:
        diffusion_config_path: Path to diffusion config
        ae_checkpoint_path: Path to VAE checkpoint
        embedded_manifest_path: Path to embedded manifest
        scale_factor: Optional scale_factor to add
    """
    print(f"\n{'='*60}")
    print("PHASE 4: Updating Diffusion Config")
    print(f"{'='*60}")
    
    # Try loading with custom loader first, fallback to FullLoader if it fails
    try:
        with open(diffusion_config_path, 'r') as f:
            config = yaml.load(f, Loader=NumpySafeLoader)
    except (yaml.constructor.ConstructorError, yaml.YAMLError) as e:
        # If custom loader fails (e.g., other numpy types), try FullLoader with numpy support
        print(f"Warning: Custom loader failed, trying FullLoader with numpy support: {e}")
    with open(diffusion_config_path, 'r') as f:
            config = yaml.load(f, Loader=NumpyFullLoader)
    
    # Validate checkpoint path exists
    ae_checkpoint_resolved = Path(ae_checkpoint_path).resolve()
    if not ae_checkpoint_resolved.exists():
        error_msg = (
            f"\n{'='*60}\n"
            f"ERROR: Autoencoder checkpoint not found!\n"
            f"  Expected path: {ae_checkpoint_resolved}\n"
            f"  Original path: {ae_checkpoint_path}\n"
            f"{'='*60}\n"
            f"Please ensure:\n"
            f"  1. Autoencoder training completed successfully\n"
            f"  2. Checkpoint was saved to the expected location\n"
            f"  3. If using --skip-ae, verify the checkpoint path is correct\n"
        )
        raise FileNotFoundError(error_msg)
    
    # Validate embedded manifest exists
    embedded_manifest_resolved = Path(embedded_manifest_path).resolve()
    if not embedded_manifest_resolved.exists():
        error_msg = (
            f"\n{'='*60}\n"
            f"ERROR: Embedded manifest not found!\n"
            f"  Expected path: {embedded_manifest_resolved}\n"
            f"  Original path: {embedded_manifest_path}\n"
            f"{'='*60}\n"
            f"Please ensure:\n"
            f"  1. Dataset embedding completed successfully\n"
            f"  2. If using --skip-embedding, verify the manifest path is correct\n"
        )
        raise FileNotFoundError(error_msg)
    
    # Update autoencoder checkpoint
    if 'autoencoder' not in config:
        config['autoencoder'] = {}
    config['autoencoder']['checkpoint'] = str(ae_checkpoint_resolved)
    
    # Update dataset manifest
    if 'dataset' not in config:
        config['dataset'] = {}
    config['dataset']['manifest'] = str(embedded_manifest_resolved)
    config['dataset']['outputs'] = {
        'latent': 'latent_path',
        'text_emb': 'graph_embedding_path',
        'pov_emb': 'pov_embedding_path'
    }
    
    # Update scale_factor if provided
    if scale_factor is not None:
        config['scale_factor'] = scale_factor
    
    # Save updated config
    with open(diffusion_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"Updated config:")
    print(f"  - autoencoder.checkpoint: {ae_checkpoint_resolved}")
    print(f"  - dataset.manifest: {embedded_manifest_resolved}")
    print(f"  - dataset.outputs: latent, text_emb, pov_emb")
    if scale_factor:
        print(f"  - scale_factor: {scale_factor:.6f}")
    print(f"{'='*60}\n")


def train_diffusion(diffusion_config_path):
    """
    Train conditional cross-attention diffusion model.
    
    Args:
        diffusion_config_path: Path to diffusion config
        
    Returns:
        True if training succeeded, False otherwise
    """
    print(f"\n{'='*60}")
    print("PHASE 5: Training Conditional Cross-Attention Diffusion")
    print(f"{'='*60}")
    print(f"Config: {diffusion_config_path}")
    print(f"{'='*60}\n")
    
    try:
        base_dir = Path(__file__).parent.parent
        diffusion_config_abs = Path(diffusion_config_path).resolve()
        train_script = base_dir / "training" / "train_diffusion.py"
        
        result = subprocess.run(
            [sys.executable, str(train_script), str(diffusion_config_abs), "--resume"],
            check=True,
            cwd=base_dir
        )
        
        print(f"\n{'='*60}")
        print("Diffusion training completed successfully!")
        print(f"{'='*60}\n")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Diffusion training failed with exit code {e.returncode}")
        print(f"{'='*60}\n")
        return False
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Diffusion training failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline for conditional cross-attention diffusion"
    )
    parser.add_argument(
        "--ae-config",
        type=Path,
        required=True,
        help="Path to 32x32 VAE autoencoder config YAML"
    )
    parser.add_argument(
        "--diffusion-config",
        type=Path,
        required=True,
        help="Path to conditional cross-attention diffusion config YAML"
    )
    parser.add_argument(
        "--controlnet-manifest",
        type=Path,
        help="Path to ControlNet manifest with embeddings (default: from diffusion config)"
    )
    parser.add_argument(
        "--skip-ae",
        action="store_true",
        help="Skip VAE training (use existing checkpoint)"
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip dataset embedding (use existing embedded manifest)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip diffusion training (only do VAE and embedding)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding (default: 32)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers for data loading (default: 8)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Conditional Cross-Attention Diffusion Pipeline")
    print("="*60)
    print(f"VAE config: {args.ae_config}")
    print(f"Diffusion config: {args.diffusion_config}")
    print("="*60)
    print()
    
    # Load configs
    ae_config = load_config(args.ae_config)
    diffusion_config = load_config(args.diffusion_config)
    
    # Get paths - resolve to absolute paths immediately
    ae_exp_name = ae_config.get("experiment", {}).get("name", "unnamed")
    ae_save_path_raw = ae_config.get("experiment", {}).get("save_path", "outputs")
    ae_save_path = Path(ae_save_path_raw).resolve()
    
    diffusion_exp_name = diffusion_config.get("experiment", {}).get("name", "unnamed")
    diffusion_save_path_raw = diffusion_config.get("experiment", {}).get("save_path", "outputs")
    diffusion_save_path = Path(diffusion_save_path_raw).resolve()
    
    # Get ControlNet manifest path
    if args.controlnet_manifest:
        controlnet_manifest = Path(args.controlnet_manifest).resolve()
    else:
        # Try to get from diffusion config
        manifest_path_str = diffusion_config.get("dataset", {}).get("manifest", "")
        if manifest_path_str:
            controlnet_manifest = Path(manifest_path_str)
            # Resolve relative paths relative to config file location
            if not controlnet_manifest.is_absolute():
                config_dir = Path(args.diffusion_config).parent
                controlnet_manifest = (config_dir / controlnet_manifest).resolve()
            else:
                controlnet_manifest = controlnet_manifest.resolve()
        else:
            controlnet_manifest = None
    
    # Validate manifest exists
    if controlnet_manifest is None or not controlnet_manifest.exists():
        error_msg = (
            f"\n{'='*60}\n"
            f"ERROR: ControlNet manifest not found\n"
        )
        if controlnet_manifest:
            error_msg += f"  Path: {controlnet_manifest}\n"
        else:
            error_msg += (
                f"  No manifest path provided.\n"
                f"  Please provide one of:\n"
                f"    1. --controlnet-manifest <path> argument\n"
                f"    2. dataset.manifest in diffusion config\n"
            )
        error_msg += f"{'='*60}\n"
        raise FileNotFoundError(error_msg)
    
    # Output manifest (with embedded latents + preserved embeddings)
    embeddings_dir = diffusion_save_path / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    embedded_manifest = embeddings_dir / "manifest_with_latents_and_embeddings.csv"
    
    # Pipeline flags file for resuming
    flags_file = diffusion_save_path / "flags.txt"
    
    print(f"\n{'='*60}")
    print("Pipeline Resume Check")
    print(f"{'='*60}")
    print(f"Flags file: {flags_file}")
    if flags_file.exists():
        flags = read_flags(flags_file)
        print(f"Found {len(flags)} saved flags")
        for key, value in sorted(flags.items()):
            print(f"  {key}: {value}")
    else:
        print("No existing flags file - starting fresh pipeline")
    print(f"{'='*60}\n")
    
    # Step 1: Train VAE (or find existing)
    ae_checkpoint = None
    saved_ae_checkpoint = check_flag(flags_file, "VAE_CHECKPOINT")
    
    if saved_ae_checkpoint and Path(saved_ae_checkpoint).exists():
        print(f"Resuming: Found saved VAE checkpoint in flags: {saved_ae_checkpoint}")
        ae_checkpoint = Path(saved_ae_checkpoint)
    elif not args.skip_ae:
        ae_checkpoint = train_autoencoder(args.ae_config)
        if ae_checkpoint is None:
            print("ERROR: VAE training failed or checkpoint not found")
            sys.exit(1)
        # Save checkpoint to flags
        write_flag(flags_file, "VAE_CHECKPOINT", str(ae_checkpoint.resolve()))
        write_flag(flags_file, "VAE_TRAINED", "true")
        print(f"Saved VAE checkpoint to flags: {ae_checkpoint}")
    else:
        ae_checkpoint = find_ae_checkpoint(ae_config)
        if ae_checkpoint is None:
            exp_name = ae_config.get("experiment", {}).get("name", "unnamed")
            save_path = Path(ae_config.get("experiment", {}).get("save_path", "outputs")).resolve()
            error_msg = (
                f"\n{'='*60}\n"
                f"ERROR: --skip-ae specified but no checkpoint found\n"
                f"  Experiment name: {exp_name}\n"
                f"  Save path: {save_path}\n"
                f"  Expected locations:\n"
                f"    - {save_path / f'{exp_name}_checkpoint_best.pt'}\n"
                f"    - {save_path / 'checkpoints' / f'{exp_name}_checkpoint_best.pt'}\n"
                f"    - {save_path / f'{exp_name}_checkpoint_latest.pt'}\n"
                f"    - {save_path / 'checkpoints' / f'{exp_name}_checkpoint_latest.pt'}\n"
                f"{'='*60}\n"
            )
            print(error_msg)
            sys.exit(1)
        print(f"Using existing VAE checkpoint: {ae_checkpoint}")
        write_flag(flags_file, "VAE_CHECKPOINT", str(ae_checkpoint.resolve()))
        write_flag(flags_file, "VAE_TRAINED", "true")
    
    # Step 2: Embed dataset
    # First check for shared embedding (created by standalone embedding script)
    SHARED_EMBEDDING_MANIFEST = Path("/work3/s233249/ImgiNav/experiments/shared_embeddings/manifest_with_embeddings.csv")
    
    saved_embedded_manifest = check_flag(flags_file, "EMBEDDED_MANIFEST")
    
    if saved_embedded_manifest and Path(saved_embedded_manifest).exists():
        print(f"Resuming: Found saved embedded manifest in flags: {saved_embedded_manifest}")
        embedded_manifest = Path(saved_embedded_manifest)
    elif SHARED_EMBEDDING_MANIFEST.exists():
        print(f"Found shared embedding manifest: {SHARED_EMBEDDING_MANIFEST}")
        print("Using shared embedding - skipping embedding step")
        embedded_manifest = SHARED_EMBEDDING_MANIFEST
        write_flag(flags_file, "EMBEDDED_MANIFEST", str(embedded_manifest.resolve()))
        write_flag(flags_file, "EMBEDDED", "true")
        write_flag(flags_file, "USED_SHARED_EMBEDDING", "true")
        print(f"Saved shared embedding manifest to flags: {embedded_manifest}")
    elif not args.skip_embedding:
        success = embed_controlnet_dataset_with_vae(
            ae_checkpoint,
            args.ae_config,
            controlnet_manifest,
            embedded_manifest,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        if not success:
            print("ERROR: Dataset embedding failed")
            sys.exit(1)
        # Save embedded manifest to flags
        write_flag(flags_file, "EMBEDDED_MANIFEST", str(embedded_manifest.resolve()))
        write_flag(flags_file, "EMBEDDED", "true")
        print(f"Saved embedded manifest to flags: {embedded_manifest}")
    else:
        if not embedded_manifest.exists():
            print(f"ERROR: --skip-embedding specified but embedded manifest not found: {embedded_manifest}")
            sys.exit(1)
        print(f"Using existing embedded manifest: {embedded_manifest}")
        write_flag(flags_file, "EMBEDDED_MANIFEST", str(embedded_manifest.resolve()))
        write_flag(flags_file, "EMBEDDED", "true")
    
    # Step 3: Calculate scale factor
    saved_scale_factor = check_flag(flags_file, "SCALE_FACTOR")
    
    if saved_scale_factor:
        print(f"Resuming: Found saved scale factor in flags: {saved_scale_factor}")
        scale_factor = float(saved_scale_factor)
    else:
        scale_factor = calculate_and_update_scale_factor(
            args.diffusion_config,
            embedded_manifest
        )
        if scale_factor is not None:
            write_flag(flags_file, "SCALE_FACTOR", str(scale_factor))
            write_flag(flags_file, "SCALE_FACTOR_CALCULATED", "true")
            print(f"Saved scale factor to flags: {scale_factor}")
    
    # Step 4: Update diffusion config
    config_updated = check_flag(flags_file, "CONFIG_UPDATED")
    
    if not config_updated:
        update_diffusion_config(
            args.diffusion_config,
            ae_checkpoint,
            embedded_manifest,
            scale_factor
        )
        write_flag(flags_file, "CONFIG_UPDATED", datetime.now().isoformat())
        print(f"Saved config update timestamp to flags")
    else:
        print(f"Resuming: Config already updated (timestamp: {config_updated})")
    
    # Step 5: Train diffusion
    diffusion_trained = check_flag(flags_file, "DIFFUSION_TRAINED")
    
    if diffusion_trained == "true":
        print(f"Resuming: Diffusion training already marked as complete in flags")
    elif not args.skip_training:
        success = train_diffusion(args.diffusion_config)
        if not success:
            print("ERROR: Diffusion training failed")
            sys.exit(1)
        write_flag(flags_file, "DIFFUSION_TRAINED", "true")
        write_flag(flags_file, "DIFFUSION_COMPLETED", datetime.now().isoformat())
        print(f"Saved diffusion training completion to flags")
    else:
        print("Skipping diffusion training (--skip-training)")
    
    print("\n" + "="*60)
    print("Pipeline COMPLETE - SUCCESS")
    print("="*60)
    print(f"VAE checkpoint: {ae_checkpoint}")
    print(f"Embedded manifest: {embedded_manifest}")
    print(f"Scale factor: {scale_factor:.6f if scale_factor else 'N/A'}")
    print(f"Diffusion config: {args.diffusion_config}")
    print(f"Flags file: {flags_file}")
    print("="*60)


if __name__ == "__main__":
    main()

