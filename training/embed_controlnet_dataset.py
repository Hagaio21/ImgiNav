#!/usr/bin/env python3
"""
Standalone script to embed ControlNet dataset with VAE, ResNet18, and SentenceTransformer.

This script creates embeddings that can be shared across all experiments:
1. Embeds layouts using the VAE to create latent_path
2. Embeds POVs using ResNet18 to create pov_embedding_path
3. Embeds graphs using SentenceTransformer to create graph_embedding_path
4. Creates a manifest with all three: latent_path, graph_embedding_path, pov_embedding_path

Usage:
    python training/embed_controlnet_dataset.py \
        --ae-checkpoint <vae_checkpoint> \
        --ae-config <vae_config> \
        --input-manifest <input_manifest> \
        --output-manifest <output_manifest> \
        [--batch-size 32] [--num-workers 8]
"""

import torch
import sys
import os
import argparse
import pandas as pd
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from itertools import islice

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.autoencoder import Autoencoder
from data_preparation.create_embeddings import (
    create_layout_embeddings_from_manifest,
    load_resnet_model,
    load_sentence_transformer_model,
)
from common.file_io import read_manifest, create_manifest


def embed_controlnet_dataset_with_vae(
    ae_checkpoint_path=None,
    ae_config_path=None,
    input_manifest_path=None,
    output_manifest_path=None,
    batch_size=32,
    num_workers=8,
    create_layout_embeddings=True,
    update_existing_manifest=False
):
    """
    Embed ControlNet dataset: layouts (optional), POVs, and graphs.
    
    This function:
    1. (Optional) Embeds layouts using the VAE to create latent_path (if ae_checkpoint_path provided)
    2. Embeds POVs using ResNet18 to create pov_embedding_path
    3. Embeds graphs using SentenceTransformer to create graph_embedding_path
    4. Creates a new manifest with: latent_path (if created), graph_embedding_path, pov_embedding_path
    
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
    print("Embedding ControlNet Dataset (Layouts + POVs + Graphs)")
    print(f"{'='*60}")
    print(f"VAE checkpoint: {ae_checkpoint_path}")
    print(f"Input manifest: {input_manifest_path}")
    print(f"Output manifest: {output_manifest_path}")
    print(f"{'='*60}\n")
    
    try:
        input_manifest_abs = Path(input_manifest_path).resolve()
        output_manifest_abs = Path(output_manifest_path).resolve()
        
        # Check if output already exists
        if output_manifest_abs.exists() and not update_existing_manifest:
            print(f"Output manifest already exists: {output_manifest_abs}")
            print("Skipping embedding - using existing manifest")
            print(f"{'='*60}\n")
            return True
        
        # If updating existing manifest, read it first
        if update_existing_manifest and output_manifest_abs.exists():
            print(f"Updating existing manifest: {output_manifest_abs}")
            rows = read_manifest(output_manifest_abs)
            print(f"Loaded existing manifest with {len(rows)} samples")
        else:
            # Read input manifest
            rows = read_manifest(input_manifest_abs)
            print(f"Loaded manifest with {len(rows)} samples")
        
        # Ensure output directory exists
        output_manifest_abs.parent.mkdir(parents=True, exist_ok=True)
        pov_embeddings_dir = output_manifest_abs.parent / "embeddings" / "povs"
        graph_embeddings_dir = output_manifest_abs.parent / "embeddings" / "graphs"
        
        pov_embeddings_dir.mkdir(parents=True, exist_ok=True)
        graph_embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        layout_emb_mapping = {}
        layout_emb_mapping_ae_name = None  # Store autoencoder name for this embedding
        ae_exp_name = None  # Store autoencoder experiment name
        
        # Step 1: Embed layouts using VAE (optional)
        if ae_checkpoint_path and ae_config_path:
            print(f"\n{'='*60}")
            print("Step 1/3: Embedding layouts with VAE")
            print(f"{'='*60}")
            
            ae_checkpoint_abs = Path(ae_checkpoint_path).resolve()
            ae_config_abs = Path(ae_config_path).resolve()
            
            # Extract autoencoder experiment name from config
            import yaml
            ae_exp_name = "unnamed"  # Will be set below
            if ae_config_abs.exists():
                try:
                    with open(ae_config_abs, 'r') as f:
                        ae_config_data = yaml.safe_load(f)
                        ae_exp_name = ae_config_data.get("experiment", {}).get("name", "unnamed")
                except Exception as e:
                    print(f"[WARNING] Could not read autoencoder name from config: {e}")
                    # Try to extract from checkpoint path as fallback
                    checkpoint_path = Path(ae_checkpoint_abs)
                    ae_exp_name = checkpoint_path.stem.replace("_checkpoint_best", "").replace("_checkpoint_latest", "").replace("_checkpoint", "")
            else:
                # Extract from checkpoint path
                checkpoint_path = Path(ae_checkpoint_abs)
                ae_exp_name = checkpoint_path.stem.replace("_checkpoint_best", "").replace("_checkpoint_latest", "").replace("_checkpoint", "")
            
            print(f"[INFO] Autoencoder experiment name: {ae_exp_name}")
            
            # Create latents directory organized by autoencoder: latents/{exp_name}/
            latents_base_dir = output_manifest_abs.parent / "latents"
            latents_dir = latents_base_dir / ae_exp_name
            latents_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Latents will be saved to: {latents_dir}")
            
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
            
            # Replace layout_path with recolored version if it exists
            layout_rows = []
            for row in rows:
                layout_path_str = row.get("layout_path", "")
                if not layout_path_str:
                    continue
                
                # Check if recolored version exists
                layout_path = Path(layout_path_str)
                if "layouts" in layout_path.parts:
                    # Replace 'layouts' with 'layouts_recolored' in path
                    parts = list(layout_path.parts)
                    try:
                        layouts_idx = parts.index("layouts")
                        parts[layouts_idx] = "layouts_recolored"
                        recolored_path = Path(*parts)
                        if recolored_path.exists():
                            layout_path_str = str(recolored_path)
                    except ValueError:
                        pass  # 'layouts' not in path, use original
                
                layout_rows.append({
                    "layout_path": layout_path_str,
                    "is_empty": 0
                })
            
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
                
                # Store autoencoder name for manifest column naming
                layout_emb_mapping_ae_name = ae_exp_name
                print(f"[INFO] Created {len(layout_emb_mapping)} layout embeddings using autoencoder: {ae_exp_name}")
                
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
        
        # Step 2: Embed POVs using ResNet18 (skip if layout-only mode)
        if not update_existing_manifest:
            step_num = "2/2" if not ae_checkpoint_path else "2/3"
            print(f"\n{'='*60}")
            print(f"Step {step_num}: Embedding POVs with ResNet18")
            print(f"{'='*60}")
        
        pov_rows = [
            row for row in rows
            if row.get("pov_path") and row.get("pov_path") != "0"
        ]
        
        pov_emb_mapping = {}
        if not update_existing_manifest and pov_rows:
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
        
        # Step 3: Embed graphs using SentenceTransformer (skip if layout-only mode)
        if not update_existing_manifest:
            step_num = "2/2" if not ae_checkpoint_path else "3/3"
            print(f"\n{'='*60}")
            print(f"Step {step_num}: Embedding graphs with SentenceTransformer")
            print(f"{'='*60}")
        
        graph_rows = [
            row for row in rows
            if row.get("graph_text_path")
        ]
        
        graph_emb_mapping = {}
        if not update_existing_manifest and graph_rows:
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
        if update_existing_manifest:
            print(f"\n{'='*60}")
            print("Updating manifest with layout embeddings")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print("Step 5/5: Creating final manifest")
            print(f"{'='*60}")
        
        # Determine latent column name for this autoencoder
        # Find next available column: latent_path1, latent_path2, etc.
        latent_column_name = None
        if layout_emb_mapping:
            # Get existing columns from first row
            existing_columns = set(rows[0].keys()) if rows else set()
            
            # Find all existing latent_path columns
            latent_path_columns = [col for col in existing_columns if col.startswith("latent_path")]
            
            if not latent_path_columns:
                # No existing latent columns, use latent_path (for backward compatibility)
                latent_column_name = "latent_path"
            else:
                # Find next available number
                max_num = 0
                for col in latent_path_columns:
                    if col == "latent_path":
                        max_num = max(max_num, 1)  # latent_path counts as 1
                    else:
                        # Extract number from latent_path1, latent_path2, etc.
                        try:
                            num = int(col.replace("latent_path", ""))
                            max_num = max(max_num, num)
                        except ValueError:
                            pass
                
                # Use next number
                latent_column_name = f"latent_path{max_num + 1}"
            
            # Also add autoencoder name column to track which AE created which latents
            ae_name_column = f"latent_ae_{latent_column_name.replace('latent_path', '')}" if latent_column_name != "latent_path" else "latent_ae_1"
            if latent_column_name == "latent_path":
                ae_name_column = "latent_ae_name"  # For backward compatibility with first column
        
        output_rows = []
        for row in rows:
            output_row = row.copy()
            
            # Add/update layout latent path in appropriate column
            layout_path = row.get("layout_path", "")
            if layout_path and layout_emb_mapping and latent_column_name:
                output_row[latent_column_name] = layout_emb_mapping.get(layout_path, "")
                # Add autoencoder name
                if layout_emb_mapping_ae_name and 'ae_name_column' in locals():
                    output_row[ae_name_column] = layout_emb_mapping_ae_name
            else:
                # Preserve existing latent paths if no new embeddings created
                if latent_column_name and latent_column_name in row:
                    output_row[latent_column_name] = row.get(latent_column_name, "")
                elif "latent_path" in row:
                    output_row["latent_path"] = row.get("latent_path", "")
            
            # Add POV embedding path (only if not updating existing)
            if not update_existing_manifest:
                pov_path = row.get("pov_path", "")
                if pov_path and pov_path != "0":
                    output_row["pov_embedding_path"] = pov_emb_mapping.get(pov_path, "")
                elif pov_path == "0":
                    output_row["pov_embedding_path"] = str(zero_emb_path.resolve())
                else:
                    output_row["pov_embedding_path"] = ""
            # else: preserve existing pov_embedding_path from row.copy()
            
            # Add graph embedding path (only if not updating existing)
            if not update_existing_manifest:
                graph_text_path = row.get("graph_text_path", "")
                if graph_text_path:
                    output_row["graph_embedding_path"] = graph_emb_mapping.get(graph_text_path, "")
                else:
                    output_row["graph_embedding_path"] = ""
            # else: preserve existing graph_embedding_path from row.copy()
            
            output_rows.append(output_row)
        
        # Write final manifest
        output_manifest_abs.parent.mkdir(parents=True, exist_ok=True)
        # Get all fieldnames from first row
        fieldnames = list(output_rows[0].keys()) if output_rows else list(rows[0].keys()) if rows else []
        
        # Ensure latent column is included
        if layout_emb_mapping and latent_column_name:
            if latent_column_name not in fieldnames:
                fieldnames.append(latent_column_name)
            # Add autoencoder name column
            if 'ae_name_column' in locals() and ae_name_column not in fieldnames:
                fieldnames.append(ae_name_column)
        elif "latent_path" not in fieldnames:
            # Fallback for backward compatibility
            fieldnames.append("latent_path")
        
        if not update_existing_manifest:
            # Add POV and graph embedding paths if creating new manifest
            if "pov_embedding_path" not in fieldnames:
                fieldnames.append("pov_embedding_path")
            if "graph_embedding_path" not in fieldnames:
                fieldnames.append("graph_embedding_path")
        
        create_manifest(output_rows, output_manifest_abs, fieldnames)
        
        # Clean up temporary files
        if ae_checkpoint_path and ae_config_path:
            for temp_file in [layout_temp_manifest, layout_output_manifest]:
                if 'temp_file' in locals() and temp_file.exists():
                    temp_file.unlink()
        
        print(f"\n{'='*60}")
        print("Dataset embedding completed successfully!")
        print(f"Output manifest: {output_manifest_abs}")
        if layout_emb_mapping:
            if latent_column_name:
                ae_name_display = layout_emb_mapping_ae_name if layout_emb_mapping_ae_name else "unknown"
                print(f"  - {latent_column_name}: Created from VAE ({ae_name_display})")
                if 'ae_name_column' in locals() and ae_name_column:
                    print(f"  - {ae_name_column}: Autoencoder name column")
            else:
                ae_name_display = layout_emb_mapping_ae_name if layout_emb_mapping_ae_name else "unknown"
                print(f"  - latent_path: Created from VAE ({ae_name_display})")
        if not update_existing_manifest:
            print(f"  - graph_embedding_path: Created from SentenceTransformer")
            print(f"  - pov_embedding_path: Created from ResNet18")
        else:
            print(f"  - Updated existing manifest with layout embeddings")
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


def main():
    parser = argparse.ArgumentParser(
        description="Embed ControlNet dataset with VAE, ResNet18, and SentenceTransformer"
    )
    parser.add_argument(
        "--ae-checkpoint",
        type=Path,
        required=False,
        default=None,
        help="Path to VAE checkpoint (optional - if not provided, only POV and graph embeddings will be created)"
    )
    parser.add_argument(
        "--ae-config",
        type=Path,
        required=False,
        default=None,
        help="Path to VAE config YAML (optional - only needed if --ae-checkpoint is provided)"
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        required=True,
        help="Path to input ControlNet manifest (with layout_path, pov_path, graph_text_path)"
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        required=True,
        help="Path to output manifest (with latent_path, pov_embedding_path, graph_embedding_path)"
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
    parser.add_argument(
        "--layout-only",
        action="store_true",
        help="Only embed layouts (requires VAE). Updates existing manifest with latent_path. Use this after VAE is trained."
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("ControlNet Dataset Embedding")
    print("="*60)
    if args.layout_only:
        if not args.ae_checkpoint:
            print("ERROR: --layout-only requires --ae-checkpoint")
            sys.exit(1)
        print(f"VAE checkpoint: {args.ae_checkpoint}")
        print(f"VAE config: {args.ae_config}")
        print("Mode: Embedding layouts only (updating existing manifest)")
    elif args.ae_checkpoint:
        print(f"VAE checkpoint: {args.ae_checkpoint}")
        print(f"VAE config: {args.ae_config}")
        print("Mode: Creating layouts + POVs + graphs embeddings")
    else:
        print("Mode: Creating POVs + graphs embeddings only (no VAE)")
    print(f"Input manifest: {args.input_manifest}")
    print(f"Output manifest: {args.output_manifest}")
    print("="*60)
    print()
    
    # Validate arguments
    if args.ae_checkpoint and not args.ae_config:
        print("ERROR: --ae-config is required when --ae-checkpoint is provided")
        sys.exit(1)
    if args.ae_config and not args.ae_checkpoint:
        print("ERROR: --ae-checkpoint is required when --ae-config is provided")
        sys.exit(1)
    
    success = embed_controlnet_dataset_with_vae(
        args.ae_checkpoint,
        args.ae_config,
        args.input_manifest,
        args.output_manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        create_layout_embeddings=True,
        update_existing_manifest=args.layout_only
    )
    
    if not success:
        print("ERROR: Embedding failed")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Embedding COMPLETE - SUCCESS")
    print("="*60)
    print(f"Output manifest: {args.output_manifest}")
    print("="*60)


if __name__ == "__main__":
    main()

