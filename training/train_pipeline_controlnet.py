#!/usr/bin/env python3
"""
Training pipeline script for ControlNet experiments.
Embeds the dataset before training starts, similar to diffusion pipeline.

This script:
1. Embeds layouts using autoencoder (saves to experiment_dir/latents/)
2. Embeds POVs using ResNet (saves to experiment_dir/embeddings/povs/)
3. Embeds graphs using SentenceTransformer (saves to experiment_dir/embeddings/graphs/)
4. Creates manifest with all embedding paths
5. Updates ControlNet config to use embedded manifest
6. Trains ControlNet model

Usage:
    python training/train_pipeline_controlnet.py --config <controlnet_config>
"""

import torch
import sys
import os
import yaml
import argparse
import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.utils import (
    set_deterministic,
    load_config,
    get_device,
)
from training.train_controlnet import main as train_controlnet_main
from data_preparation.create_embeddings import (
    create_layout_embeddings_from_manifest,
    load_resnet_model,
    load_sentence_transformer_model,
)
from data_preparation.utils.text_utils import graph2text
from models.autoencoder import Autoencoder
from common.file_io import read_manifest, create_manifest
from common.taxonomy import Taxonomy


def update_controlnet_config_manifest(controlnet_config_path, manifest_path):
    """
    Update ControlNet config to use the embedded manifest.
    
    Args:
        controlnet_config_path: Path to ControlNet config YAML
        manifest_path: Path to manifest with embedded latents and embeddings
    """
    print(f"\n{'='*60}")
    print("Updating ControlNet config with embedded manifest")
    print(f"{'='*60}")
    
    # Verify manifest exists
    manifest_abs = Path(manifest_path).resolve()
    if not manifest_abs.exists():
        raise FileNotFoundError(
            f"Embedded manifest not found: {manifest_abs}\n"
            f"Embedding step may have failed."
        )
    
    # Load ControlNet config
    with open(controlnet_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update dataset manifest and outputs
    if 'dataset' not in config:
        config['dataset'] = {}
    
    old_manifest = config['dataset'].get('manifest', 'not set')
    config['dataset']['manifest'] = str(manifest_abs)
    
    # Update outputs to use pre-embedded paths
    config['dataset']['outputs'] = {
        'latent': 'latent_path',           # Layout latents
        'text_emb': 'graph_embedding_path',  # Graph embeddings
        'pov_emb': 'pov_embedding_path'    # POV embeddings
    }
    
    # Save updated config
    with open(controlnet_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"Updated manifest path:")
    print(f"  Old: {old_manifest}")
    print(f"  New: {manifest_abs}")
    print(f"Config saved to: {controlnet_config_path}")
    print(f"{'='*60}\n")


def embed_controlnet_dataset(
    controlnet_config_path,
    input_manifest_path,
    output_manifest_path,
    batch_size=32,
    num_workers=8
):
    """
    Embed ControlNet dataset: layouts, POVs, and graphs.
    
    Args:
        controlnet_config_path: Path to ControlNet config YAML
        input_manifest_path: Path to input manifest (from create_controlnet_manifests.py)
        output_manifest_path: Path to output manifest (with all embedding paths)
        batch_size: Batch size for encoding
        num_workers: Number of workers for data loading
        
    Returns:
        True if embedding succeeded, False otherwise
    """
    print(f"\n{'='*60}")
    print("PHASE 1: Embedding ControlNet Dataset")
    print(f"{'='*60}")
    print(f"Input manifest: {input_manifest_path}")
    print(f"Output manifest: {output_manifest_path}")
    print(f"{'='*60}\n")
    
    try:
        # Load ControlNet config
        config = load_config(controlnet_config_path)
        
        # Get experiment directory
        exp_config = config.get("experiment", {})
        exp_name = exp_config.get("name", "controlnet")
        save_path = Path(exp_config.get("save_path", "experiments/controlnet"))
        experiment_dir = save_path / exp_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create embedding directories
        latents_dir = experiment_dir / "latents"
        pov_embeddings_dir = experiment_dir / "embeddings" / "povs"
        graph_embeddings_dir = experiment_dir / "embeddings" / "graphs"
        
        latents_dir.mkdir(parents=True, exist_ok=True)
        pov_embeddings_dir.mkdir(parents=True, exist_ok=True)
        graph_embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Experiment directory: {experiment_dir}")
        print(f"  Latents: {latents_dir}")
        print(f"  POV embeddings: {pov_embeddings_dir}")
        print(f"  Graph embeddings: {graph_embeddings_dir}")
        print()
        
        # Get autoencoder checkpoint from config
        # Load autoencoder separately (diffusion model doesn't store full autoencoder)
        autoencoder_config = config.get("autoencoder", {})
        autoencoder_checkpoint = autoencoder_config.get("checkpoint")
        
        if not autoencoder_checkpoint:
            raise ValueError("ControlNet config must specify 'autoencoder.checkpoint' for layout embeddings")
        
        # Load autoencoder directly
        print(f"Loading autoencoder from: {autoencoder_checkpoint}")
        autoencoder = Autoencoder.load_checkpoint(
            autoencoder_checkpoint,
            map_location="cpu"  # Load to CPU first, will move to GPU when needed
        )
        autoencoder.eval()
        
        # Get autoencoder config path (try to find it from checkpoint path)
        checkpoint_path = Path(autoencoder_checkpoint)
        possible_configs = [
            checkpoint_path.parent / f"{checkpoint_path.stem.replace('_checkpoint_best', '').replace('_checkpoint_latest', '')}.yaml",
            checkpoint_path.parent.parent / "experiment_config.yaml",
        ]
        autoencoder_config_path = None
        for pc in possible_configs:
            if pc.exists():
                autoencoder_config_path = str(pc)
                break
        
        if not autoencoder_config_path:
            print("[WARNING] Could not find autoencoder config, using defaults for transform")
        
        # Read input manifest
        input_manifest_abs = Path(input_manifest_path).resolve()
        if not input_manifest_abs.exists():
            raise FileNotFoundError(f"Input manifest not found: {input_manifest_abs}")
        
        rows = read_manifest(input_manifest_abs)
        print(f"Loaded manifest with {len(rows)} samples")
        
        # Step 1: Embed layouts
        print(f"\n{'='*60}")
        print("Step 1/3: Embedding layouts")
        print(f"{'='*60}")
        
        # Create temporary manifest for layouts
        layout_temp_manifest = experiment_dir / "temp_layout_manifest.csv"
        layout_output_manifest = experiment_dir / "temp_layout_output_manifest.csv"
        
        # Filter rows that have layout_path
        layout_rows = [
            {"layout_path": row.get("layout_path", ""), "is_empty": 0}
            for row in rows
            if row.get("layout_path")
        ]
        
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
        else:
            layout_emb_mapping = {}
        
        # Step 2: Embed POVs
        print(f"\n{'='*60}")
        print("Step 2/3: Embedding POVs")
        print(f"{'='*60}")
        
        # Filter rows that have pov_path (and it's not "0" for scenes)
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
            
            from PIL import Image
            import torchvision.transforms as T
            
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
            
            # Process in batches
            from itertools import islice
            def batched(iterable, n):
                it = iter(iterable)
                while True:
                    batch = list(islice(it, n))
                    if not batch:
                        break
                    yield batch
            
            processed = 0
            for batch_rows in tqdm(batched(pov_rows, batch_size), desc="Embedding POVs"):
                # Process batch
                batch_imgs = []
                valid_rows = []
                
                for row in batch_rows:
                    img, row_data = process_pov_item(row)
                    if img is not None:
                        batch_imgs.append(transform(img))
                        valid_rows.append(row_data)
                
                if not valid_rows:
                    continue
                
                # Encode batch
                imgs_tensor = torch.stack(batch_imgs).to(device)
                with torch.no_grad():
                    embeddings = encode_povs(imgs_tensor)
                
                # Save embeddings
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
        
        # Step 3: Embed graphs
        print(f"\n{'='*60}")
        print("Step 3/3: Embedding graphs")
        print(f"{'='*60}")
        
        # Get taxonomy path (not needed for text files, but kept for compatibility)
        taxonomy_path = config.get("dataset", {}).get("taxonomy", "config/taxonomy.json")
        if not Path(taxonomy_path).is_absolute():
            taxonomy_path = Path(__file__).parent.parent / taxonomy_path
        
        # Filter rows that have graph_text_path
        graph_rows = [
            row for row in rows
            if row.get("graph_text_path")
        ]
        
        graph_emb_mapping = {}
        
        if graph_rows:
            from sentence_transformers import SentenceTransformer
            
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
                    # Read text file directly (no need to convert from JSON)
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
                    
                    # Generate embedding
                    embedding = embedder.encode(text, normalize_embeddings=True)
                    
                    # Save embedding
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
        
        # Step 4: Create zero embeddings for scenes (pov_path == "0")
        print(f"\n{'='*60}")
        print("Step 4/4: Creating zero embeddings for scenes")
        print(f"{'='*60}")
        
        # Get POV embedding dimension from a sample (or use default)
        pov_emb_dim = 512  # ResNet18 output dimension
        if pov_emb_mapping:
            # Load a sample to get dimension
            sample_emb_path = list(pov_emb_mapping.values())[0]
            sample_emb = torch.load(sample_emb_path)
            pov_emb_dim = sample_emb.shape[0] if sample_emb.dim() == 1 else sample_emb.numel()
        
        # Create zero embedding file for scenes
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
                # Scene sample: use zero embedding
                output_row["pov_embedding_path"] = str(zero_emb_path.resolve())
            else:
                output_row["pov_embedding_path"] = ""
            
            # Add graph embedding path (use graph_text_path for embedding)
            graph_text_path = row.get("graph_text_path", "")
            if graph_text_path:
                output_row["graph_embedding_path"] = graph_emb_mapping.get(graph_text_path, "")
            else:
                output_row["graph_embedding_path"] = ""
            
            output_rows.append(output_row)
        
        # Write final manifest
        output_manifest_abs = Path(output_manifest_path).resolve()
        output_manifest_abs.parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = list(rows[0].keys()) + ["latent_path", "pov_embedding_path", "graph_embedding_path"]
        create_manifest(output_rows, output_manifest_abs, fieldnames)
        
        # Clean up temporary files
        for temp_file in [
            layout_temp_manifest, layout_output_manifest
        ]:
            if temp_file.exists():
                temp_file.unlink()
        
        print(f"\n{'='*60}")
        print("Dataset embedding completed successfully!")
        print(f"{'='*60}\n")
        return True
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Dataset embedding failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        return False


def train_controlnet(controlnet_config_path):
    """
    Train ControlNet model using the training script.
    
    Args:
        controlnet_config_path: Path to ControlNet config YAML
        
    Returns:
        True if training succeeded, False otherwise
    """
    print(f"\n{'='*60}")
    print("PHASE 2: Training ControlNet Model")
    print(f"{'='*60}")
    print(f"Config: {controlnet_config_path}")
    print(f"{'='*60}\n")
    
    try:
        # Call the training script directly
        train_controlnet_main(argparse.Namespace(
            config=str(controlnet_config_path),
            device=None,
            no_resume=False
        ))
        return True
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: ControlNet training failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="ControlNet training pipeline with dataset embedding"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to ControlNet config YAML"
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding step (use existing embeddings)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training step (only embed dataset)"
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
    
    # Load config
    config = load_config(args.config)
    
    # Get paths
    exp_config = config.get("experiment", {})
    exp_name = exp_config.get("name", "controlnet")
    save_path = Path(exp_config.get("save_path", "experiments/controlnet"))
    experiment_dir = save_path / exp_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Get input manifest from config
    dataset_config = config.get("dataset", {})
    input_manifest = dataset_config.get("manifest")
    if not input_manifest:
        raise ValueError("ControlNet config must specify 'dataset.manifest'")
    
    # Output manifest will be in experiment directory
    output_manifest = experiment_dir / "manifest_with_embeddings.csv"
    
    print(f"\n{'='*60}")
    print("ControlNet Training Pipeline")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Input manifest: {input_manifest}")
    print(f"Output manifest: {output_manifest}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"{'='*60}\n")
    
    # Step 1: Embed dataset (unless skipped)
    if not args.skip_embedding:
        if output_manifest.exists():
            print(f"\n{'='*60}")
            print("Embedded manifest already exists in experiment directory")
            print(f"{'='*60}")
            print(f"Found: {output_manifest}")
            print("")
            print("WARNING: Using existing embeddings. If models were retrained,")
            print("you MUST re-embed the dataset or results will be wrong!")
            print("")
            print("To force re-embedding, delete the manifest:")
            print(f"  rm {output_manifest}")
            print("")
            print("Skipping embedding step.")
            print(f"{'='*60}\n")
        else:
            success = embed_controlnet_dataset(
                args.config,
                input_manifest,
                str(output_manifest),
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            if not success:
                print("ERROR: Embedding failed. Cannot proceed with training.")
                sys.exit(1)
    else:
        print("Skipping embedding step (--skip-embedding flag set)")
    
    # Step 2: Update config with embedded manifest
    if output_manifest.exists():
        update_controlnet_config_manifest(args.config, str(output_manifest))
    else:
        print("WARNING: Embedded manifest not found. Training will use original manifest.")
        print("This may cause errors if embeddings are required.")
    
    # Step 3: Train ControlNet (unless skipped)
    if not args.skip_training:
        success = train_controlnet(args.config)
        if not success:
            print("ERROR: Training failed.")
            sys.exit(1)
    else:
        print("Skipping training step (--skip-training flag set)")
    
    print(f"\n{'='*60}")
    print("ControlNet pipeline completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

