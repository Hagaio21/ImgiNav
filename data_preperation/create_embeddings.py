#!/usr/bin/env python3

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
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import yaml
from sentence_transformers import SentenceTransformer

# Add parent directory to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from models.autoencoder import AutoEncoder
except ImportError:
    AutoEncoder = None  # Will fail gracefully if not available

from utils.text_utils import articleize, graph2text
from utils.semantic_utils import Taxonomy


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


# =============================================================================
# Model Loading
# =============================================================================

def load_autoencoder_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
    if AutoEncoder is None:
        raise ImportError("AutoEncoder module not available")
    
    print(f"[INFO] Loading config: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if "model" not in config:
        raise KeyError(f"'model' key missing in config file {config_path}")

    model_cfg = config["model"]
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

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    
    # Handle potential DataParallel prefix
    if state_dict and list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
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
# Layout Embeddings
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


def create_layout_embeddings(model, data_root: Path, device: str = "cuda", 
                            batch_size: int = 32, overwrite: bool = False) -> List[Dict[str, Any]]:
    layout_paths = sorted(list(data_root.rglob("*layout.png")))
    
    if not layout_paths:
        raise RuntimeError(f"No layout images found under {data_root}")
    
    # Prepare transform
    image_size = model.encoder.image_size
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
    with torch.no_grad():
        for i in tqdm(range(0, total_to_encode, batch_size), desc="Encoding layouts", unit="batch"):
            batch_job_paths = paths_to_encode[i:i + batch_size]
            imgs, valid_out_paths = [], []
            
            for in_path, out_path in batch_job_paths:
                try:
                    img = Image.open(in_path).convert("RGB")
                    imgs.append(transform(img))
                    valid_out_paths.append(out_path)
                except Exception as e:
                    print(f"[WARN] Failed to load {in_path}: {e}")
                    failed_load += 1
            
            if not imgs:
                continue
            
            imgs_tensor = torch.stack(imgs).to(device)
            z = model.encode_latent(imgs_tensor, deterministic=True)
            
            for j, out_path in enumerate(valid_out_paths):
                emb = z[j].cpu()
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
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Found {len(rows)} POV images to process")
    
    output_rows = []
    skipped = 0
    processed = 0
    
    for batch_rows in tqdm(batched(rows, batch_size),
                           total=len(rows)//batch_size + 1,
                           desc="Processing POV images"):
        batch_imgs = []
        valid_rows = []
        
        for row in batch_rows:
            pov_path = row.get('pov_path') or row.get('pov_image', '')
            if not pov_path:
                out = row.copy()
                out['embedding_path'] = ''
                output_rows.append(out)
                skipped += 1
                continue
            
            # Check if empty flag exists
            is_empty = int(row.get('is_empty', 0))
            if is_empty or not Path(pov_path).exists():
                out = row.copy()
                out['embedding_path'] = ''
                output_rows.append(out)
                skipped += 1
                continue
                
            try:
                img = Image.open(pov_path).convert("RGB")
                x = transform(img)
                batch_imgs.append(x)
                valid_rows.append(row)
            except Exception as e:
                print(f"Error reading {pov_path}: {e}")
                out = row.copy()
                out['embedding_path'] = ''
                output_rows.append(out)
                skipped += 1
        
        if not valid_rows:
            continue
        
        x = torch.stack(batch_imgs).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            emb = model(x).cpu()  # [B,512]
        
        for r, e in zip(valid_rows, emb):
            pov_path_obj = Path(r.get('pov_path') or r.get('pov_image', ''))
            embedding_path = pov_path_obj.with_suffix('.pt')
            
            if save_format == "pt":
                torch.save(e, embedding_path)
            else:  # npy
                np.save(embedding_path.with_suffix('.npy'), e.numpy())
            
            out = r.copy()
            out['embedding_path'] = str(embedding_path)
            output_rows.append(out)
            processed += 1
    
    # Write output manifest
    output_path = Path(output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = list(rows[0].keys()) + ['embedding_path']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    
    print(f"\n✓ Processed {processed}/{len(rows)} POV images successfully")
    print(f"✓ Skipped {skipped} images (empty or errors)")
    print(f"✓ Output manifest: {output_path}")


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
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
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
    output_path = Path(output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = list(rows[0].keys()) + ['embedding_path']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    
    print(f"\n✓ Processed {len(rows) - skipped}/{len(rows)} graphs successfully")
    print(f"✓ Skipped {skipped} graphs")
    print(f"✓ Output manifest: {output_path}")


# =============================================================================
# Graph Text Generation (from create_graph_text.py)
# =============================================================================

def create_graph_text_files(manifest_path: Path, taxonomy_path: Path):
    print(f"Loading taxonomy from {taxonomy_path}")
    taxonomy = Taxonomy(taxonomy_path)
    
    print(f"Reading manifest: {manifest_path}")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
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
        help="Path to input manifest CSV (required for pov and graph types)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for output manifest or directory"
    )
    parser.add_argument(
        "--batch_size",
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
        "--config",
        help="Path to AutoEncoder config YAML (required for layout type)"
    )
    parser.add_argument(
        "--ckpt",
        help="Path to AutoEncoder checkpoint (required for layout type)"
    )
    parser.add_argument(
        "--data_root",
        help="Root folder containing scenes/rooms (required for layout type)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--manifest_out",
        default=None,
        help="Output manifest filename (for layout type, saved in data_root)"
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
        if not args.config or not args.ckpt or not args.data_root:
            parser.error("--config, --ckpt, and --data_root are required for layout type")
        
        print(f"[INFO] Using device: {device}")
        model = load_autoencoder_model(args.config, args.ckpt, device=device)
        manifest_data = create_layout_embeddings(
            model,
            Path(args.data_root),
            device=device,
            batch_size=args.batch_size,
            overwrite=args.overwrite
        )
        
        if manifest_data and args.manifest_out:
            import pandas as pd
            df = pd.DataFrame(manifest_data)
            cols = ["scene", "type", "room_id", "layout_path", "layout_emb_path"]
            df = df[cols]
            output_csv_path = Path(args.data_root) / args.manifest_out
            df.to_csv(output_csv_path, index=False, sep="|")
            print(f"\n[INFO] Manifest saved to {output_csv_path}")
    
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

