#!/usr/bin/env python3
"""
Embed for Training - Create POV image and graph text embeddings.

Takes the manifest CSV from collect_manifest.py and creates:
1. POV image embeddings (ResNet18) - one per room
2. Graph text embeddings (sentence-transformers) - one per room

Updates the manifest in-place or creates a new one.

Usage:
    # Create both POV and graph embeddings
    python embed_for_training.py \
        --manifest dataset_v2/manifests/manifest_tex.csv \
        --dataset-root dataset_v2

    # Only POV embeddings
    python embed_for_training.py \
        --manifest dataset_v2/manifests/manifest_tex.csv \
        --dataset-root dataset_v2 \
        --skip-graph

    # Only graph embeddings
    python embed_for_training.py \
        --manifest dataset_v2/manifests/manifest_tex.csv \
        --dataset-root dataset_v2 \
        --skip-pov

Output Structure:
    dataset_v2/
    ├── povs/
    │   └── embeddings_{variant}/
    │       └── {scene_id}_{room_id}_pov.pt
    └── graphs/
        └── embeddings/
            └── {scene_id}_{room_id}_text.pt
"""

import argparse
import logging
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision import transforms, models
from typing import Optional, Dict, Tuple
import warnings

warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# POV Image Encoder (ResNet18)
# =============================================================================

class POVEncoder(nn.Module):
    """ResNet18-based POV image encoder (512-dim output)."""
    
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove final FC layer, keep avgpool output
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.output_dim = 512
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x.flatten(1)  # [B, 512]


def get_pov_transform():
    """ImageNet-style transform for POV images."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# =============================================================================
# Text Encoder (Sentence Transformers)
# =============================================================================

class TextEncoder:
    """Sentence-transformers text encoder (384-dim for all-MiniLM-L6-v2)."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cuda"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.output_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded text encoder: {model_name} (dim={self.output_dim})")
    
    def encode_batch(self, texts: list) -> torch.Tensor:
        embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        return embeddings


# =============================================================================
# Embedding Functions
# =============================================================================

def embed_povs(
    df: pd.DataFrame,
    dataset_root: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int = 64,
    skip_existing: bool = True,
) -> Dict[Tuple[str, str], str]:
    """
    Embed POV images. One embedding per (scene_id, room_id).
    
    Returns: dict of (scene_id, room_id) -> relative_embedding_path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique rooms with POVs
    has_pov = df["pov_path"].notna() & (df["pov_path"] != "")
    rooms_df = df[has_pov][["scene_id", "room_id", "pov_path"]].drop_duplicates(subset=["scene_id", "room_id"])
    
    # Filter already processed if skip_existing
    if skip_existing:
        to_process = []
        for _, row in rooms_df.iterrows():
            emb_path = output_dir / f"{row['scene_id']}_{row['room_id']}_pov.pt"
            if not emb_path.exists():
                to_process.append(row)
        rooms_df = pd.DataFrame(to_process)
        logger.info(f"POV: {len(to_process)} to process, {len(rooms_df)} skipped (existing)")
    
    if len(rooms_df) == 0:
        logger.info("POV: All embeddings exist, skipping")
        # Return map for existing files
        embedding_map = {}
        for _, row in df[has_pov][["scene_id", "room_id"]].drop_duplicates().iterrows():
            rel_path = f"povs/embeddings_{output_dir.name.split('_')[-1]}/{row['scene_id']}_{row['room_id']}_pov.pt"
            embedding_map[(row["scene_id"], row["room_id"])] = rel_path
        return embedding_map
    
    logger.info(f"Embedding {len(rooms_df)} POV images...")
    
    # Load model
    encoder = POVEncoder().to(device).eval()
    transform = get_pov_transform()
    
    embedding_map = {}
    
    with torch.no_grad():
        for start in tqdm(range(0, len(rooms_df), batch_size), desc="POV embeddings"):
            batch = rooms_df.iloc[start:start + batch_size]
            
            images = []
            valid_rows = []
            
            for _, row in batch.iterrows():
                pov_path = dataset_root / row["pov_path"]
                if not pov_path.exists():
                    logger.warning(f"POV not found: {pov_path}")
                    continue
                
                try:
                    img = Image.open(pov_path).convert("RGB")
                    images.append(transform(img))
                    valid_rows.append(row)
                except Exception as e:
                    logger.warning(f"Failed to load {pov_path}: {e}")
            
            if not images:
                continue
            
            # Encode
            batch_tensor = torch.stack(images).to(device)
            embeddings = encoder(batch_tensor)
            
            # Save
            for emb, row in zip(embeddings, valid_rows):
                scene_id, room_id = row["scene_id"], row["room_id"]
                emb_name = f"{scene_id}_{room_id}_pov.pt"
                emb_path = output_dir / emb_name
                torch.save(emb.cpu(), emb_path)
                
                rel_path = str(emb_path.relative_to(dataset_root))
                embedding_map[(scene_id, room_id)] = rel_path
    
    # Add existing embeddings to map
    has_pov_unique = df[has_pov][["scene_id", "room_id"]].drop_duplicates()
    for _, row in has_pov_unique.iterrows():
        key = (row["scene_id"], row["room_id"])
        if key not in embedding_map:
            emb_path = output_dir / f"{row['scene_id']}_{row['room_id']}_pov.pt"
            if emb_path.exists():
                embedding_map[key] = str(emb_path.relative_to(dataset_root))
    
    logger.info(f"POV: Saved {len(embedding_map)} embeddings")
    return embedding_map


def embed_graph_texts(
    df: pd.DataFrame,
    dataset_root: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int = 128,
    skip_existing: bool = True,
) -> Dict[Tuple[str, str], str]:
    """
    Embed graph text descriptions. One embedding per (scene_id, room_id).
    
    Returns: dict of (scene_id, room_id) -> relative_embedding_path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique rooms with graph text
    has_text = df["graph_text_path"].notna() & (df["graph_text_path"] != "")
    rooms_df = df[has_text][["scene_id", "room_id", "graph_text_path"]].drop_duplicates(subset=["scene_id", "room_id"])
    
    # Filter already processed if skip_existing
    if skip_existing:
        to_process = []
        for _, row in rooms_df.iterrows():
            emb_path = output_dir / f"{row['scene_id']}_{row['room_id']}_text.pt"
            if not emb_path.exists():
                to_process.append(row)
        original_count = len(rooms_df)
        rooms_df = pd.DataFrame(to_process)
        logger.info(f"Graph text: {len(rooms_df)} to process, {original_count - len(rooms_df)} skipped (existing)")
    
    if len(rooms_df) == 0:
        logger.info("Graph text: All embeddings exist, skipping")
        embedding_map = {}
        for _, row in df[has_text][["scene_id", "room_id"]].drop_duplicates().iterrows():
            rel_path = f"graphs/embeddings/{row['scene_id']}_{row['room_id']}_text.pt"
            embedding_map[(row["scene_id"], row["room_id"])] = rel_path
        return embedding_map
    
    logger.info(f"Embedding {len(rooms_df)} graph texts...")
    
    # Load model
    encoder = TextEncoder(device=str(device))
    
    embedding_map = {}
    
    # Load all texts first
    texts = []
    valid_rows = []
    
    for _, row in rooms_df.iterrows():
        text_path = dataset_root / row["graph_text_path"]
        if not text_path.exists():
            logger.warning(f"Graph text not found: {text_path}")
            continue
        
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                texts.append(text)
                valid_rows.append(row)
        except Exception as e:
            logger.warning(f"Failed to load {text_path}: {e}")
    
    # Encode in batches
    for start in tqdm(range(0, len(texts), batch_size), desc="Graph text embeddings"):
        batch_texts = texts[start:start + batch_size]
        batch_rows = valid_rows[start:start + batch_size]
        
        embeddings = encoder.encode_batch(batch_texts)
        
        for emb, row in zip(embeddings, batch_rows):
            scene_id, room_id = row["scene_id"], row["room_id"]
            emb_name = f"{scene_id}_{room_id}_text.pt"
            emb_path = output_dir / emb_name
            torch.save(emb.cpu(), emb_path)
            
            rel_path = str(emb_path.relative_to(dataset_root))
            embedding_map[(scene_id, room_id)] = rel_path
    
    # Add existing embeddings to map
    has_text_unique = df[has_text][["scene_id", "room_id"]].drop_duplicates()
    for _, row in has_text_unique.iterrows():
        key = (row["scene_id"], row["room_id"])
        if key not in embedding_map:
            emb_path = output_dir / f"{row['scene_id']}_{row['room_id']}_text.pt"
            if emb_path.exists():
                embedding_map[key] = str(emb_path.relative_to(dataset_root))
    
    logger.info(f"Graph text: Saved {len(embedding_map)} embeddings")
    return embedding_map


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Create POV and graph text embeddings for training")
    parser.add_argument("--manifest", required=True, type=Path, help="Input manifest CSV")
    parser.add_argument("--dataset-root", required=True, type=Path, help="Dataset root directory")
    parser.add_argument("--output-manifest", type=Path, default=None, help="Output manifest path (default: overwrite input)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--skip-pov", action="store_true", help="Skip POV embedding")
    parser.add_argument("--skip-graph", action="store_true", help="Skip graph text embedding")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing embeddings")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load manifest
    df = pd.read_csv(args.manifest)
    logger.info(f"Loaded manifest: {len(df)} rows")
    
    # Verify sample_weight column exists (warn if missing)
    if "sample_weight" not in df.columns:
        logger.warning("'sample_weight' column not found in manifest. Weights will not be preserved.")
    else:
        logger.info(f"Found 'sample_weight' column - weights will be preserved")
    
    # Determine variant from manifest name (manifest_tex.csv -> tex)
    variant = "tex"
    if "seg" in args.manifest.stem:
        variant = "seg"
    logger.info(f"Detected variant: {variant}")
    
    # POV embeddings
    if not args.skip_pov:
        pov_output_dir = args.dataset_root / "povs" / f"embeddings_{variant}"
        pov_map = embed_povs(
            df=df,
            dataset_root=args.dataset_root,
            output_dir=pov_output_dir,
            device=device,
            batch_size=args.batch_size,
            skip_existing=not args.overwrite,
        )
        
        # Update manifest
        df["pov_embedding_path"] = df.apply(
            lambda row: pov_map.get((row["scene_id"], row["room_id"]), ""),
            axis=1
        )
    
    # Graph text embeddings
    if not args.skip_graph:
        graph_output_dir = args.dataset_root / "graphs" / "embeddings"
        graph_map = embed_graph_texts(
            df=df,
            dataset_root=args.dataset_root,
            output_dir=graph_output_dir,
            device=device,
            batch_size=args.batch_size * 2,  # Text is faster
            skip_existing=not args.overwrite,
        )
        
        # Update manifest
        df["graph_embedding_path"] = df.apply(
            lambda row: graph_map.get((row["scene_id"], row["room_id"]), ""),
            axis=1
        )
    
    # Save manifest (preserve all columns including sample_weight)
    output_path = args.output_manifest or args.manifest
    
    # Verify sample_weight is preserved
    if "sample_weight" in df.columns:
        logger.info(f"Preserving 'sample_weight' column in output manifest")
    else:
        logger.warning("'sample_weight' column missing - weights will not be available for training")
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved manifest: {output_path} ({len(df.columns)} columns)")
    
    # Print summary
    pov_count = (df["pov_embedding_path"] != "").sum() if "pov_embedding_path" in df.columns else 0
    graph_count = (df["graph_embedding_path"] != "").sum() if "graph_embedding_path" in df.columns else 0
    logger.info(f"Summary: {pov_count} POV embeddings, {graph_count} graph embeddings")


if __name__ == "__main__":
    main()