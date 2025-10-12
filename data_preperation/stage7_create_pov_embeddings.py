#!/usr/bin/env python3
"""
stage8_create_pov_embeddings.py

Reads povs.csv manifest, generates ResNet embeddings for POV images,
and creates povs_with_embeddings.csv manifest.
"""

import argparse
import csv
import torch
from pathlib import Path
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

from itertools import islice

def batched(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch

def load_resnet_model(device="cuda"):
    """Load ResNet18 with removed classifier head for feature extraction."""
    resnet = models.resnet18(weights="IMAGENET1K_V1").to(device)
    resnet.fc = torch.nn.Identity()  # remove classifier head
    resnet.eval()
    return resnet


def get_transform():
    """Get image preprocessing transform for ResNet."""
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def extract_embedding(image_path, model, transform, device):
    """Extract ResNet embedding from a single image."""
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        embedding = model(x).squeeze(0).cpu()  # [512]
    return embedding



def process_povs(manifest_path: str, output_manifest: str,
                 save_format: str = "pt", batch_size: int = 1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading ResNet18 model...")
    model = load_resnet_model(device)
    transform = get_transform()

    manifest_path = Path(manifest_path)
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
            pov_path = row['pov_path']
            if int(row['is_empty']) or not Path(pov_path).exists():
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
            pov_path_obj = Path(r['pov_path'])
            embedding_path = pov_path_obj.with_suffix('.pt')
            torch.save(e, embedding_path)
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

def main():
    parser = argparse.ArgumentParser(
        description="Generate ResNet embeddings for POV images and create updated manifest"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to povs.csv manifest"
    )
    parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Number of images processed per GPU batch"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for output povs_with_embeddings.csv"
    )
    parser.add_argument(
        "--format",
        choices=["pt", "npy"],
        default="pt",
        help="Embedding save format: pt (PyTorch) or npy (NumPy)"
    )
    
    args = parser.parse_args()
    
    process_povs(
        manifest_path=args.manifest,
        output_manifest=args.output,
        save_format=args.format,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()