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


def load_resnet_model(device="cuda"):
    """Load ResNet18 with removed classifier head for feature extraction."""
    resnet = models.resnet18(weights="IMAGENET1K_V1").to(device)
    resnet.fc = torch.nn.Identity()  # remove classifier head
    resnet.eval()
    return resnet


def get_transform():
    """Get image preprocessing transform for ResNet."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def extract_embedding(image_path, model, transform, device):
    """
    Extract ResNet embedding from a single image.
    
    Returns:
        torch.Tensor: 512-dimensional embedding vector
    """
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Forward through ResNet layers
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        embedding = x.view(x.size(0), -1)  # flatten to [1, 512]
    
    return embedding.squeeze(0).cpu()  # return [512]


def process_povs(manifest_path: str, output_manifest: str, 
                 save_format: str = "pt", batch_size: int = 1):
    """
    Process all POV images in manifest: extract embeddings and save.
    
    Args:
        manifest_path: Path to povs.csv
        output_manifest: Path for povs_with_embeddings.csv
        save_format: 'pt' for PyTorch or 'npy' for NumPy
        batch_size: Currently only supports 1 (for future optimization)
    """
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and transform
    print("Loading ResNet18 model...")
    model = load_resnet_model(device)
    transform = get_transform()
    
    # Read manifest
    print(f"Reading manifest: {manifest_path}")
    manifest_path = Path(manifest_path)
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Found {len(rows)} POV images to process")
    
    # Process each POV
    output_rows = []
    skipped = 0
    processed = 0
    
    for row in tqdm(rows, desc="Processing POV images"):
        pov_path = row['pov_path']
        is_empty = int(row['is_empty'])
        
        # Skip empty POVs
        if is_empty:
            output_row = row.copy()
            output_row['embedding_path'] = ''
            output_rows.append(output_row)
            skipped += 1
            continue
        
        try:
            # Check if file exists
            pov_path_obj = Path(pov_path)
            if not pov_path_obj.exists():
                print(f"Warning: File not found: {pov_path}")
                output_row = row.copy()
                output_row['embedding_path'] = ''
                output_rows.append(output_row)
                skipped += 1
                continue
            
            # Extract embedding
            embedding = extract_embedding(pov_path, model, transform, device)
            
            # Determine save path
            if save_format == "pt":
                embedding_path = pov_path_obj.with_suffix('.pt')
                torch.save(embedding, embedding_path)
            else:  # npy
                import numpy as np
                embedding_path = pov_path_obj.with_suffix('.npy')
                np.save(embedding_path, embedding.numpy())
            
            # Add to output manifest
            output_row = row.copy()
            output_row['embedding_path'] = str(embedding_path)
            output_rows.append(output_row)
            processed += 1
            
        except Exception as e:
            print(f"Error processing {pov_path}: {e}")
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
        save_format=args.format
    )


if __name__ == "__main__":
    main()