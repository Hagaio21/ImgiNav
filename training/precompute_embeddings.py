import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import argparse
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from modules.autoencoder import AutoEncoder


def load_image(path):
    """Load image and convert to tensor."""
    img = Image.open(path).convert("RGB")
    return transforms.ToTensor()(img)


def validate_embedding_shape(z, expected_dims=3):
    """
    Validate and fix embedding shape.
    Expected shape: (channels, height, width) - 3D tensor
    """
    if z.dim() == 4 and z.shape[0] == 1:
        # Remove batch dimension if present
        z = z.squeeze(0)
    
    if z.dim() != expected_dims:
        raise ValueError(f"Expected {expected_dims}D tensor, got {z.dim()}D with shape {z.shape}")
    
    return z


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_latent_dir, exist_ok=True)
    
    # Create log file
    log_path = os.path.join(args.output_latent_dir, "encoding_log.txt")
    
    print(f"[1] Loading manifest: {args.manifest}")
    df = pd.read_csv(args.manifest)
    print(f"    Found {len(df)} entries to process")
    
    print(f"[2] Loading AutoEncoder from {args.autoencoder_config}")
    ae = AutoEncoder.from_config(args.autoencoder_config)
    
    # Load checkpoint
    checkpoint = torch.load(args.autoencoder_ckpt, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        ae.load_state_dict(checkpoint['model'])
    else:
        ae.load_state_dict(checkpoint)
    
    ae.eval().to(device)
    
    # Get expected latent shape from a test encoding
    print("[3] Testing encoder output shape...")
    with torch.no_grad():
        test_img = torch.randn(1, 3, 256, 256).to(device)  # Dummy image
        test_z = ae.encode_latent(test_img)
        expected_shape = test_z.shape[1:]  # Remove batch dimension
    print(f"    Expected latent shape: {expected_shape}")
    
    # Pre-create directory structure
    os.makedirs(os.path.join(args.output_latent_dir, "rooms"), exist_ok=True)
    os.makedirs(os.path.join(args.output_latent_dir, "scenes"), exist_ok=True)
    
    latent_paths = []
    success_count = 0
    error_count = 0
    
    # Log file
    with open(log_path, 'w') as log:
        log.write(f"Encoding log for {args.manifest}\n")
        log.write(f"Expected latent shape: {expected_shape}\n")
        log.write("=" * 50 + "\n")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Encoding layouts"):
            layout_path = row["layout_path"]
            scene_id = str(row["scene_id"])
            room_id = str(row.get("room_id", "scene"))
            entry_type = row.get("type", "scene")
            
            try:
                # Load and encode
                img = load_image(layout_path).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    z = ae.encode_latent(img, deterministic=True)  # Use deterministic encoding
                    
                # Validate and fix shape
                z = validate_embedding_shape(z, expected_dims=3)
                
                # Verify shape matches expected
                if z.shape != expected_shape:
                    raise ValueError(f"Shape mismatch: got {z.shape}, expected {expected_shape}")
                
                # Save path
                subdir = "rooms" if entry_type == "room" else "scenes"
                latent_file = os.path.join(
                    args.output_latent_dir, subdir, 
                    f"{scene_id}_{room_id}_{entry_type}.pt"
                )
                
                # Save as half precision to save space
                torch.save(z.half().cpu(), latent_file)
                latent_paths.append(latent_file)
                success_count += 1
                
                # Log success every 100 items
                if success_count % 100 == 0:
                    log.write(f"[Success] Processed {success_count} items\n")
                    
            except Exception as e:
                error_msg = f"Failed on {layout_path}: {str(e)}"
                print(f"\n[Error] {error_msg}")
                log.write(f"[Error] Row {idx}: {error_msg}\n")
                latent_paths.append(None)
                error_count += 1
    
    print(f"\n[4] Encoding complete:")
    print(f"    Successes: {success_count}")
    print(f"    Errors: {error_count}")
    
    # Add latent paths to dataframe
    print("\n[5] Creating new manifest with embedding paths...")
    df["layout_emb"] = latent_paths
    
    # Remove rows with failed encodings if desired
    if args.drop_failed:
        print(f"    Dropping {error_count} failed entries")
        df = df[df["layout_emb"].notna()]
    
    # Save new manifest
    df.to_csv(args.new_manifest, index=False)
    print(f"\n[✓] Saved new manifest → {args.new_manifest}")
    print(f"[✓] Latent files saved to → {args.output_latent_dir}")
    print(f"[✓] Encoding log saved to → {log_path}")
    
    # Verify a few saved embeddings
    print("\n[6] Verifying saved embeddings...")
    verify_count = min(5, success_count)
    valid_paths = [p for p in latent_paths if p is not None][:verify_count]
    
    for i, path in enumerate(valid_paths):
        try:
            loaded = torch.load(path, map_location='cpu')
            print(f"    Sample {i+1}: {Path(path).name} → shape {loaded.shape}")
        except Exception as e:
            print(f"    Sample {i+1}: Error loading {path}: {e}")


def verify_embeddings(manifest_path, n_samples=10):
    """Standalone function to verify embeddings in a manifest."""
    print(f"Verifying embeddings from: {manifest_path}")
    df = pd.read_csv(manifest_path)
    
    # Filter to entries with embeddings
    df_with_emb = df[df["layout_emb"].notna()]
    print(f"Found {len(df_with_emb)} entries with embeddings (out of {len(df)} total)")
    
    # Sample entries
    sample_size = min(n_samples, len(df_with_emb))
    samples = df_with_emb.sample(n=sample_size, random_state=42)
    
    shapes = []
    for _, row in samples.iterrows():
        try:
            emb = torch.load(row["layout_emb"], map_location='cpu')
            shapes.append(emb.shape)
            print(f"  {Path(row['layout_emb']).name}: shape={emb.shape}, dtype={emb.dtype}")
        except Exception as e:
            print(f"  Error loading {row['layout_emb']}: {e}")
    
    # Check consistency
    if shapes:
        unique_shapes = set(shapes)
        if len(unique_shapes) == 1:
            print(f"\n✓ All embeddings have consistent shape: {shapes[0]}")
        else:
            print(f"\n⚠ Warning: Found {len(unique_shapes)} different shapes: {unique_shapes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robustly precompute latent embeddings for layouts.")
    parser.add_argument("--manifest", required=True, help="Path to dataset CSV (layouts.csv)")
    parser.add_argument("--autoencoder_config", required=True, help="Path to AE config YAML")
    parser.add_argument("--autoencoder_ckpt", required=True, help="Path to AE checkpoint")
    parser.add_argument("--output_latent_dir", required=True, help="Directory to save latent .pt files")
    parser.add_argument("--new_manifest", required=True, help="Path to save new manifest CSV")
    parser.add_argument("--drop_failed", action="store_true", help="Drop failed entries from new manifest")
    parser.add_argument("--verify", action="store_true", help="Run verification after encoding")
    
    args = parser.parse_args()
    
    if args.verify and os.path.exists(args.new_manifest):
        # Just verify existing embeddings
        verify_embeddings(args.new_manifest)
    else:
        # Run the encoding
        main(args)
        if args.verify:
            verify_embeddings(args.new_manifest)