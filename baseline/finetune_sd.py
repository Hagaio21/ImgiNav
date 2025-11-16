#!/usr/bin/env python3
"""
Fine-tune Stable Diffusion on layout dataset for baseline comparison.

This script fine-tunes SD's UNet on your layout images to create a baseline model
that can generate layouts (though likely not as good as your custom model).

Usage:
    python baseline/finetune_sd.py \
        --dataset_dir /path/to/layout/images \
        --output_dir outputs/baseline_sd_finetuned \
        --epochs 50 \
        --batch_size 4
"""

import argparse
import shutil
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from diffusers.optimization import get_scheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers library not installed. Install with: pip install diffusers transformers accelerate")


class LayoutDataset(Dataset):
    """Dataset for layout images from directory or manifest."""
    
    def __init__(self, image_dir=None, manifest_path=None, image_size=512, transform=None):
        self.image_size = image_size
        
        if manifest_path is not None:
            # Load from manifest
            import pandas as pd
            manifest_path = Path(manifest_path)
            self.manifest_dir = manifest_path.parent
            df = pd.read_csv(manifest_path)
            if "layout_path" in df.columns:
                self.image_paths = [Path(p) for p in df["layout_path"].tolist() if pd.notna(p)]
            elif "image_path" in df.columns:
                self.image_paths = [Path(p) for p in df["image_path"].tolist() if pd.notna(p)]
            else:
                raise ValueError("Manifest must contain 'layout_path' or 'image_path' column")
        elif image_dir is not None:
            # Load from directory
            self.image_dir = Path(image_dir)
            self.image_paths = sorted(list(self.image_dir.glob("*.png")) + list(self.image_dir.glob("*.jpg")))
        else:
            raise ValueError("Must provide either 'image_dir' or 'manifest_path'")
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # To [-1, 1]
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        if not img_path.is_absolute():
            # Try relative to current working directory
            if not img_path.exists():
                # Try relative to manifest directory if available
                if hasattr(self, 'manifest_dir'):
                    img_path = self.manifest_dir / img_path
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return {"pixel_values": image}


def finetune_sd_unet(
    dataset_dir=None,
    manifest_path=None,
    output_dir=None,
    model_id="runwayml/stable-diffusion-v1-5",
    epochs=50,
    batch_size=4,
    learning_rate=1e-5,
    num_workers=4,
    device="cuda",
    seed=42
):
    """
    Fine-tune Stable Diffusion UNet on layout dataset.
    
    Args:
        dataset_dir: Directory containing layout images (optional if manifest_path provided)
        manifest_path: Path to CSV manifest with layout_path column (optional if dataset_dir provided)
        output_dir: Directory to save fine-tuned model
        model_id: HuggingFace model ID
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        num_workers: DataLoader workers
        device: Device to use
        seed: Random seed
    """
    if not DIFFUSERS_AVAILABLE:
        raise ImportError("diffusers library required. Install with: pip install diffusers transformers accelerate")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device_obj = torch.device(device)
    
    print(f"Loading Stable Diffusion model: {model_id}")
    print(f"Device: {device}")
    
    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(device_obj)
    
    # Get components
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    unet = pipe.unet
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    
    # Freeze VAE and text encoder
    # Convert VAE to float32 for encoding (needed for dtype consistency)
    vae = vae.to(torch.float32)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Only train UNet
    # Convert UNet to float32 for training (gradient scaling requires float32)
    unet = unet.to(torch.float32)
    unet.requires_grad_(True)
    unet.train()
    
    print(f"Trainable parameters: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}")
    
    # Create dataset
    if manifest_path is not None:
        print(f"Loading dataset from manifest: {manifest_path}")
        dataset = LayoutDataset(manifest_path=manifest_path, image_size=512)
    elif dataset_dir is not None:
        print(f"Loading dataset from directory: {dataset_dir}")
        dataset = LayoutDataset(image_dir=dataset_dir, image_size=512)
    else:
        raise ValueError("Must provide either 'dataset_dir' or 'manifest_path'")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device == "cuda"
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Optimizer (reduced learning rate for stability)
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Use gradient scaler for mixed precision training (updated API)
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None
    
    # Learning rate scheduler
    num_update_steps_per_epoch = len(dataloader)
    max_train_steps = num_update_steps_per_epoch * epochs
    
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps
    )
    
    # Training loop
    global_step = 0
    progress_bar = tqdm(range(max_train_steps), desc="Training")
    best_loss = float('inf')
    best_checkpoint_dir = None
    
    for epoch in range(epochs):
        unet.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            # Get images
            images = batch["pixel_values"].to(device_obj)
            
            # Encode images to latents (VAE requires float32)
            with torch.no_grad():
                # VAE is in float32, so convert images to float32 for encoding
                images_fp32 = images.to(torch.float32)
                latents = vae.encode(images_fp32).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                # Latents are already in float32 (VAE output matches VAE dtype)
                
                # Check for NaN/Inf in latents
                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    print(f"\nWARNING: NaN/Inf detected in latents at step {global_step}, skipping batch")
                    continue
            
            # Sample noise and timesteps (in float32 to match UNet)
            noise = torch.randn_like(latents, dtype=torch.float32)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device_obj)
            
            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get text embeddings (use empty prompt for unconditional)
            with torch.no_grad():
                prompt = [""] * latents.shape[0]  # Empty prompt for unconditional
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                text_embeddings = text_encoder(text_inputs.input_ids.to(device_obj))[0]
                # Convert to float32 for UNet training
                text_embeddings = text_embeddings.to(torch.float32)
            
            # Predict noise with mixed precision
            if device == "cuda" and scaler is not None:
                with torch.amp.autocast('cuda'):
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
                    loss = F.mse_loss(model_pred, noise, reduction="mean")
            else:
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
                loss = F.mse_loss(model_pred, noise, reduction="mean")
            
            # Check for NaN/Inf (moved outside if/else to avoid duplication)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nERROR: NaN/Inf loss detected at step {global_step}, epoch {epoch + 1}")
                print(f"  Loss value: {loss.item()}")
                print(f"  Model pred stats: min={model_pred.min().item():.4f}, max={model_pred.max().item():.4f}, mean={model_pred.mean().item():.4f}")
                print(f"  Noise stats: min={noise.min().item():.4f}, max={noise.max().item():.4f}, mean={noise.mean().item():.4f}")
                print(f"  Latents stats: min={latents.min().item():.4f}, max={latents.max().item():.4f}, mean={latents.mean().item():.4f}")
                print("  Stopping training to prevent further issues.")
                break
            
            # Backward pass
            if device == "cuda" and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                optimizer.step()
            
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Update progress
            epoch_loss += loss.item()
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item(), "epoch": epoch + 1})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch + 1}/{epochs} - Average Loss: {avg_loss:.6f}")
        
        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            
            # Remove old best checkpoint if it exists
            if best_checkpoint_dir is not None and best_checkpoint_dir.exists():
                shutil.rmtree(best_checkpoint_dir)
            
            # Save new best checkpoint
            best_checkpoint_dir = output_dir / "checkpoint-best"
            best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            unet.save_pretrained(best_checkpoint_dir / "unet")
            
            # Also save pipeline from best checkpoint for easy sampling
            # Update pipe with current best UNet and save
            pipe.unet = unet
            pipe.save_pretrained(best_checkpoint_dir / "pipeline")
            
            print(f"  ✓ New best checkpoint saved (loss: {best_loss:.6f})")
    
    # Save final model
    print(f"\nSaving final model to {output_dir}")
    unet.save_pretrained(output_dir / "unet")
    
    # Save full pipeline for easy loading
    pipe.unet = unet
    pipe.save_pretrained(output_dir / "pipeline")
    
    print(f"\n{'='*60}")
    print("Fine-tuning Complete!")
    print(f"Model saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion on layout dataset")
    parser.add_argument("--dataset_dir", type=Path, default=None,
                       help="Directory containing layout images (optional if --manifest_path provided)")
    parser.add_argument("--manifest_path", type=Path, default=None,
                       help="Path to CSV manifest with layout_path column (optional if --dataset_dir provided)")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Output directory for fine-tuned model")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5",
                       help="HuggingFace model ID")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    if args.dataset_dir is None and args.manifest_path is None:
        parser.error("Must provide either --dataset_dir or --manifest_path")
    
    finetune_sd_unet(
        dataset_dir=args.dataset_dir,
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        model_id=args.model_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

