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
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers library not installed. Install with: pip install diffusers transformers accelerate")


class LayoutDataset(Dataset):
    """Dataset for layout images."""
    
    def __init__(self, image_dir, image_size=512, transform=None):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(list(self.image_dir.glob("*.png")) + list(self.image_dir.glob("*.jpg")))
        self.image_size = image_size
        
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
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return {"pixel_values": image}


def finetune_sd_unet(
    dataset_dir,
    output_dir,
    model_id="runwayml/stable-diffusion-v1-5",
    epochs=50,
    batch_size=4,
    learning_rate=1e-5,
    num_workers=4,
    save_steps=500,
    device="cuda",
    seed=42
):
    """
    Fine-tune Stable Diffusion UNet on layout dataset.
    
    Args:
        dataset_dir: Directory containing layout images
        output_dir: Directory to save fine-tuned model
        model_id: HuggingFace model ID
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        num_workers: DataLoader workers
        save_steps: Save checkpoint every N steps
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
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Only train UNet
    unet.requires_grad_(True)
    unet.train()
    
    print(f"Trainable parameters: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}")
    
    # Create dataset
    print(f"Loading dataset from: {dataset_dir}")
    dataset = LayoutDataset(dataset_dir, image_size=512)
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
    
    # Use gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    
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
    
    for epoch in range(epochs):
        unet.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            # Get images
            images = batch["pixel_values"].to(device_obj, dtype=torch.float16 if device == "cuda" else torch.float32)
            
            # Encode images to latents
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Check for NaN/Inf in latents
                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    print(f"\nWARNING: NaN/Inf detected in latents at step {global_step}, skipping batch")
                    continue
            
            # Sample noise and timesteps
            noise = torch.randn_like(latents)
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
            
            # Predict noise with mixed precision
            if device == "cuda" and scaler is not None:
                with torch.cuda.amp.autocast():
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\nERROR: NaN/Inf loss detected at step {global_step}, epoch {epoch + 1}")
                    print(f"  Loss value: {loss.item()}")
                    print(f"  Model pred stats: min={model_pred.min().item():.4f}, max={model_pred.max().item():.4f}, mean={model_pred.mean().item():.4f}")
                    print(f"  Noise stats: min={noise.min().item():.4f}, max={noise.max().item():.4f}, mean={noise.mean().item():.4f}")
                    print(f"  Latents stats: min={latents.min().item():.4f}, max={latents.max().item():.4f}, mean={latents.mean().item():.4f}")
                    print("  Stopping training to prevent further issues.")
                    break
                
                # Backward with gradient scaling
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\nERROR: NaN/Inf loss detected at step {global_step}, epoch {epoch + 1}")
                    print(f"  Loss value: {loss.item()}")
                    print(f"  Model pred stats: min={model_pred.min().item():.4f}, max={model_pred.max().item():.4f}, mean={model_pred.mean().item():.4f}")
                    print(f"  Noise stats: min={noise.min().item():.4f}, max={noise.max().item():.4f}, mean={noise.mean().item():.4f}")
                    print(f"  Latents stats: min={latents.min().item():.4f}, max={latents.max().item():.4f}, mean={latents.mean().item():.4f}")
                    print("  Stopping training to prevent further issues.")
                    break
                
                # Backward
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
            
            # Save checkpoint
            if global_step % save_steps == 0:
                checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                # Save UNet
                unet.save_pretrained(checkpoint_dir / "unet")
                
                print(f"\nSaved checkpoint at step {global_step} to {checkpoint_dir}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch + 1}/{epochs} - Average Loss: {avg_loss:.6f}")
    
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
    parser.add_argument("--dataset_dir", type=Path, required=True,
                       help="Directory containing layout images")
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
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    finetune_sd_unet(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        save_steps=args.save_steps,
        device=args.device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

