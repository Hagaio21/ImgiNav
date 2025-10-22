import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import json


class PipelineTrainer:
    """
    Trainer for DiffusionPipeline with comprehensive evaluation:
    - Training/validation loss curves
    - Condition correlation tracking
    - Conditioned vs unconditioned samples (from fixed noise)
    - Per-sample output directories with conditions saved
    """
    def __init__(
        self,
        pipeline,
        sample_loader,
        optimizer=None,
        loss_fn=None,
        epochs=100,
        lr=1e-4,
        weight_decay=0.0,
        grad_clip=None,
        log_interval=100,
        eval_interval=1000,
        sample_interval=2000,
        eval_num_samples=8,
        ckpt_dir="checkpoints",
        output_dir="outputs",
        mixed_precision=False,
        ema_decay=None,
        cond_dropout_pov=0.0,
        cond_dropout_graph=0.0,
        cond_dropout_both=0.0,
        taxonomy=None,
        use_modalities="both",
    ):
        self.pipeline = pipeline
        self.sample_loader = sample_loader
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.sample_interval = sample_interval
        self.eval_num_samples = eval_num_samples
        self.ckpt_dir = Path(ckpt_dir)
        self.output_dir = Path(output_dir)
        self.mixed_precision = mixed_precision
        self.ema_decay = ema_decay
        self.taxonomy = taxonomy
        
        # Ablation settings
        self.use_modalities = use_modalities.lower()
        self.use_pov = self.use_modalities in ["pov_only", "both"]
        self.use_graph = self.use_modalities in ["graph_only", "both"]
        
        # Conditional dropout probabilities
        self.cond_dropout_pov = cond_dropout_pov
        self.cond_dropout_graph = cond_dropout_graph
        self.cond_dropout_both = cond_dropout_both
        
        # Create directories
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "samples").mkdir(exist_ok=True)
        (self.output_dir / "curves").mkdir(exist_ok=True)
        
        # Setup optimizer
        if optimizer is None:
            trainable_params = []
            trainable_params.extend(self.pipeline.unet.parameters())
            trainable_params.extend(self.pipeline.mixer.parameters())
            self.optimizer = torch.optim.AdamW(
                trainable_params, lr=lr, weight_decay=weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # Setup loss
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
        
        # Setup mixed precision
        if mixed_precision:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # EMA model
        self.ema_model = None
        if ema_decay is not None:
            self.ema_model = self._create_ema_model()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_steps = []
        self.val_steps = []
        self.correlation_history = {
            'pov': [],
            'graph': [],
            'steps': []
        }
        
        # Fixed noise for consistent sampling
        self.fixed_noise = None
        self.fixed_batch = None
        
        print(f"[Trainer] Initialized with modality mode: {self.use_modalities}")
        print(f"  - Use POV: {self.use_pov}")
        print(f"  - Use Graph: {self.use_graph}")
        print(f"  - Conditional dropout: POV={cond_dropout_pov}, Graph={cond_dropout_graph}, Both={cond_dropout_both}")

    def _create_ema_model(self):
        """Create EMA shadow model."""
        import copy
        ema = copy.deepcopy(self.pipeline)
        ema.eval()
        for param in ema.parameters():
            param.requires_grad = False
        ema.to(self.pipeline.device)
        return ema

    def _update_ema(self):
        """Update EMA model parameters."""
        if self.ema_model is None:
            return
        
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(), self.pipeline.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    model_param.data, alpha=1 - self.ema_decay
                )

    def _apply_conditional_dropout(self, pov_emb, graph_emb, has_pov_mask=None):
        """
        Apply conditional dropout during training.
        """
        if not self.pipeline.training:
            return pov_emb, graph_emb
        
        # Determine batch size
        if pov_emb is not None:
            batch_size = pov_emb.shape[0]
            device = pov_emb.device
        elif graph_emb is not None:
            batch_size = graph_emb.shape[0]
            device = graph_emb.device
        else:
            return None, None
        
        # Generate dropout masks
        drop_pov = torch.rand(batch_size, device=device) < self.cond_dropout_pov
        drop_graph = torch.rand(batch_size, device=device) < self.cond_dropout_graph
        drop_both = torch.rand(batch_size, device=device) < self.cond_dropout_both
        
        # Apply "drop both" first
        drop_pov = drop_pov | drop_both
        drop_graph = drop_graph | drop_both
        
        # Only drop POV for samples that actually have POVs
        if has_pov_mask is not None and pov_emb is not None:
            if not isinstance(has_pov_mask, torch.Tensor):
                has_pov_mask = torch.tensor(has_pov_mask, device=device, dtype=torch.bool)
            drop_pov = drop_pov & has_pov_mask.to(device)
        
        # Apply dropout
        if pov_emb is not None and drop_pov.any():
            pov_emb = pov_emb.clone()
            pov_emb[drop_pov] = 0
        
        if graph_emb is not None and drop_graph.any():
            graph_emb = graph_emb.clone()
            graph_emb[drop_graph] = 0
        
        return pov_emb, graph_emb

    def _prepare_batch(self, batch):
        """
        Prepare batch for training: embed conditions and handle mixed batches.
        """
        layout = batch["layout"].to(self.pipeline.device)
        
        # Get raw data
        pov_raw = batch.get("pov")
        graph_raw = batch.get("graph")
        
        # Determine which samples have POVs
        if pov_raw is not None and isinstance(pov_raw, list):
            has_pov_mask = torch.tensor(
                [p is not None for p in pov_raw],
                device=self.pipeline.device,
                dtype=torch.bool
            )
        else:
            has_pov_mask = torch.zeros(layout.shape[0], device=self.pipeline.device, dtype=torch.bool)
        
        # Embed conditions
        pov_emb, graph_emb = self.pipeline._prepare_conditions(
            pov_raw=pov_raw if self.use_pov else None,
            graph_raw=graph_raw if self.use_graph else None,
            use_pov=self.use_pov,
            use_graph=self.use_graph
        )
        
        # Apply conditional dropout
        pov_emb, graph_emb = self._apply_conditional_dropout(pov_emb, graph_emb, has_pov_mask)
        
        return layout, pov_emb, graph_emb, has_pov_mask

    def train_step(self, batch):
        """Single training step with correlation tracking."""
        self.optimizer.zero_grad()
        
        # Prepare batch
        layout, pov_emb, graph_emb, has_pov_mask = self._prepare_batch(batch)
        
        # Encode layout to latent space
        with torch.no_grad():
            latents = self.pipeline.encode_layout(layout)
        
        # Sample random timesteps
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0, 
            self.pipeline.scheduler.num_steps, 
            (batch_size,), 
            device=self.pipeline.device
        ).long()
        
        # Add noise to latents
        noise = torch.randn_like(latents)
        noisy_latents = self.pipeline.scheduler.add_noise(latents, noise, timesteps)
        
        # Forward pass
        if self.mixed_precision:
            with autocast():
                noise_pred = self.pipeline(noisy_latents, pov_emb, graph_emb, timesteps)
                loss = self.loss_fn(noise_pred, noise)
        else:
            noise_pred = self.pipeline(noisy_latents, pov_emb, graph_emb, timesteps)
            loss = self.loss_fn(noise_pred, noise)
        
        # Compute correlation between noise prediction and conditions
        from pipeline.pipeline import compute_condition_correlation
        correlation = compute_condition_correlation(noise_pred, [pov_emb, graph_emb])
        
        # Backward pass
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.pipeline.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.pipeline.parameters(), self.grad_clip)
            self.optimizer.step()
        
        # Update EMA
        if self.ema_model is not None:
            self._update_ema()
        
        return {
            "loss": loss.item(),
            "num_pov": has_pov_mask.sum().item(),
            "num_no_pov": (~has_pov_mask).sum().item(),
            "correlation": correlation
        }

    def fit(self, train_loader, val_loader=None):
        """Main training loop."""
        print(f"[Trainer] Starting training for {self.epochs} epochs")
        print(f"[Trainer] Total steps per epoch: {len(train_loader)}")
        
        # Initialize fixed batch for consistent sampling
        if self.fixed_batch is None:
            self.fixed_batch = next(iter(self.sample_loader))
            # Create fixed noise based on batch size
            batch_size = self.fixed_batch["layout"].shape[0]
            self.fixed_noise = torch.randn(
                batch_size,
                self.pipeline.autoencoder.encoder.latent_channels,
                self.pipeline.autoencoder.encoder.latent_base,
                self.pipeline.autoencoder.encoder.latent_base,
                device=self.pipeline.device
            )
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            self.pipeline.train()
            
            epoch_loss = 0.0
            epoch_pov_count = 0
            epoch_no_pov_count = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                # Training step
                metrics = self.train_step(batch)
                
                epoch_loss += metrics["loss"]
                epoch_pov_count += metrics["num_pov"]
                epoch_no_pov_count += metrics["num_no_pov"]
                
                # Track metrics
                self.train_losses.append(metrics["loss"])
                self.train_steps.append(self.global_step)
                
                # Track correlation
                if metrics["correlation"]:
                    for key, val in metrics["correlation"].items():
                        self.correlation_history[key].append(val)
                    self.correlation_history['steps'].append(self.global_step)
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "pov": metrics["num_pov"],
                    "no_pov": metrics["num_no_pov"]
                })
                
                # Logging
                if self.global_step % self.log_interval == 0:
                    corr_str = ", ".join([f"{k}: {v:.3f}" for k, v in metrics["correlation"].items()])
                    print(f"\n[Step {self.global_step}] Loss: {metrics['loss']:.4f}, Correlation: {corr_str}")
                
                # Evaluation
                if val_loader is not None and self.global_step % self.eval_interval == 0:
                    val_loss = self.evaluate(val_loader)
                    self.val_losses.append(val_loss)
                    self.val_steps.append(self.global_step)
                    self.pipeline.train()
                    
                    # Plot curves
                    self.plot_curves()
                
                # Sampling with fixed noise
                if self.global_step % self.sample_interval == 0:
                    self.sample_and_save_comparison()
                    self.pipeline.train()
                
                self.global_step += 1
            
            # Epoch summary
            avg_loss = epoch_loss / len(train_loader)
            print(f"\n[Epoch {epoch+1}] Average Loss: {avg_loss:.4f}")
            print(f"  - Samples with POV: {epoch_pov_count}")
            print(f"  - Samples without POV: {epoch_no_pov_count}")
            
            # Save checkpoint
            self.save_checkpoint(f"epoch_{epoch+1}.pt")
            
            # End of epoch evaluation
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                self.val_losses.append(val_loss)
                self.val_steps.append(self.global_step)
                self.plot_curves()
                self.pipeline.train()
        
        # Final comprehensive evaluation
        print("\n[Final Evaluation] Generating comprehensive comparison...")
        self.sample_and_save_comparison(final=True)

    @torch.no_grad()
    def evaluate(self, val_loader):
        """Evaluation loop."""
        self.pipeline.eval()
        print(f"\n[Eval] Running evaluation at step {self.global_step}")
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in val_loader:
            layout, pov_emb, graph_emb, has_pov_mask = self._prepare_batch(batch)
            
            # Encode layout
            latents = self.pipeline.encode_layout(layout)
            
            # Sample timesteps
            batch_size = latents.shape[0]
            timesteps = torch.randint(
                0, 
                self.pipeline.scheduler.num_steps, 
                (batch_size,), 
                device=self.pipeline.device
            ).long()
            
            # Add noise
            noise = torch.randn_like(latents)
            noisy_latents = self.pipeline.scheduler.add_noise(latents, noise, timesteps)
            
            # Predict noise
            noise_pred = self.pipeline(noisy_latents, pov_emb, graph_emb, timesteps)
            loss = self.loss_fn(noise_pred, noise)
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches >= 10:  # Evaluate on 10 batches
                break
        
        avg_val_loss = total_loss / num_batches
        print(f"[Eval] Validation Loss: {avg_val_loss:.4f}")
        
        return avg_val_loss

    @torch.no_grad()
    def sample_and_save_comparison(self, final=False):
        """
        Generate conditioned vs unconditioned samples from fixed noise.
        Save in per-sample directories with conditions.
        """
        self.pipeline.eval()
        
        # Prepare conditions from fixed batch
        layout, pov_emb, graph_emb, has_pov_mask = self._prepare_batch(self.fixed_batch)
        
        # Take subset for sampling
        num_samples = min(self.eval_num_samples, layout.shape[0])
        layout = layout[:num_samples]
        
        if pov_emb is not None:
            pov_emb = pov_emb[:num_samples]
        if graph_emb is not None:
            graph_emb = graph_emb[:num_samples]
        
        fixed_noise = self.fixed_noise[:num_samples]
        
        # Generate conditioned samples (with POV and Graph)
        print("[Sampling] Generating conditioned samples...")
        conditioned_samples = self.pipeline.sample(
            batch_size=num_samples,
            cond_pov_emb=pov_emb,
            cond_graph_emb=graph_emb,
            image=True,
            noise=fixed_noise,
            use_pov=self.use_pov,
            use_graph=self.use_graph
        )
        
        # Generate unconditioned samples (no conditions)
        print("[Sampling] Generating unconditioned samples...")
        unconditioned_samples = self.pipeline.sample(
            batch_size=num_samples,
            cond_pov_emb=None,
            cond_graph_emb=None,
            image=True,
            noise=fixed_noise,
            use_pov=False,
            use_graph=False
        )
        
        # Create output directory for this step
        step_dir = self.output_dir / "samples" / f"step_{self.global_step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        
        # Save per-sample comparisons
        for i in range(num_samples):
            sample_dir = step_dir / f"sample_{i:02d}"
            sample_dir.mkdir(exist_ok=True)
            
            # Save target
            vutils.save_image(
                layout[i],
                sample_dir / "target.png",
                normalize=True,
                value_range=(-1, 1)
            )
            
            # Save conditioned sample
            vutils.save_image(
                conditioned_samples[i],
                sample_dir / "conditioned.png",
                normalize=True,
                value_range=(-1, 1)
            )
            
            # Save unconditioned sample
            vutils.save_image(
                unconditioned_samples[i],
                sample_dir / "unconditioned.png",
                normalize=True,
                value_range=(-1, 1)
            )
            
            # Save POV condition if available
            if "pov" in self.fixed_batch and self.fixed_batch["pov"] is not None:
                pov_list = self.fixed_batch["pov"]
                if isinstance(pov_list, list) and i < len(pov_list) and pov_list[i] is not None:
                    pov_list[i].save(sample_dir / "pov_condition.png")
            
            # Save graph text condition
            if "graph" in self.fixed_batch and self.fixed_batch["graph"] is not None:
                graph_list = self.fixed_batch["graph"]
                if isinstance(graph_list, list) and i < len(graph_list):
                    with open(sample_dir / "graph_condition.txt", "w") as f:
                        f.write(graph_list[i])
            
            # Save metadata
            metadata = {
                "step": self.global_step,
                "epoch": self.current_epoch,
                "sample_id": self.fixed_batch.get("sample_id", [None] * num_samples)[i] if "sample_id" in self.fixed_batch else None,
                "has_pov": has_pov_mask[i].item() if i < len(has_pov_mask) else False,
                "use_modalities": self.use_modalities
            }
            with open(sample_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        
        # Create grid comparison
        grid_images = []
        for i in range(num_samples):
            grid_images.extend([
                layout[i],
                conditioned_samples[i],
                unconditioned_samples[i]
            ])
        
        grid = vutils.make_grid(grid_images, nrow=3, normalize=True, value_range=(-1, 1))
        vutils.save_image(grid, step_dir / "comparison_grid.png")
        
        print(f"[Sampling] Saved comparison to {step_dir}")

    def plot_curves(self):
        """Plot training/validation loss and correlation curves."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curve
        ax1 = axes[0]
        if self.train_losses:
            ax1.plot(self.train_steps, self.train_losses, label='Train Loss', alpha=0.6)
        if self.val_losses:
            ax1.plot(self.val_steps, self.val_losses, label='Val Loss', marker='o', linestyle='--')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Correlation curve
        ax2 = axes[1]
        if self.correlation_history['steps']:
            if self.correlation_history['pov']:
                ax2.plot(self.correlation_history['steps'], 
                        self.correlation_history['pov'], 
                        label='POV Correlation', alpha=0.7)
            if self.correlation_history['graph']:
                ax2.plot(self.correlation_history['steps'], 
                        self.correlation_history['graph'], 
                        label='Graph Correlation', alpha=0.7)
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Cosine Similarity')
        ax2.set_title('Condition Correlation (Denoised vs Conditions)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "curves" / f"metrics_step_{self.global_step:06d}.png", dpi=150)
        plt.close()

    def save_checkpoint(self, filename):
        """Save training checkpoint."""
        ckpt_path = self.ckpt_dir / filename
        
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "unet_state_dict": self.pipeline.unet.state_dict(),
            "mixer_state_dict": self.pipeline.mixer.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_steps": self.train_steps,
            "val_steps": self.val_steps,
            "correlation_history": self.correlation_history,
        }
        
        if self.ema_model is not None:
            checkpoint["ema_unet_state_dict"] = self.ema_model.unet.state_dict()
            checkpoint["ema_mixer_state_dict"] = self.ema_model.mixer.state_dict()
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, ckpt_path)
        print(f"[Checkpoint] Saved to {ckpt_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.pipeline.device)
        
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.pipeline.unet.load_state_dict(checkpoint["unet_state_dict"])
        self.pipeline.mixer.load_state_dict(checkpoint["mixer_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load metric history
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.train_steps = checkpoint.get("train_steps", [])
        self.val_steps = checkpoint.get("val_steps", [])
        self.correlation_history = checkpoint.get("correlation_history", {'pov': [], 'graph': [], 'steps': []})
        
        # Move scheduler to device
        self.pipeline.scheduler.to(self.pipeline.device)
        
        if self.ema_model is not None and "ema_unet_state_dict" in checkpoint:
            self.ema_model.unet.load_state_dict(checkpoint["ema_unet_state_dict"])
            self.ema_model.mixer.load_state_dict(checkpoint["ema_mixer_state_dict"])
            self.ema_model.scheduler.to(self.pipeline.device)
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        print(f"[Checkpoint] Loaded from {checkpoint_path}")
        print(f"  - Resuming from epoch {self.current_epoch}, step {self.global_step}")