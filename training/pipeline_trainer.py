import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import torchvision.utils as vutils


class PipelineTrainer:
    """
    Trainer for DiffusionPipeline with ablation support and mixed batch handling.
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
        use_modalities="both",  # 'pov_only', 'graph_only', 'both', 'none'
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
        self.scaler = GradScaler() if mixed_precision else None
        
        # EMA model
        self.ema_model = None
        if ema_decay is not None:
            self.ema_model = self._create_ema_model()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
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
        Respects ablation settings and only drops conditions that are actually used.
        
        Args:
            pov_emb: POV embeddings [B, D] or None
            graph_emb: Graph embeddings [B, D] or None
            has_pov_mask: Boolean tensor [B] indicating which samples have POVs
        
        Returns:
            Tuple of (pov_emb, graph_emb) with dropout applied
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
        
        # Only drop POV for samples that actually have POVs (respect has_pov_mask)
        if has_pov_mask is not None and pov_emb is not None:
            # has_pov_mask should be a boolean tensor on the same device
            if not isinstance(has_pov_mask, torch.Tensor):
                has_pov_mask = torch.tensor(has_pov_mask, device=device, dtype=torch.bool)
            drop_pov = drop_pov & has_pov_mask.to(device)
        
        # Apply dropout (zero out embeddings)
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
        
        Returns:
            layout: Tensor [B, C, H, W]
            pov_emb: Tensor [B, D] or None
            graph_emb: Tensor [B, D] or None
            has_pov_mask: Boolean tensor [B]
        """
        layout = batch["layout"].to(self.pipeline.device)
        
        # Get raw data
        pov_raw = batch.get("pov")  # List of PIL Images or None values
        graph_raw = batch.get("graph")  # List of strings
        
        # Determine which samples have POVs
        if pov_raw is not None and isinstance(pov_raw, list):
            has_pov_mask = torch.tensor(
                [p is not None for p in pov_raw],
                device=self.pipeline.device,
                dtype=torch.bool
            )
        else:
            has_pov_mask = torch.zeros(layout.shape[0], device=self.pipeline.device, dtype=torch.bool)
        
        # Embed conditions using pipeline's embedder (respecting ablation settings)
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
        """Single training step."""
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
        
        # Forward pass with mixed precision
        if self.mixed_precision:
            with autocast():
                noise_pred = self.pipeline(noisy_latents, pov_emb, graph_emb, timesteps)
                loss = self.loss_fn(noise_pred, noise)
        else:
            noise_pred = self.pipeline(noisy_latents, pov_emb, graph_emb, timesteps)
            loss = self.loss_fn(noise_pred, noise)
        
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
        }

    def fit(self, train_loader, val_loader=None):
        """Main training loop."""
        print(f"[Trainer] Starting training for {self.epochs} epochs")
        print(f"[Trainer] Total steps per epoch: {len(train_loader)}")
        
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
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "pov": metrics["num_pov"],
                    "no_pov": metrics["num_no_pov"]
                })
                
                # Logging
                if self.global_step % self.log_interval == 0:
                    print(f"\n[Step {self.global_step}] Loss: {metrics['loss']:.4f}")
                
                # Evaluation
                if val_loader is not None and self.global_step % self.eval_interval == 0:
                    self.evaluate(val_loader)
                    self.pipeline.train()
                
                # Sampling
                if self.global_step % self.sample_interval == 0:
                    self.sample_and_save(batch)
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
                self.evaluate(val_loader)
                self.pipeline.train()

    @torch.no_grad()
    def evaluate(self, val_loader):
        """Evaluation loop."""
        self.pipeline.eval()
        print(f"\n[Eval] Running evaluation at step {self.global_step}")
        
        val_batch = next(iter(val_loader))
        
        metrics = self.pipeline.evaluate(
            val_batch,
            num_samples=self.eval_num_samples,
            step=self.global_step,
            use_pov=self.use_pov,
            use_graph=self.use_graph
        )
        
        print(f"[Eval] Metrics: {metrics}")
        return metrics

    @torch.no_grad()
    def sample_and_save(self, batch):
        """Generate and save samples."""
        self.pipeline.eval()
        
        # Prepare conditions from batch
        layout, pov_emb, graph_emb, has_pov_mask = self._prepare_batch(batch)
        
        # Take first sample from batch
        if pov_emb is not None:
            pov_emb = pov_emb[:1]
        if graph_emb is not None:
            graph_emb = graph_emb[:1]
        
        # Generate sample
        samples = self.pipeline.sample(
            batch_size=1,
            cond_pov_emb=pov_emb,
            cond_graph_emb=graph_emb,
            image=True,
            use_pov=self.use_pov,
            use_graph=self.use_graph
        )
        
        # Save
        save_path = self.output_dir / f"sample_step_{self.global_step}.png"
        vutils.save_image(
            samples,
            save_path,
            normalize=True,
            value_range=(-1, 1)
        )
        print(f"[Sample] Saved to {save_path}")

    def save_checkpoint(self, filename):
        """Save training checkpoint."""
        ckpt_path = self.ckpt_dir / filename
        
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "unet_state_dict": self.pipeline.unet.state_dict(),
            "mixer_state_dict": self.pipeline.mixer.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
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
        
        if self.ema_model is not None and "ema_unet_state_dict" in checkpoint:
            self.ema_model.unet.load_state_dict(checkpoint["ema_unet_state_dict"])
            self.ema_model.mixer.load_state_dict(checkpoint["ema_mixer_state_dict"])
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        print(f"[Checkpoint] Loaded from {checkpoint_path}")
        print(f"  - Resuming from epoch {self.current_epoch}, step {self.global_step}")