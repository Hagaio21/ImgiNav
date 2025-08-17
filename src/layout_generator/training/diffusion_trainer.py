import os
import torch
import tqdm
import json
import re
import glob
from models.diffusion import DiffusionModel
from utils.logger import Logger




class DiffusionTrainer:
    def __init__(
        self,
        model: DiffusionModel,
        dataloader,
        optimizer,
        loss_fn,
        device,
        checkpoint_dir,
        image_dir,
        checkpoint_interval=10,
        checkpoint_regex=r"epoch(\d+)", # Refactored
        logger=None,
    ):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.image_dir = image_dir
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_regex = checkpoint_regex # Refactored
        self.start_epoch = 0

        log_file = os.path.join(checkpoint_dir, f"{model.name}_log.txt")
        self.logger = logger or Logger(
            debug=False, info=True, log_file=log_file,
            source=f"{self.__class__.__name__}:{model.name}"
        )
        
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_metadata()
        self._try_resume_latest()
        
        self.logger.info(f"Training model: {self.model.name} on device: {self.device}")

    def train(self, epochs):
        self.model.train()
        self.logger.info(f"Starting training from epoch {self.start_epoch + 1} for {epochs} epochs...")

        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            epoch_loss = 0.0
            iterator = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{self.start_epoch + epochs}")

            for batch in iterator:
                # Assuming batch is a dict with 'image_latent' and 'token_embedding'
                latents = batch["image_latent"].to(self.device)
                conditions = batch["token_embedding"].to(self.device)
                
                # Timesteps t ~ U(0, T)
                t = torch.randint(0, self.model.scheduler.config["num_timesteps"], (latents.shape[0],), device=self.device).long()

                pred_noise, true_noise, _ = self.model(latents, t, conditions)
                loss = self.loss_fn(pred_noise, true_noise)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.dataloader)
            self.logger.info(f"[Train] Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")
            self.logger.log_metric("avg_loss", avg_loss, step=epoch + 1)
            
            if (epoch + 1) % self.checkpoint_interval == 0:
                self.save_model(epoch + 1)
        
        self.logger.info("✓ Training complete.")

    def save_model(self, epoch):
        model_path = os.path.join(self.checkpoint_dir, f"{self.model.name}_epoch{epoch:03}.pt")
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"[Checkpoint] Saved model to {model_path}")

    def save_metadata(self):
        meta_path = os.path.join(self.checkpoint_dir, f"{self.model.name}_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(self.model.info(), f, indent=4)
        self.logger.debug(f"Saved model metadata to {meta_path}")

    def _try_resume_latest(self):
        pattern = os.path.join(self.checkpoint_dir, f"{self.model.name}_epoch*.pt")
        files = glob(pattern)
        if not files:
            self.logger.info("No checkpoint found, starting from scratch.")
            return

        def extract_epoch(path):
            # Use the configurable regex
            match = re.search(self.checkpoint_regex, path) # Refactored
            return int(match.group(1)) if match else -1

        latest_file = sorted(files, key=extract_epoch)[-1]
        latest_epoch = extract_epoch(latest_file)

        if latest_epoch <= 0:
            self.logger.warning(f"Could not parse epoch from checkpoint: {latest_file}")
            return
        
        try:
            self.model.load(latest_file, map_location=self.device)
            self.start_epoch = latest_epoch
            self.logger.info(f"✅ [AutoResume] Resumed training from epoch {latest_epoch}")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from {latest_file}: {e}")


