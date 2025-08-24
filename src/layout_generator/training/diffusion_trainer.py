import os
import torch
import tqdm
import json
import re
import glob
import shutil
from torchvision.utils import save_image
from ..utils.factories import create_model
from ..utils.logger import Logger

class DiffusionTrainer:
    def __init__(
        self,
        config_path: str,
        dataloader,
        optimizer_class,
        optimizer_params: dict,
        loss_fn,
        device,
        checkpoint_dir,
        image_dir,
        checkpoint_interval=10,
        sample_interval=5, # <-- New: How often to sample images
        num_samples=4,     # <-- New: Number of images to generate
        logger=None,
    ):
        """
        Initializes the trainer from a configuration file.
        """
        self.config_path = config_path
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.image_dir = image_dir
        self.checkpoint_interval = checkpoint_interval
        self.sample_interval = sample_interval # <-- New
        self.num_samples = num_samples         # <-- New
        self.start_epoch = 0
        
        log_file = os.path.join(checkpoint_dir, "training_log.txt")
        self.logger = logger or Logger(source=self.__class__.__name__)

        # 1. Build the model from the config file
        self.logger.info(f"Building model from config: {self.config_path}")
        self.model = create_model(self.config_path, device=self.device)
        
        # 2. Initialize the optimizer
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_params)

        # 3. Set up directories
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 4. Create a fixed conditioning vector for consistent sampling
        self.sample_conditions = self._get_persistent_conditions()

        # 5. Save metadata and try to resume
        self.save_metadata()
        self._try_resume_latest()

        self.logger.info(f"Trainer initialized for model: {self.model.name}")

    def _get_persistent_conditions(self):
        """Grabs a fixed batch of conditions from the dataloader for consistent sampling."""
        try:
            fixed_batch = next(iter(self.dataloader))
            conditions = fixed_batch["token_embedding"][:self.num_samples].to(self.device)
            self.logger.info(f"Created a fixed conditioning tensor of shape {conditions.shape} for sampling.")
            return conditions
        except Exception as e:
            self.logger.warning(f"Could not create fixed conditions for sampling: {e}. Will use random conditions.")
            return None

    def train(self, epochs):
        self.model.train()
        self.logger.info(f"Starting training from epoch {self.start_epoch + 1} for {epochs} epochs...")
        
        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            epoch_loss = 0.0
            iterator = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{self.start_epoch + epochs}")
            
            for batch in iterator:
                latents = batch["image_latent"].to(self.device)
                conditions = batch["token_embedding"].to(self.device)
                t = torch.randint(0, self.model.scheduler.config["num_timesteps"], (latents.shape[0],), device=self.device).long()
                
                pred_noise, true_noise, _ = self.model(latents, t, conditions)
                loss = self.loss_fn(pred_noise, true_noise)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(self.dataloader)
            self.logger.info(f"[Train] Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % self.checkpoint_interval == 0:
                self.save_model(epoch + 1)
            
            # --- New Sampling Logic ---
            if (epoch + 1) % self.sample_interval == 0:
                self._sample_and_save_images(epoch + 1)
        
        self.logger.info("✔ Training complete.")

    def _sample_and_save_images(self, epoch):
        """Generates and saves a grid of images from the diffusion model."""
        self.logger.info(f"--- Sampling images at epoch {epoch} ---")
        self.model.eval() # Switch to evaluation mode

        # Determine latent shape from the model's config
        # This assumes your UNet config has the latent dimensions, which is good practice
        try:
            # Example: Reading from UNet config. Adjust if needed.
            latent_channels = self.model.unet.info['in_channels']
            H = W = 32 # Assuming a latent size, you might want to configure this
            shape = (self.num_samples, latent_channels, H, W)
        except KeyError:
            self.logger.warning("Could not determine latent shape from config. Defaulting to (4, 4, 32, 32).")
            shape = (self.num_samples, 4, 32, 32)

        # Use the persistent conditions if available
        conditions = self.sample_conditions
        if conditions is None:
            # Fallback to random noise if persistent conditions failed
            cond_dim = self.model.unet.info.get('cond_dim', 128)
            conditions = torch.randn(self.num_samples, cond_dim).to(self.device)

        with torch.no_grad():
            generated_latents = self.model.sample(shape, conditions, device=self.device)
        
        # NOTE: This saves the raw latent space images. You would need to pass
        # them through your VAE decoder to see the final pixel-space images.
        # For now, we'll save the latents to visualize their structure.
        img_path = os.path.join(self.image_dir, f"sample_epoch_{epoch:03}.png")
        save_image(generated_latents.clamp(-1, 1) * 0.5 + 0.5, img_path)
        
        self.logger.info(f"✔ Saved sample image to {img_path}")
        self.model.train() # Switch back to training mode


    def save_model(self, epoch):
        model_path = os.path.join(self.checkpoint_dir, f"{self.model.name}_epoch{epoch:03}.pt")
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"[Checkpoint] Saved model to {model_path}")

    def save_metadata(self):
        meta_path = os.path.join(self.checkpoint_dir, f"{self.model.name}_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(self.model.info(), f, indent=4)
        shutil.copy(self.config_path, os.path.join(self.checkpoint_dir, "training_config.yaml"))
        self.logger.debug(f"Saved model metadata and training config to {self.checkpoint_dir}")

    def _try_resume_latest(self):
        self.logger.info("Searching for latest checkpoint...")
        pattern = os.path.join(self.checkpoint_dir, f"{self.model.name}_epoch*.pt")
        files = glob.glob(pattern)
        
        if not files:
            self.logger.info("No checkpoint found, starting from scratch.")
            return

        def extract_epoch(path):
            match = re.search(r"epoch(\d+)", path)
            return int(match.group(1)) if match else -1

        latest_file = sorted(files, key=extract_epoch)[-1]
        latest_epoch = extract_epoch(latest_file)
        
        if latest_epoch <= 0:
            self.logger.warning(f"Could not parse epoch from checkpoint: {latest_file}. Starting fresh.")
            return

        try:
            self.model.load_state_dict(torch.load(latest_file, map_location=self.device))
            self.start_epoch = latest_epoch
            self.logger.info(f"✅ [AutoResume] Resumed training from epoch {latest_epoch}")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from {latest_file}: {e}")