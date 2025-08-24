import os
import torch
import yaml
from tqdm import tqdm # <-- CORRECTED IMPORT
import json
import re
import glob
import shutil
from torchvision.utils import save_image
from ..utils.factories import create_model
from ..utils.logger import Logger
from ..models.diffusion import DiffusionModel
from ..modules.scheduler import NoiseScheduler

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
        sample_interval=5,
        num_samples=4,
        logger=None,
    ):
        self.config_path = config_path
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.image_dir = image_dir
        self.checkpoint_interval = checkpoint_interval
        self.sample_interval = sample_interval
        self.num_samples = num_samples
        self.start_epoch = 0

        log_file = os.path.join(checkpoint_dir, "training_log.txt")
        self.logger = logger or Logger(source=self.__class__.__name__, log_file=log_file)

        # 1. Load the main configuration file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        diffusion_params = config['model']['params']
        
        # 2. Build the UNet component using the factory
        self.logger.info("Building UNet component...")
        unet_config_dict = {'model': diffusion_params['unet']}
        unet = create_model(unet_config_dict, device=device)

        # 3. Build the Scheduler component directly
        self.logger.info("Building Scheduler component...")
        scheduler = NoiseScheduler(**diffusion_params['scheduler']).to(device)
        
        # 4. Assemble the final DiffusionModel
        self.logger.info("Assembling final DiffusionModel...")
        self.model = DiffusionModel(
            unet=unet,
            scheduler=scheduler,
            name=diffusion_params.get("name", "diffusion_model")
        ).to(device)

        # 5. Initialize the optimizer with the assembled model's parameters
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_params)

        # 6. Set up directories and persistent conditions for sampling
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.sample_conditions = self._get_persistent_conditions()
        
        # 7. Save metadata and try to resume from a checkpoint
        self.save_metadata()
        self._try_resume_latest()

        self.logger.info(f"Trainer initialized for model: {self.model.name}")
        self.logger.info(f"Training on device: {self.device}")


    def _get_persistent_conditions(self):
        """Grabs a fixed batch of conditions and repeats it to match num_samples."""
        try:
            fixed_batch = next(iter(self.dataloader))
            
            # --- THIS IS THE FIX ---
            # Take the very first condition from the batch
            first_condition = fixed_batch["token_embedding"][0]
            
            # Repeat it to create a batch of the desired sample size
            conditions = first_condition.unsqueeze(0).repeat(self.num_samples, 1).to(self.device)
            
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
                conditions = batch.get("token_embedding")
                if isinstance(conditions, torch.Tensor):
                    conditions = conditions.to(self.device)

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
            
            if (epoch + 1) % self.sample_interval == 0:
                self._sample_and_save_images(epoch + 1)
        
        self.logger.info("✔ Training complete.")


    def _sample_and_save_images(self, epoch):
        self.logger.info(f"--- Sampling images at epoch {epoch} ---")
        self.model.eval()

        try:
            latent_channels = self.model.unet.info['in_channels']
            H = W = 16 # Default latent size for test
            shape = (self.num_samples, latent_channels, H, W)
        except (KeyError, IndexError):
            shape = (self.num_samples, 4, 16, 16)

        conditions = self.sample_conditions
        if conditions is None:
            cond_dim = 16 # Default cond dim for test
            conditions = torch.randn(self.num_samples, cond_dim).to(self.device)

        with torch.no_grad():
            generated_latents = self.model.sample(shape, conditions, device=self.device)
        
        img_path = os.path.join(self.image_dir, f"sample_epoch_{epoch:03}.png")
        save_image(generated_latents.clamp(-1, 1) * 0.5 + 0.5, img_path)
        
        self.logger.info(f"✔ Saved sample image to {img_path}")
        self.model.train()


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