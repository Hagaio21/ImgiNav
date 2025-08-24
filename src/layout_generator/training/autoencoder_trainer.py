import torch
import os
import itertools
import shutil
from ..utils.logger import Logger
import tqdm
from torchvision.utils import save_image
import json
import re
import glob
from ..utils.factories import create_model

class AutoEncoderTrainer:
    def __init__(
        self,
        encoder_config_path: str,
        decoder_config_path: str,
        dataloader,
        optimizer_class,
        optimizer_params: dict,
        loss_fn,
        device,
        image_dir,
        checkpoint_dir,
        checkpoint_interval=10,
        headless=False,
        logger=None,
    ):
        """
        Initializes the trainer for an Autoencoder from configuration files.
        """
        self.encoder_config_path = encoder_config_path
        self.decoder_config_path = decoder_config_path
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.device = device
        self.image_dir = image_dir
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.start_epoch = 0
        self.headless = headless

        # Initialize logger
        log_file = os.path.join(checkpoint_dir, "autoencoder_log.txt")
        self.logger = logger or Logger(
            source=f"{self.__class__.__name__}",
            log_file=log_file
        )

        # 1. Build Encoder and Decoder from config files using the factory
        self.logger.info(f"Building encoder from: {encoder_config_path}")
        self.encoder = create_model(encoder_config_path, device=device)
        self.logger.info(f"Building decoder from: {decoder_config_path}")
        self.decoder = create_model(decoder_config_path, device=device)

        # 2. Initialize Optimizer with parameters from both models
        self.optimizer = optimizer_class(
            itertools.chain(self.encoder.parameters(), self.decoder.parameters()),
            **optimizer_params
        )
        
        # 3. Setup directories
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 4. Save metadata and try to resume from checkpoints
        self.save_metadata()
        self._try_resume_latest()

        self.logger.info(f"Training Autoencoder pair.")
        self.logger.info(f"Device: {self.device}")


    def train(self, epochs, log_images_every=5):
        self.encoder.train()
        self.decoder.train()
        self.logger.info(f"Starting training from epoch {self.start_epoch + 1} for {epochs} epochs...")

        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            epoch_loss = 0.0
            iterator = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{self.start_epoch + epochs}", disable=self.headless)

            for batch in iterator:
                batch = batch.to(self.device)
                latent = self.encoder(batch)
                recon = self.decoder(latent)
                loss = self.loss_fn(recon, batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.dataloader)
            self.logger.info(f"[Train] Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")

            if self.image_dir and (epoch + 1) % log_images_every == 0:
                self._save_reconstruction_preview(batch, recon, epoch + 1)

            if self.checkpoint_dir and (epoch + 1) % self.checkpoint_interval == 0:
                self.save_model(epoch + 1)
        
        self.logger.info("✔ Training complete.")

    def _save_reconstruction_preview(self, batch, recon, epoch):
        # Detach tensors from graph before concatenating
        comparison = torch.cat([batch[:4].detach(), recon[:4].detach()])
        img_path = os.path.join(self.image_dir, f"recon_epoch_{epoch:03}.png")
        save_image(comparison.clamp(-1, 1) * 0.5 + 0.5, img_path)
        self.logger.debug(f"[Image] Saved reconstruction preview to {img_path}")

    def save_model(self, epoch):
        enc_path = os.path.join(self.checkpoint_dir, f"encoder_epoch{epoch:03}.pt")
        dec_path = os.path.join(self.checkpoint_dir, f"decoder_epoch{epoch:03}.pt")
        
        torch.save(self.encoder.state_dict(), enc_path)
        torch.save(self.decoder.state_dict(), dec_path)
        
        self.logger.info(f"[Checkpoint] Saved model at epoch {epoch}")

    def save_metadata(self):
        """Saves the model info and a copy of the training configs."""
        enc_meta_path = os.path.join(self.checkpoint_dir, "encoder_metadata.json")
        dec_meta_path = os.path.join(self.checkpoint_dir, "decoder_metadata.json")

        with open(enc_meta_path, 'w') as f:
            json.dump(self.encoder.info, f, indent=4)
        with open(dec_meta_path, 'w') as f:
            json.dump(self.decoder.info, f, indent=4)

        shutil.copy(self.encoder_config_path, os.path.join(self.checkpoint_dir, "encoder_config.yaml"))
        shutil.copy(self.decoder_config_path, os.path.join(self.checkpoint_dir, "decoder_config.yaml"))
        self.logger.debug(f"Saved model metadata and configs to {self.checkpoint_dir}")

    def _try_resume_latest(self):
        self.logger.info("Searching for latest autoencoder checkpoint...")
        pattern = os.path.join(self.checkpoint_dir, "encoder_epoch*.pt")
        files = glob.glob(pattern)
        
        if not files:
            self.logger.info("No checkpoint found, starting from scratch.")
            return

        def extract_epoch(path):
            match = re.search(r"epoch(\d+)", path)
            return int(match.group(1)) if match else -1

        latest_enc_file = sorted(files, key=extract_epoch)[-1]
        latest_epoch = extract_epoch(latest_enc_file)
        
        if latest_epoch <= 0:
            self.logger.warning(f"Could not parse epoch from {latest_enc_file}. Starting fresh.")
            return

        latest_dec_file = os.path.join(self.checkpoint_dir, f"decoder_epoch{latest_epoch:03}.pt")

        try:
            self.encoder.load_state_dict(torch.load(latest_enc_file, map_location=self.device))
            self.decoder.load_state_dict(torch.load(latest_dec_file, map_location=self.device))
            self.start_epoch = latest_epoch
            self.logger.info(f"✅ [AutoResume] Resumed training from epoch {latest_epoch}")
        except FileNotFoundError:
            self.logger.error(f"Checkpoint file mismatch for epoch {latest_epoch}. Cannot resume.")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from epoch {latest_epoch}: {e}")