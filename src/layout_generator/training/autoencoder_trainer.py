import torch
import os
from utils.logger import Logger
import tqdm
from torchvision.utils import save_image
import json
import re
import glob

class AutoEncoderTrainer:
    def __init__(
        self,
        encoder,
        decoder,
        dataloader,
        optimizer,
        loss_fn,
        device,
        batch_size=4,
        image_dir=None,
        checkpoint_dir=None,
        checkpoint_interval=10,
        headless=False,
        logger=None,
        log_file=None
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.batch_size = batch_size

        self.image_dir = image_dir
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.start_epoch = 0
        self.headless = headless

        if log_file is None:
            log_file = os.path.join(checkpoint_dir or ".", f"{encoder.name}_log.txt")

        self.logger = logger or Logger(
            debug=False,
            info=True,
            log_file=log_file,
            source=f"{self.__class__.__name__}:{encoder.name}"
        )
        
        self.loss_log_path = None

        if self.image_dir:
            os.makedirs(self.image_dir, exist_ok=True)
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.save_metadata()
            self._try_resume_latest()
            self.loss_log_path = os.path.join(self.checkpoint_dir, f"{self.encoder.name}_loss_log.json")

        self.logger.info(f"Training model: {self.encoder.name}")
        self.logger.info(f"Device: {self.device} | Batch size: {self.batch_size}")

    def train(self, epochs, log_images_every=5):
        self.encoder.train()
        self.decoder.train()
        all_losses = []
        self.logger.info(f"Starting training from epoch {self.start_epoch + 1} for {epochs} epochs...")

        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            epoch_loss = 0.0

            iterator = self.dataloader
            if not self.headless:
                iterator = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{self.start_epoch + epochs}")

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
            all_losses.append({"epoch": epoch + 1, "loss": avg_loss})
            self.logger.info(f"[Train] Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")
            self.logger.log_metric("avg_loss", avg_loss, step=epoch + 1)

            if self.image_dir and (epoch + 1) % log_images_every == 0:
                self._save_reconstruction_preview(batch, recon, epoch + 1)

            if self.checkpoint_dir and (epoch + 1) % self.checkpoint_interval == 0:
                self.save_model(epoch + 1)

            # Save loss log
            if self.loss_log_path:
                with open(self.loss_log_path, 'w') as f:
                    json.dump(all_losses, f, indent=2)
        
        self.logger.info("✓ Training complete.")

    def _save_reconstruction_preview(self, batch, recon, epoch):
        comparison = torch.cat([batch[:4], recon[:4]])
        img_path = os.path.join(self.image_dir, f"epoch_{epoch:03}.png")
        save_image(comparison.clamp(-1, 1) * 0.5 + 0.5, img_path)
        self.logger.debug(f"[Image] Saved reconstruction preview to {img_path}")

    def save_model(self, epoch):
        name = self.encoder.name
        enc_path = os.path.join(self.checkpoint_dir, f"encoder_{name}_epoch{epoch:03}.pt")
        dec_path = os.path.join(self.checkpoint_dir, f"decoder_{name}_epoch{epoch:03}.pt")
        torch.save(self.encoder.state_dict(), enc_path)
        torch.save(self.decoder.state_dict(), dec_path)
        self.logger.info(f"[Checkpoint] Saved model at epoch {epoch}")
        self.logger.debug(f"Encoder saved to {enc_path}")
        self.logger.debug(f"Decoder saved to {dec_path}")

    def save_metadata(self):
        name = self.encoder.name
        encoder_info = self.encoder.info()
        decoder_info = self.decoder.info()

        enc_meta_path = os.path.join(self.checkpoint_dir, f"encoder_{name}_metadata.json")
        dec_meta_path = os.path.join(self.checkpoint_dir, f"decoder_{name}_metadata.json")

        with open(enc_meta_path, 'w') as f:
            json.dump(encoder_info, f, indent=4)
        with open(dec_meta_path, 'w') as f:
            json.dump(decoder_info, f, indent=4)

        self.logger.debug(f"Saved encoder metadata to {enc_meta_path}")
        self.logger.debug(f"Saved decoder metadata to {dec_meta_path}")

    def _try_resume_latest(self):
        name = self.encoder.name
        pattern = os.path.join(self.checkpoint_dir, f"encoder_{name}_epoch*.pt")
        files = glob(pattern)
        if not files:
            self.logger.info("No checkpoint found, starting from scratch.")
            return

        def extract_epoch(path):
            match = re.search(r"epoch(\d+)", path)
            return int(match.group(1)) if match else -1

        latest_file = sorted(files, key=extract_epoch)[-1]
        latest_epoch = extract_epoch(latest_file)
        
        if latest_epoch <= 0:
            self.logger.warning(f"Could not parse epoch from checkpoint file: {latest_file}")
            return

        enc_path = os.path.join(self.checkpoint_dir, f"encoder_{name}_epoch{latest_epoch:03}.pt")
        dec_path = os.path.join(self.checkpoint_dir, f"decoder_{name}_epoch{latest_epoch:03}.pt")

        try:
            self.encoder.load_state_dict(torch.load(enc_path, map_location=self.device))
            self.decoder.load_state_dict(torch.load(dec_path, map_location=self.device))
            self.start_epoch = latest_epoch
            self.logger.info(f"✅ [AutoResume] Resumed training from epoch {latest_epoch}")
        except FileNotFoundError:
            self.logger.error(f"Checkpoint file not found for epoch {latest_epoch}, cannot resume.")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from epoch {latest_epoch}: {e}")