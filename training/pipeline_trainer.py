import torch
from torch.optim import AdamW
from tqdm import tqdm
from torchvision.utils import save_image
import os
from PIL import Image
from stage8_create_graph_embeddings import graph2text


class PipelineTrainer:
    def __init__(self,
                 pipeline,
                 optimizer=None,
                 loss_fn=None,
                 epochs=10,
                 lr=1e-4,
                 weight_decay=0.0,
                 grad_clip=None,
                 log_interval=100,
                 eval_interval=1000,
                 sample_interval=2000,
                 ckpt_dir=None,
                 output_dir=None,
                 logger=None,
                 mixed_precision=False,
                 ema_decay=None,
                 cond_dropout_pov=0.1,
                 cond_dropout_graph=0.1,
                 cond_dropout_both=0.0):
        self.pipeline = pipeline.to(pipeline.device)
        self.device = pipeline.device
        self.logger = logger

        self.loss_fn = loss_fn or torch.nn.MSELoss()
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.sample_interval = sample_interval
        self.mixed_precision = mixed_precision
        self.ema_decay = ema_decay

        self.cond_dropout_pov = cond_dropout_pov
        self.cond_dropout_graph = cond_dropout_graph
        self.cond_dropout_both = cond_dropout_both

        self.optimizer = optimizer or AdamW(
            list(pipeline.unet.parameters()) + list(pipeline.mixer.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.global_step = 0
        self.ckpt_dir = ckpt_dir
        self.output_dir = output_dir or "samples"
        os.makedirs(self.output_dir, exist_ok=True)
        if self.ckpt_dir:
            os.makedirs(self.ckpt_dir, exist_ok=True)

        self.ema_unet = None
        if ema_decay:
            import copy
            self.ema_unet = copy.deepcopy(pipeline.unet).eval()

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        # fixed noise for consistent samples
        self.fixed_noise = None

    def to(self, device):
        self.pipeline.to(device)
        return self

    def train(self, mode=True):
        self.pipeline.train(mode)
        return self

    def fit(self, train_loader, val_loader=None):
        self.train(True)
        for epoch in range(self.epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)
            for batch in pbar:
                self.optimizer.zero_grad(set_to_none=True)
                layout, cond_pov, cond_graph = batch

                # ---- modality control (from config) ----
                modality = getattr(self, "use_modalities", "both")
                if modality == "pov_only":
                    cond_graph = None
                elif modality == "graph_only":
                    cond_pov = None
                elif modality == "none":
                    cond_pov = None
                    cond_graph = None

                # ---- stochastic dropout within selected modalities ----
                if torch.rand(1).item() < self.cond_dropout_both:
                    cond_pov = None
                    cond_graph = None
                else:
                    if torch.rand(1).item() < self.cond_dropout_pov:
                        cond_pov = None
                    if torch.rand(1).item() < self.cond_dropout_graph:
                        cond_graph = None

                batch = (layout, cond_pov, cond_graph)
                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    loss = self.pipeline.training_step(batch, self.loss_fn, step=self.global_step)

                self.scaler.scale(loss).backward()
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.pipeline.unet.parameters()) + list(self.pipeline.mixer.parameters()),
                        self.grad_clip,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.global_step += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                if self.ema_unet and self.ema_decay:
                    self.update_ema()

                if self.logger and self.global_step % self.log_interval == 0:
                    self.logger.log({"loss": loss.item(), "epoch": epoch, "step": self.global_step})

                if val_loader and self.global_step % self.eval_interval == 0:
                    self.evaluate(val_loader, step=self.global_step)
                    self.train(True)

                if self.global_step % self.sample_interval == 0:
                    example_batch = next(iter(train_loader))
                    self.sample_and_save(self.global_step, example_batch)

            if self.ckpt_dir:
                self.save_checkpoint(f"{self.ckpt_dir}/epoch_{epoch+1}.pt")

    @torch.no_grad()
    def evaluate(self, val_loader, step=None):
        self.train(False)
        batch = next(iter(val_loader))
        metrics = self.pipeline.evaluate(batch, step=step)
        if self.logger and step is not None:
            self.logger.log({"val_step": step, **metrics}, step=step)
        return metrics

    @torch.no_grad()
    def sample_and_save(self, step, batch, sample_num=4):
        self.pipeline.eval()
        layout, pov_img, graph_path = batch

        # initialize fixed noise once
        if self.fixed_noise is None:
            latent_shape = (sample_num, self.pipeline.scheduler.latent_channels,
                            self.pipeline.scheduler.latent_base,
                            self.pipeline.scheduler.latent_base)
            self.fixed_noise = torch.randn(latent_shape, device=self.device)

        # conditioned samples
        cond_dir = os.path.join(self.output_dir, "samples", "condition", str(sample_num))
        os.makedirs(cond_dir, exist_ok=True)

        for i in range(min(sample_num, len(layout))):
            item_dir = os.path.join(cond_dir, f"{i}")
            os.makedirs(item_dir, exist_ok=True)

            pov_path = os.path.join(item_dir, "pov.png")
            target_path = os.path.join(item_dir, "target.png")
            graph_txt = os.path.join(item_dir, "graph.txt")

            if isinstance(pov_img, torch.Tensor):
                save_image(pov_img[i], pov_path, normalize=True, value_range=(0, 1))
            elif isinstance(pov_img[i], Image.Image):
                pov_img[i].save(pov_path)

            save_image(layout[i], target_path, normalize=True, value_range=(0, 1))
            with open(graph_txt, "w", encoding="utf-8") as f:
                f.write(graph2text(graph_path[i]))

            cond_sample = self.pipeline.sample(
                batch_size=1,
                image=True,
                noise=self.fixed_noise[i].unsqueeze(0),
                pov=pov_img[i:i+1],
                graph_path=graph_path[i]
            )
            save_image(cond_sample, os.path.join(item_dir, f"generated_sample_{step}.png"),
                       normalize=True, value_range=(0, 1))

        # unconditioned samples
        uncond_dir = os.path.join(self.output_dir, "samples", "unconditioned")
        os.makedirs(uncond_dir, exist_ok=True)
        uncond_samples = self.pipeline.sample(
            batch_size=sample_num, image=True, noise=self.fixed_noise
        )
        save_image(uncond_samples,
                   os.path.join(uncond_dir, f"generated_sample_{step}.png"),
                   nrow=2, normalize=True, value_range=(0, 1))
        print(f"[Step {step}] Saved conditioned and unconditioned samples.")

    def save_checkpoint(self, path):
        state = {
            "unet": self.pipeline.unet.state_dict(),
            "mixer": self.pipeline.mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,
        }
        torch.save(state, path)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device)
        self.pipeline.unet.load_state_dict(state["unet"])
        self.pipeline.mixer.load_state_dict(state["mixer"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.global_step = state["step"]
        print(f"Checkpoint loaded from {path} at step {self.global_step}")

    def update_ema(self):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_unet.parameters(), self.pipeline.unet.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
