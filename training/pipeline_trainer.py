import torch
from torch.optim import AdamW
from tqdm import tqdm
from torchvision.utils import save_image
import os
from PIL import Image
from stage8_create_graph_embeddings import graph2text
import torch.nn.functional as F
import matplotlib.pyplot as plt


def compute_condition_correlation(pred, conds):
    result = {}
    flat_pred = pred.flatten(1)
    for name, c in zip(["pov", "graph"], conds):
        if c is None:
            continue
        flat_cond = c.flatten(1)
        sim = F.cosine_similarity(flat_pred, flat_cond, dim=1)
        result[name] = sim.mean().item()
    return result


def safe_grad_norm(model):
    norms = []
    for p in model.parameters():
        if p.grad is not None:
            norms.append(p.grad.norm() ** 2)
    if not norms:
        return 0.0
    return torch.sqrt(sum(norms)).item()


class PipelineTrainer:
    def __init__(self,
                 pipeline,
                 sample_loader=None,
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
                 cond_dropout_both=0.0,
                 use_modalities="both"):
        self.pipeline = pipeline.to(pipeline.device)
        self.device = pipeline.device
        self.logger = logger
        self.sample_loader = sample_loader

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
        self.use_modalities = use_modalities

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

        # tracking for plots
        self.train_losses = []
        self.train_steps = []
        self.val_losses = []
        self.val_steps = []
        self.corr_pov_history = []
        self.corr_graph_history = []
        self.corr_steps = []

    def to(self, device):
        self.pipeline.to(device)
        return self

    def train(self, mode=True):
        self.pipeline.train(mode)
        return self

    def training_step(self, batch):
        """
        Training expects pre-computed embeddings for efficiency.
        """
        layout = batch["layout"]
        cond_pov_emb = batch["pov"]
        cond_graph_emb = batch["graph"]
        
        z = self.pipeline.encode_layout(layout)
        t = torch.randint(0, self.pipeline.scheduler.num_steps, (z.size(0),), device=self.device)
        noise = torch.randn_like(z)
        z_noisy = self.pipeline.scheduler.add_noise(z, noise, t)

        cond = self.pipeline.mixer([cond_pov_emb, cond_graph_emb])
        noise_pred = self.pipeline.unet(z_noisy, t, cond)

        loss = self.loss_fn(noise_pred, noise)

        if self.logger is not None:
            log = {"loss": loss.item(), "timestep": t.float().mean().item()}

            # correlation and cosine
            corr = compute_condition_correlation(noise_pred.detach(), [cond_pov_emb, cond_graph_emb])
            for k, v in corr.items():
                log[f"corr_{k}"] = v
            cos = F.cosine_similarity(noise_pred.flatten(1), noise.flatten(1)).mean().item()
            log["cosine_pred_true"] = cos

            # SNR
            snr = (z_noisy.var(dim=(1,2,3)) / (noise_pred - noise).var(dim=(1,2,3))).mean().item()
            log["snr"] = snr

            # grad norm
            log["grad_norm"] = safe_grad_norm(self.pipeline.unet)

            self.logger.log(log, step=self.global_step)

        return loss

    def plot_metrics(self):
        """Generate and save metric plots."""
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Plot 1: Loss per step (train and val)
        if self.train_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(self.train_steps, self.train_losses, label='Train Loss', alpha=0.7)
            if self.val_losses:
                plt.plot(self.val_steps, self.val_losses, label='Val Loss', marker='o', alpha=0.7)
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title('Loss per Step')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'loss_per_step.png'), dpi=150, bbox_inches='tight')
            plt.close()

        # Plot 2: Correlation per step (pov and graph)
        if self.corr_pov_history or self.corr_graph_history:
            plt.figure(figsize=(10, 6))
            if self.corr_pov_history:
                plt.plot(self.corr_steps, self.corr_pov_history, label='corr_pov', alpha=0.7)
            if self.corr_graph_history:
                plt.plot(self.corr_steps, self.corr_graph_history, label='corr_graph', alpha=0.7)
            plt.xlabel('Step')
            plt.ylabel('Correlation')
            plt.title('Condition Correlation per Step')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'correlation_per_step.png'), dpi=150, bbox_inches='tight')
            plt.close()

    def fit(self, train_loader, val_loader=None):
        self.train(True)
        for epoch in range(self.epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)
            for batch in pbar:
                self.optimizer.zero_grad(set_to_none=True)
                
                # Unpack dict batch
                layout = batch["layout"]
                cond_pov = batch["pov"]
                cond_graph = batch["graph"]

                # Modality control (from config)
                if self.use_modalities == "pov_only":
                    cond_graph = None
                elif self.use_modalities == "graph_only":
                    cond_pov = None
                elif self.use_modalities == "none":
                    cond_pov = None
                    cond_graph = None

                # Stochastic dropout within selected modalities
                if torch.rand(1).item() < self.cond_dropout_both:
                    cond_pov = None
                    cond_graph = None
                else:
                    if torch.rand(1).item() < self.cond_dropout_pov:
                        cond_pov = None
                    if torch.rand(1).item() < self.cond_dropout_graph:
                        cond_graph = None

                # Reconstruct batch dict for training_step
                batch_dict = {
                    "layout": layout,
                    "pov": cond_pov,
                    "graph": cond_graph
                }
                
                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    loss = self.training_step(batch_dict)

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
                    
                    # Track train loss
                    self.train_losses.append(loss.item())
                    self.train_steps.append(self.global_step)
                    
                    # Track correlations from last training step
                    with torch.no_grad():
                        z = self.pipeline.encode_layout(layout)
                        t = torch.randint(0, self.pipeline.scheduler.num_steps, (z.size(0),), device=self.device)
                        noise = torch.randn_like(z)
                        z_noisy = self.pipeline.scheduler.add_noise(z, noise, t)
                        cond = self.pipeline.mixer([cond_pov, cond_graph])
                        noise_pred = self.pipeline.unet(z_noisy, t, cond)
                        corr = compute_condition_correlation(noise_pred, [cond_pov, cond_graph])
                        
                        if "pov" in corr:
                            self.corr_pov_history.append(corr["pov"])
                        if "graph" in corr:
                            self.corr_graph_history.append(corr["graph"])
                        if corr:
                            self.corr_steps.append(self.global_step)
                    
                    # Generate plots
                    self.plot_metrics()

                if val_loader and self.global_step % self.eval_interval == 0:
                    val_metrics = self.evaluate(val_loader, step=self.global_step)
                    
                    # Track val loss if available
                    if "loss" in val_metrics:
                        self.val_losses.append(val_metrics["loss"])
                        self.val_steps.append(self.global_step)
                    
                    # Generate plots after evaluation
                    self.plot_metrics()
                    
                    self.train(True)

                if self.global_step % self.sample_interval == 0:
                    self.sample_and_save(self.global_step)

            if self.ckpt_dir:
                self.save_checkpoint(f"{self.ckpt_dir}/epoch_{epoch+1}.pt")

    @torch.no_grad()
    def evaluate(self, val_loader, step=None):
        self.train(False)
        batch = next(iter(val_loader))
        
        # Compute validation loss
        layout = batch["layout"]
        cond_pov_emb = batch["pov"]
        cond_graph_emb = batch["graph"]
        
        z = self.pipeline.encode_layout(layout)
        t = torch.randint(0, self.pipeline.scheduler.num_steps, (z.size(0),), device=self.device)
        noise = torch.randn_like(z)
        z_noisy = self.pipeline.scheduler.add_noise(z, noise, t)
        cond = self.pipeline.mixer([cond_pov_emb, cond_graph_emb])
        noise_pred = self.pipeline.unet(z_noisy, t, cond)
        val_loss = self.loss_fn(noise_pred, noise)
        
        metrics = self.pipeline.evaluate(batch, step=step)
        metrics["loss"] = val_loss.item()
        
        if self.logger and step is not None:
            self.logger.log({"val_step": step, **metrics}, step=step)
        return metrics

    @torch.no_grad()
    def sample_and_save(self, step, sample_num=4):
        if self.sample_loader is None:
            print(f"[Warning] No sample_loader provided, skipping sampling at step {step}")
            return
            
        self.pipeline.eval()
        
        # Get batch from sample loader (raw data, not embeddings)
        batch = next(iter(self.sample_loader))
        layout = batch["layout"]
        pov_img = batch["pov"]
        graph_path = batch["graph"]

        # Initialize fixed noise once
        if self.fixed_noise is None:
            latent_shape = (sample_num, self.pipeline.scheduler.latent_channels,
                            self.pipeline.scheduler.latent_base,
                            self.pipeline.scheduler.latent_base)
            self.fixed_noise = torch.randn(latent_shape, device=self.device)

        # Use step in directory name to avoid overwriting
        cond_dir = os.path.join(self.output_dir, "samples", "conditioned", f"step_{step}")
        os.makedirs(cond_dir, exist_ok=True)

        for i in range(min(sample_num, len(layout))):
            item_dir = os.path.join(cond_dir, f"sample_{i}")
            os.makedirs(item_dir, exist_ok=True)

            pov_path = os.path.join(item_dir, "pov.png")
            target_path = os.path.join(item_dir, "target.png")
            graph_txt = os.path.join(item_dir, "graph.txt")
            generated_path = os.path.join(item_dir, "generated.png")

            # Save inputs
            if isinstance(pov_img, torch.Tensor):
                save_image(pov_img[i], pov_path, normalize=True, value_range=(0, 1))
            elif isinstance(pov_img[i], Image.Image):
                pov_img[i].save(pov_path)

            save_image(layout[i], target_path, normalize=True, value_range=(0, 1))
            with open(graph_txt, "w", encoding="utf-8") as f:
                if isinstance(graph_path[i], str):
                    f.write(graph2text(graph_path[i]))
                else:
                    f.write(str(graph_path[i]))

            # Generate and save
            cond_sample = self.pipeline.sample(
                batch_size=1,
                pov_raw=pov_img[i] if not isinstance(pov_img, torch.Tensor) else pov_img[i].cpu(),
                graph_raw=graph_path[i] if isinstance(graph_path[i], str) else None,
                image=True,
                noise=self.fixed_noise[i].unsqueeze(0)
            )
            save_image(cond_sample, generated_path, normalize=True, value_range=(0, 1))

        # Unconditioned samples
        uncond_dir = os.path.join(self.output_dir, "samples", "unconditioned")
        os.makedirs(uncond_dir, exist_ok=True)
        uncond_samples = self.pipeline.sample(
            batch_size=sample_num, image=True, noise=self.fixed_noise
        )
        save_image(uncond_samples,
                   os.path.join(uncond_dir, f"step_{step}.png"),
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