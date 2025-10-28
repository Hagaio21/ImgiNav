import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.functional import structural_similarity_index_measure
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness


@torch.no_grad()
def compute_recon_metrics(model, dataloader, device, max_batches=10):
    """Compute average MSE and SSIM for a model with possibly unnormalized outputs."""
    mse_vals, ssim_vals = [], []
    count = 0

    # detect normalization mode
    use_sigmoid = getattr(model.decoder, "use_sigmoid", False)

    for batch in dataloader:
        if batch is None:
            continue
        imgs = batch["layout"].to(device)
        out = model(imgs)
        recon = out["recon"]

        # normalize both tensors to [0,1] if decoder not using sigmoid
        if not use_sigmoid:
            recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)
            imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8)
        else:
            recon = recon.clamp(0, 1)
            imgs = imgs.clamp(0, 1)

        mse = F.mse_loss(recon, imgs, reduction="none").mean(dim=[1, 2, 3])
        ssim = structural_similarity_index_measure(recon, imgs, data_range=1.0)

        mse_vals.append(mse.cpu())
        ssim_vals.append(ssim.cpu())

        count += 1
        if count >= max_batches:
            break

    mse_mean = torch.cat(mse_vals).mean().item() if mse_vals else None
    ssim_mean = torch.stack(ssim_vals).mean().item() if ssim_vals else None
    return {"mse_mean": mse_mean, "ssim_mean": ssim_mean}


@torch.no_grad()
def compute_distribution_metrics(latents):
    # latents: (N, C, H, W)
    N, C, H, W = latents.shape
    flat = latents.permute(1, 0, 2, 3).reshape(C, -1)
    mu_abs = flat.mean(dim=1).abs().mean().item()
    std_mean = flat.std(dim=1).mean().item()

    cov = torch.cov(flat)
    isotropy_ratio = torch.trace(cov) / cov.sum()
    return {
        "mean_abs_mu": mu_abs,
        "std_mean": std_mean,
        "isotropy_ratio": isotropy_ratio.item(),
    }


@torch.no_grad()
def compute_geometric_metrics(latents, inputs, max_samples=256):
    # latents: (N, C, H, W) -> (N, D)
    N = min(latents.shape[0], max_samples)
    lat_flat = latents[:N].reshape(N, -1).cpu().numpy()
    inp_flat = inputs[:N].reshape(N, -1).cpu().numpy()

    # compute pairwise distances (sample subset)
    d_lat = np.linalg.norm(lat_flat[:, None, :] - lat_flat[None, :, :], axis=-1)
    d_pix = np.linalg.norm(inp_flat[:, None, :] - inp_flat[None, :, :], axis=-1)

    corr, _ = spearmanr(d_lat.flatten(), d_pix.flatten())
    corr = float(corr) if not np.isnan(corr) else 0.0

    try:
        trust = float(trustworthiness(inp_flat, lat_flat, n_neighbors=5))
    except Exception:
        trust = 0.0

    return {"latent_pixel_corr": corr, "trustworthiness": trust}


@torch.no_grad()
def compute_information_metrics(latents):
    flat = latents.reshape(latents.shape[0], -1).cpu().numpy()
    pca = PCA(n_components=10)
    pca.fit(flat)
    evr = pca.explained_variance_ratio_
    return {
        "pca_var_ratio1": float(evr[0]),
        "pca_var_ratio10": float(evr[:10].sum()),
    }
