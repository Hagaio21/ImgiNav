# modules/alignment.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Alignment projection module
# ---------------------------

class AlignmentMLP(nn.Module):
    """Small MLP to project embeddings into a shared latent space."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or (input_dim + output_dim) // 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------
# Cosine alignment loss
# ---------------------------

class AlignmentLoss(nn.Module):
    """Cosine similarity loss between condition and layout projections."""
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, cond_emb, layout_emb):
        cond_emb = F.normalize(cond_emb, dim=-1)
        layout_emb = F.normalize(layout_emb, dim=-1)
        pos_sim = (cond_emb * layout_emb).sum(dim=-1)

        # optional negative term for contrast
        neg_sim = torch.mm(cond_emb, layout_emb.t())
        neg_mask = 1 - torch.eye(cond_emb.size(0), device=cond_emb.device)
        neg_sim = (neg_sim * neg_mask).max(dim=1)[0]

        loss = (1 - pos_sim) + F.relu(neg_sim - pos_sim + self.margin)
        return loss.mean()

class InfoNCELoss(nn.Module):
    """InfoNCE that allows mismatched input dims by projecting to shared length."""
    def __init__(self, temperature: float = 0.07, proj_dim: int = 512):
        super().__init__()
        self.temperature = temperature
        self.proj_dim = proj_dim

    def forward(self, cond_emb, layout_emb):
        B = cond_emb.size(0)

        # Flatten any spatial dims
        cond_emb = cond_emb.view(B, -1)
        layout_emb = layout_emb.view(B, -1)

        # Project both to the same temporary dimension
        if cond_emb.size(1) != layout_emb.size(1):
            # choose smaller dimension for both
            target_dim = min(cond_emb.size(1), layout_emb.size(1), self.proj_dim)
            cond_emb = F.adaptive_avg_pool1d(cond_emb.unsqueeze(1), target_dim).squeeze(1)
            layout_emb = F.adaptive_avg_pool1d(layout_emb.unsqueeze(1), target_dim).squeeze(1)

        # Normalize and compute logits
        cond_emb = F.normalize(cond_emb, dim=-1)
        layout_emb = F.normalize(layout_emb, dim=-1)
        logits = torch.mm(cond_emb, layout_emb.t()) / self.temperature

        labels = torch.arange(B, device=cond_emb.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_i2t + loss_t2i)


# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    B, d_g, d_p, d_l = 32, 384, 512, 256
    graph = torch.randn(B, d_g)
    pov = torch.randn(B, d_p)
    layout = torch.randn(B, d_l)

    align_graph = AlignmentMLP(d_g, d_l)
    align_pov = AlignmentMLP(d_p, d_l)
    align_loss = AlignmentLoss()

    proj_graph = align_graph(graph)
    proj_pov = align_pov(pov)

    loss_graph = align_loss(proj_graph, layout)
    loss_pov = align_loss(proj_pov, layout)

    print(f"Graph loss: {loss_graph.item():.4f} | POV loss: {loss_pov.item():.4f}")
