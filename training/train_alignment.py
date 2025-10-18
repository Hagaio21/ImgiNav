#!/usr/bin/env python3
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from modules.alignment import AlignmentMLP, InfoNCELoss
from modules.alignment_datasets import POVEmbeddingDataset, GraphEmbeddingDataset


# ---------------------------------------------------------------
# Training and Validation
# ---------------------------------------------------------------

def train_epoch(pov_loader, graph_loader, align_pov, align_graph, align_layout, loss_fn, opt, device):
    align_pov.train()
    align_graph.train()
    align_layout.train()

    total_loss, total_pov, total_graph = 0, 0, 0
    n_pov, n_graph = 0, 0

    for batch in tqdm(pov_loader, desc="Train POV"):
        pov, layout = batch["pov"].to(device), batch["layout"].to(device)
        layout = layout.view(layout.size(0), -1)  # flatten
        proj_layout = align_layout(layout)
        proj_pov = align_pov(pov)

        loss_pov = loss_fn(proj_pov, proj_layout)

        opt.zero_grad()
        loss_pov.backward()
        opt.step()

        total_pov += loss_pov.item()
        n_pov += 1

    for batch in tqdm(graph_loader, desc="Train Graph"):
        graph, layout = batch["graph"].to(device), batch["layout"].to(device)
        layout = layout.view(layout.size(0), -1)
        proj_layout = align_layout(layout)
        proj_graph = align_graph(graph)

        loss_graph = loss_fn(proj_graph, proj_layout)

        opt.zero_grad()
        loss_graph.backward()
        opt.step()

        total_graph += loss_graph.item()
        n_graph += 1

    total_loss = (total_pov + total_graph) / max(n_pov + n_graph, 1)
    return total_loss, total_pov / max(n_pov, 1), total_graph / max(n_graph, 1)


@torch.no_grad()
def validate_epoch(pov_loader, graph_loader, align_pov, align_graph, align_layout, loss_fn, device):
    align_pov.eval()
    align_graph.eval()
    align_layout.eval()

    total_loss, total_pov, total_graph = 0, 0, 0
    n_pov, n_graph = 0, 0

    for batch in tqdm(pov_loader, desc="Val POV"):
        pov, layout = batch["pov"].to(device), batch["layout"].to(device)
        layout = layout.view(layout.size(0), -1)
        proj_layout = align_layout(layout)
        proj_pov = align_pov(pov)
        loss_pov = loss_fn(proj_pov, proj_layout)
        total_pov += loss_pov.item()
        n_pov += 1

    for batch in tqdm(graph_loader, desc="Val Graph"):
        graph, layout = batch["graph"].to(device), batch["layout"].to(device)
        layout = layout.view(layout.size(0), -1)
        proj_layout = align_layout(layout)
        proj_graph = align_graph(graph)
        loss_graph = loss_fn(proj_graph, proj_layout)
        total_graph += loss_graph.item()
        n_graph += 1

    total_loss = (total_pov + total_graph) / max(n_pov + n_graph, 1)
    return total_loss, total_pov / max(n_pov, 1), total_graph / max(n_graph, 1)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--room_manifest", type=str, required=True)
    parser.add_argument("--scene_manifest", type=str, required=True)
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # load datasets
    pov_ds = POVEmbeddingDataset(args.room_manifest, subsample=args.subsample)
    graph_ds = GraphEmbeddingDataset([args.room_manifest, args.scene_manifest], subsample=args.subsample)

    def split_dataset(ds):
        n_val = int(0.05 * len(ds))
        n_train = len(ds) - n_val
        return torch.utils.data.random_split(ds, [n_train, n_val])

    pov_train, pov_val = split_dataset(pov_ds)
    graph_train, graph_val = split_dataset(graph_ds)

    pov_loader = DataLoader(pov_train, batch_size=args.batch_size, shuffle=True)
    pov_val_loader = DataLoader(pov_val, batch_size=args.batch_size, shuffle=False)
    graph_loader = DataLoader(graph_train, batch_size=args.batch_size, shuffle=True)
    graph_val_loader = DataLoader(graph_val, batch_size=args.batch_size, shuffle=False)

    # infer dimensions
    sample_pov = pov_ds[0]
    sample_graph = graph_ds[0]

    d_pov = len(sample_pov["pov"])
    d_graph = len(sample_graph["graph"])
    d_layout = sample_graph["layout"].numel()

    shared_dim = 512
    align_pov = AlignmentMLP(d_pov, shared_dim).to(device)
    align_graph = AlignmentMLP(d_graph, shared_dim).to(device)
    align_layout = AlignmentMLP(d_layout, shared_dim).to(device)

    loss_fn = InfoNCELoss(temperature=0.07)
    opt = torch.optim.Adam(
        list(align_pov.parameters()) +
        list(align_graph.parameters()) +
        list(align_layout.parameters()),
        lr=args.lr
    )

    stats = {"epoch": [], "train": [], "val": [], "train_pov": [], "val_pov": [], "train_graph": [], "val_graph": []}
    best_val = float("inf")

    for epoch in range(args.epochs):
        train_loss, train_pov, train_graph = train_epoch(pov_loader, graph_loader, align_pov, align_graph, align_layout, loss_fn, opt, device)
        val_loss, val_pov, val_graph = validate_epoch(pov_val_loader, graph_val_loader, align_pov, align_graph, align_layout, loss_fn, device)

        print(f"Epoch {epoch+1}/{args.epochs} | Train {train_loss:.4f} | Val {val_loss:.4f}", flush=True)

        stats["epoch"].append(epoch + 1)
        stats["train"].append(train_loss)
        stats["val"].append(val_loss)
        stats["train_pov"].append(train_pov)
        stats["val_pov"].append(val_pov)
        stats["train_graph"].append(train_graph)
        stats["val_graph"].append(val_graph)

        # save checkpoints
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "align_pov": align_pov.state_dict(),
                "align_graph": align_graph.state_dict(),
                "align_layout": align_layout.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss
            }, exp_dir / "best.pt")

        torch.save({
            "align_pov": align_pov.state_dict(),
            "align_graph": align_graph.state_dict(),
            "align_layout": align_layout.state_dict(),
            "epoch": epoch
        }, exp_dir / "latest.pt")

    # plot curves
    plt.figure(figsize=(8, 5))
    plt.plot(stats["epoch"], stats["train"], label="Train total")
    plt.plot(stats["epoch"], stats["val"], label="Val total")
    plt.plot(stats["epoch"], stats["train_pov"], "--", label="Train POV")
    plt.plot(stats["epoch"], stats["val_pov"], "--", label="Val POV")
    plt.plot(stats["epoch"], stats["train_graph"], ":", label="Train Graph")
    plt.plot(stats["epoch"], stats["val_graph"], ":", label="Val Graph")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Alignment Pretraining Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(exp_dir / "alignment_curves.png", dpi=150, bbox_inches="tight")

    with open(exp_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
