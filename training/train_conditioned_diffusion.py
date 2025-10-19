"""
Training script for conditioned latent diffusion model.
Refactored version using modular utilities.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse
import random
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from modules.unet import UNet
from modules.scheduler import *
from modules.autoencoder import AutoEncoder
from modules.unified_dataset import UnifiedLayoutDataset, collate_fn
from modules.condition_mixer import LinearConcatMixer, NonLinearConcatMixer
from modules.diffusion import LatentDiffusion


# Import utilities
from training_utils import (
    load_experiment_config, save_experiment_config, setup_experiment_dir,
    save_checkpoint, load_checkpoint, save_training_stats, init_training_stats,
    update_training_stats, train_epoch_conditioned, validate_conditioned,
    save_split_files
)
from sampling_utils import generate_samples_conditioned



def main():
    parser = argparse.ArgumentParser(description='Train Conditioned Latent Diffusion Model')
    parser.add_argument("--exp_config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--room_manifest", required=True, help="Path to room manifest")
    parser.add_argument("--scene_manifest", required=True, help="Path to scene manifest")
    parser.add_argument("--pov_type", default=None, help="POV type (e.g., 'rendered', 'semantic')")
    parser.add_argument("--mixer_type", choices=["LinearConcat", "NonLinearConcat"], 
                    default=None, help="Override mixer type from config")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    args = parser.parse_args()

    # Load and setup experiment
    config = load_experiment_config(args.exp_config)
    exp_dir = setup_experiment_dir(
        config["experiment"]["exp_dir"], 
        config["unet"]["config_path"], 
        args.resume
    )
    
    if not args.resume:
        save_experiment_config(config, exp_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Load Models
    
    print(f"Loading U-Net from {config['unet']['config_path']}", flush=True)
    # Get C from the [C, H, W] shape in your main config
    latent_channels = config["latent_shape"][0] 
    
    # Pass it to the from_config method
    unet = UNet.from_config(
        config["unet"]["config_path"],
        latent_channels=latent_channels
    ).to(device)
    
    print(f"Loading {config['scheduler']['type']} with {config['scheduler']['num_steps']} steps", flush=True)
    scheduler_class = globals()[config["scheduler"]["type"]]
    scheduler = scheduler_class(num_steps=config["scheduler"]["num_steps"]).to(device)
    
    print(f"Loading Autoencoder from {config['autoencoder']['config_path']}", flush=True)
    autoencoder = AutoEncoder.from_config(config["autoencoder"]["config_path"]).to(device)
    if config["autoencoder"]["checkpoint_path"]:
        autoencoder.load_state_dict(
            torch.load(config["autoencoder"]["checkpoint_path"], map_location=device, weights_only=False)
        )
    autoencoder.eval()

    # Create LatentDiffusion wrapper
    latent_shape = tuple(config["latent_shape"])
    diffusion_model = LatentDiffusion(
        unet=unet,
        scheduler=scheduler,
        autoencoder=autoencoder,
        latent_shape=latent_shape
    ).to(device)

    # Load Dataset
    
    print(f"Loading unified dataset...", flush=True)
    dataset = UnifiedLayoutDataset(
        args.room_manifest, 
        args.scene_manifest, 
        use_embeddings=True,
        pov_type=args.pov_type
    )
            
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}", flush=True)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=True, 
        num_workers=4,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=False, 
        num_workers=4,
        collate_fn=collate_fn
    )

    # Save split info
    train_df = dataset.df.iloc[train_dataset.indices]
    val_df = dataset.df.iloc[val_dataset.indices]
    save_split_files(exp_dir, train_df, val_df)

    # Create Mixer

    latent_hw = tuple(config["latent_shape"][-2:])  # (H, W)

    # Get UNet conditioning channels from config
    with open(config["unet"]["config_path"], 'r') as f:
        unet_cfg = yaml.safe_load(f)
    cond_channels = unet_cfg["unet"]["cond_channels"]

    # Get embedding dimensions from config
    pov_dim = config["mixer"]["pov_dim"]
    graph_dim = config["mixer"]["graph_dim"]
    
    print(f"Input dimensions - POV: {pov_dim}, Graph: {graph_dim}", flush=True)

    # --- Get Mixer Configuration ---
    mixer_config = config.get("mixer", {})
    mixer_type_name = mixer_config.get("type", "LinearConcatMixer") # Default type
    if args.mixer_type is not None:
        mixer_type_name = args.mixer_type # Override with CLI arg

    print(f"Creating {mixer_type_name} (out_channels={cond_channels}, target_size={latent_hw})...", flush=True)

    # --- Map type name string to actual class object ---
    # (Make sure these classes are imported at the top)
    mixer_class_map = {
        "LinearConcatMixer": LinearConcatMixer,
        "NonLinearConcatMixer": NonLinearConcatMixer,
    }
    mixer_class = mixer_class_map.get(mixer_type_name)

    if mixer_class is None:
        print(f"ERROR: Unknown mixer type '{mixer_type_name}'. Available: {list(mixer_class_map.keys())}", flush=True)
        print("Defaulting to LinearConcatMixer.")
        mixer_class = LinearConcatMixer
        mixer_type_name = "LinearConcatMixer" # Update name to reflect default

    # --- Prepare Arguments ---
    # Core arguments required by all mixers in this setup
    mixer_args = {
        "out_channels": cond_channels,
        "target_size": latent_hw,
        "pov_channels": pov_dim,
        "graph_channels": graph_dim
    }

    # Add arguments specific to certain mixer types
    if mixer_class == NonLinearConcatMixer:
        # NonLinearConcatMixer expects 'hidden_dim_mlp'
        mlp_hidden_dim = mixer_config.get("mlp_hidden_dim", None) # Read from config
        if mlp_hidden_dim is not None:
            mixer_args["hidden_dim_mlp"] = mlp_hidden_dim
            print(f"  Using MLP hidden dimension: {mlp_hidden_dim}", flush=True)
        else:
            print(f"  Using default MLP hidden dimension.", flush=True)
    # Add elif blocks here for other mixers if they have specific args

    # --- Instantiate the mixer ---
    try:
        mixer = mixer_class(**mixer_args).to(device)
        print(f"Mixer '{mixer_type_name}' instantiated successfully.", flush=True)
        print(f"  Mixer parameters are trainable.", flush=True)

    except TypeError as e:
        print(f"ERROR: Failed to instantiate mixer '{mixer_type_name}'.")
        print(f"  Provided arguments: {mixer_args}")
        print(f"  Error message: {e}")
        print("  Please check your config file and the mixer class __init__ signature.")
        raise e # Re-raise error


    # Optimizer and Scheduler
    
    # Combine parameters from both models to train them jointly
    params_to_train = list(unet.parameters()) + list(mixer.parameters())
    optimizer = Adam(params_to_train, lr=config["training"]["learning_rate"])
    scheduler_lr = CosineAnnealingLR(optimizer, T_max=config["training"]["num_epochs"])
    print(f"Optimizer training {len(params_to_train)} parameters (UNet + Mixer).", flush=True)

    # Fixed samples for visual tracking
    
    num_fixed = min(8, len(val_dataset))
    fixed_indices = random.sample(range(len(val_dataset)), num_fixed)
    fixed_samples = [val_dataset[i] for i in fixed_indices]
    torch.save(fixed_indices, exp_dir / "fixed_indices.pt")

    # Resume from checkpoint
    
    start_epoch = 0
    best_loss = float("inf")
    training_stats = init_training_stats()
    
    latest_checkpoint = exp_dir / 'checkpoints' / 'latest.pt'
    if args.resume and latest_checkpoint.exists():
            start_epoch, best_loss, training_stats = load_checkpoint(
                latest_checkpoint,
                models_dict={'unet': unet, 'mixer': mixer},
                optimizer=optimizer,
                scheduler_lr=scheduler_lr
            )
            start_epoch += 1

    # Training Loop
    
    print(f"\nStarting training from epoch {start_epoch+1} to {config['training']['num_epochs']}")
    print(f"Batch size: {config['training']['batch_size']}, Learning rate: {config['training']['learning_rate']}")
    print(f"Batches per epoch: {len(train_loader)}")
    print("="*60)
    
    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        train_loss, corr_pov, corr_graph, corr_mix, cond_std_pov, cond_std_graph, dropout_pov, dropout_graph = train_epoch_conditioned(
            unet, scheduler, mixer, train_loader, optimizer, device, epoch,
            config=config,
            cfg_dropout_prob=config["training"]["cfg"]["dropout_prob"]
        )



        val_loss, val_corr_pov, val_corr_graph, val_corr_mix = validate_conditioned(
            unet, scheduler, mixer, val_loader, device,
            config=config)

        scheduler_lr.step()
        current_lr = optimizer.param_groups[0]['lr']

        training_stats = update_training_stats(
                        training_stats, epoch, train_loss, val_loss, current_lr,
                        corr_pov=corr_pov, corr_graph=corr_graph, corr_mix=corr_mix,
                        cond_std_pov=cond_std_pov, cond_std_graph=cond_std_graph,
                        dropout_ratio_pov=dropout_pov, dropout_ratio_graph=dropout_graph
                    )


        print(f"Epoch {epoch+1}/{config['training']['num_epochs']} - "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.6f}",
            flush=True)

        # Save checkpoint
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss

        save_periodic = (epoch + 1) % config["training"].get("periodic_checkpoint_every", 10) == 0
        save_checkpoint(
            exp_dir, epoch,
            state_dict={
                'unet': unet.state_dict(),
                'mixer': mixer.state_dict(),
                'opt': optimizer.state_dict(),
                'sched': scheduler_lr.state_dict()
            },
            training_stats=training_stats,
            val_loss=val_loss,
            best_loss=best_loss,
            is_best=is_best,
            save_periodic=save_periodic
        )

        # Save training stats
        save_training_stats(exp_dir, training_stats)

        # Generate samples
        if (epoch + 1) % config["training"]["sample_every"] == 0:
            print(f"Generating samples at epoch {epoch+1}...")
            generate_samples_conditioned(
                diffusion_model, mixer, fixed_samples, exp_dir, epoch, device, config
            )

    print(f"\nTraining complete! Best validation loss: {best_loss:.6f}")
    print(f"Checkpoints saved in: {exp_dir / 'checkpoints'}")
    print(f"Samples saved in: {exp_dir / 'samples'}")


if __name__ == "__main__":
    main()