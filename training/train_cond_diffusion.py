"""
Training script for conditioned latent diffusion model.
Refactored version using modular utilities.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse

from training_utils import (
    load_experiment_config, save_experiment_config, setup_experiment_dir,
    save_checkpoint, load_checkpoint, save_training_stats, init_training_stats,
    update_training_stats, train_epoch_conditioned, validate_conditioned,
    save_split_files,
    setup_experiment, load_core_models, load_data, create_mixer,
    create_optimizer_scheduler, prepare_fixed_samples, resume_training, diagnostic_condition_tests
)
from sampling_utils import generate_samples_conditioned 

def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Train Conditioned Latent Diffusion Model')
    parser.add_argument("--exp_config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # --- Setup ---
    config, exp_dir, device = setup_experiment(args.exp_config, args.resume)

    # --- Load Models ---
    unet, scheduler, autoencoder, diffusion_model = load_core_models(config, device)

    # --- Load Data ---
    train_loader, val_loader, val_dataset = load_data(config, exp_dir)

    # --- Create Mixer ---
    mixer = create_mixer(config, device)

    # --- Optimizer & Scheduler ---
    optimizer, scheduler_lr = create_optimizer_scheduler(config, unet, mixer)

    # --- Fixed Samples ---
    fixed_samples = prepare_fixed_samples(config, val_dataset, exp_dir)

    # --- Resume Logic ---
    start_epoch, best_loss, training_stats = resume_training(
        args, exp_dir, unet, mixer, optimizer, scheduler_lr
    )

    # --- Training Loop ---
    print(f"\nStarting training from epoch {start_epoch + 1} to {config['training']['num_epochs']}")
    print(f"Batch size: {config['training']['batch_size']}, Learning rate: {optimizer.param_groups[0]['lr']:.6f}") # Get initial LR
    print(f"Batches per epoch: {len(train_loader)}")
    print("=" * 60)

    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        # --- Train ---
        train_loss, corr_pov, corr_graph, cond_std_pov, cond_std_graph, dropout_pov, dropout_graph = train_epoch_conditioned(
        unet, scheduler, mixer, train_loader, optimizer, device, epoch,
        config=config,
        cfg_dropout_prob=config["training"]["cfg"]["dropout_prob"],
        loss_fn=loss_fn
    )


        # --- Validate ---
        val_loss, val_corr_pov, val_corr_graph = validate_conditioned(
            unet, scheduler, mixer, val_loader, device, config=config
        )

        # --- Step LR ---
        scheduler_lr.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # --- Diagnostics ---
        diag_results = diagnostic_condition_tests(
            unet, scheduler, mixer, diffusion_model, val_loader, device, config
        )

        # --- Update Stats ---
        training_stats = update_training_stats(
            training_stats, epoch, train_loss, val_loss, current_lr,
            train_corr_pov=corr_pov, train_corr_graph=corr_graph,
            val_corr_pov=val_corr_pov, val_corr_graph=val_corr_graph,
            cond_std_pov=cond_std_pov, cond_std_graph=cond_std_graph,
            dropout_ratio_pov=dropout_pov, dropout_ratio_graph=dropout_graph,
            **diag_results
        )

        # --- Logging ---
        print(f"Epoch {epoch + 1}/{config['training']['num_epochs']} - "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
            f"Corr POV: {corr_pov:.3f}/{val_corr_pov:.3f} | "
            f"Corr Graph: {corr_graph:.3f}/{val_corr_graph:.3f} | "
            f"LR: {current_lr:.6f}", flush=True)

        # --- Checkpointing ---
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

        save_training_stats(exp_dir, training_stats)

        if (epoch + 1) % config["training"]["sample_every"] == 0:
            print(f"Generating samples at epoch {epoch + 1}...", flush=True)
            generate_samples_conditioned(
                diffusion_model, mixer, fixed_samples, exp_dir, epoch, device, config
            )

    print(f"\nTraining complete! Best validation loss: {best_loss:.6f}")
    print(f"Checkpoints saved in: {exp_dir / 'checkpoints'}")
    print(f"Samples saved in: {exp_dir / 'samples'}")


if __name__ == "__main__":
    main()