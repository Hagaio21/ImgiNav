import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# Assumes you have installed your project in editable mode:
# pip install -e .
from layout_generator.data.datasets import RicoLayoutDataset, collate_fn
from layout_generator.models.autoencoder import Autoencoder
from layout_generator.utils.factories import get_model # Assuming you have a model factory


def visualize_reconstructions(originals, reconstructions, vocab, save_path):
    """Saves a plot comparing original and reconstructed layouts."""
    # This is a placeholder function. You'll need to implement the actual
    # logic to convert layout tensors back into drawable images.
    # For now, we'll just print shapes as a confirmation.
    print(f"Original batch shape: {originals.shape}")
    print(f"Reconstructed batch shape: {reconstructions.shape}")
    
    # Example of what a real implementation might look like:
    # fig, axes = plt.subplots(5, 2, figsize=(10, 20))
    # fig.suptitle("Original vs. Reconstructed Layouts")
    # for i in range(5):
    #     draw_layout(originals[i], vocab, ax=axes[i, 0], title="Original")
    #     draw_layout(reconstructions[i], vocab, ax=axes[i, 1], title="Reconstructed")
    # plt.savefig(save_path)
    # print(f"Saved reconstruction visualization to {save_path}")
    pass


def analyze_checkpoint(args):
    """
    Main function to analyze a single model checkpoint.
    """
    # 1. SETUP
    # =======================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint_path)
    config_path = Path(args.config_path)
    
    if not checkpoint_path.exists() or not config_path.exists():
        raise FileNotFoundError("Checkpoint or config file not found!")

    # Define project paths relative to this script's location
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "analysis" / "outputs" / checkpoint_path.stem
    data_dir = project_root / "data"

    # Create a unique output directory for this analysis run
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Analyzing checkpoint: {checkpoint_path.name}")
    print(f"Using config: {config_path.name}")
    print(f"Saving analysis results to: {output_dir}")

    # 2. LOAD CONFIG AND DATA
    # =======================================================================
    with open(config_path) as f:
        # Assuming your config is JSON. Change to yaml.safe_load if it's YAML.
        config = json.load(f)

    with open(data_dir / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    analysis_dataset = RicoLayoutDataset(
        data_dir / "rico_dataset", vocab, split="val", max_length=128
    )
    analysis_loader = DataLoader(
        analysis_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    # 3. INITIALIZE AND LOAD MODEL
    # =======================================================================
    # Use your factory to build the model from the config
    # This makes the script robust to changes in model architecture
    model_config = config.get("model", {})
    model_config['vocab_size'] = len(vocab) # Ensure vocab size is correct
    
    # You might need to adjust this part to match your actual factory function
    # model = get_model(model_config).to(device)
    model = Autoencoder(
        vocab_size=model_config['vocab_size'],
        d_model=model_config.get('d_model', 256),
        latent_dim=model_config.get('latent_dim', 16)
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # 4. RUN ANALYSIS
    # =======================================================================
    total_l2_distance = 0
    total_samples = 0
    
    with torch.no_grad():
        # Get one batch for visualization
        first_batch_originals, _ = next(iter(analysis_loader))
        first_batch_originals = first_batch_originals.to(device)
        first_batch_reconstructed, _, _ = model(first_batch_originals)
        
        visualize_reconstructions(
            first_batch_originals.cpu(),
            first_batch_reconstructed.cpu(),
            vocab,
            save_path=output_dir / "reconstruction_examples.png"
        )
        
        # Calculate L2 distance over the full dataset
        for src, _ in analysis_loader:
            src = src.to(device)
            reconstructed, _, _ = model(src)
            
            l2_dist = torch.norm(src.float() - reconstructed.float(), p=2, dim=1).mean()
            total_l2_distance += l2_dist.item() * src.size(0)
            total_samples += src.size(0)
            
    avg_l2 = total_l2_distance / total_samples
    print(f"\nAnalysis Complete:")
    print(f"  - Average L2 Reconstruction Error: {avg_l2:.4f}")

    # Save results to a file
    with open(output_dir / "results.json", "w") as f:
        json.dump({"average_l2_error": avg_l2}, f, indent=2)
    print(f"Saved numerical results to {output_dir / 'results.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Run analysis on a single model checkpoint."
    )
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        required=True, 
        help="Path to the model checkpoint (.pth) file."
    )
    parser.add_argument(
        "--config_path", 
        type=str, 
        required=True, 
        help="Path to the corresponding training config (.json or .yaml) file."
    )
    args = parser.parse_args()
    analyze_checkpoint(args)


if __name__ == "__main__":
    main()