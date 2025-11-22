"""
Training utility functions.
"""
import torch
import random
import numpy as np
import yaml
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.autoencoder import Autoencoder
from models.datasets.datasets import ManifestDataset
from models.losses.base_loss import LOSS_REGISTRY


def set_deterministic(seed: int = 42, strict_determinism: bool = False):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CUDNN deterministic settings (applies to most operations)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Strict deterministic algorithms (may warn for operations without deterministic impl)
    if strict_determinism:
        torch.use_deterministic_algorithms(True, warn_only=True)


class NumpySafeLoader(yaml.SafeLoader):
    """Custom YAML loader that handles numpy scalar types."""
    pass


def numpy_scalar_constructor(loader, node):
    """Convert numpy scalar tags to Python native types."""
    # For python/object/apply tags, the node contains a sequence with the function and args
    # numpy.core.multiarray.scalar is called with the value as argument
    try:
        # Construct the sequence which contains [numpy.core.multiarray.scalar, value]
        sequence = loader.construct_sequence(node)
        if len(sequence) >= 2:
            # The value is the second element (first is the function/class)
            value = sequence[1]
            # Convert numpy types to Python native types
            if isinstance(value, (np.integer, np.floating, np.ndarray)):
                return value.item() if hasattr(value, 'item') else float(value)
            return value
        return sequence[0] if sequence else None
    except Exception:
        # Fallback: try to construct as sequence and take first value
        try:
            sequence = loader.construct_sequence(node)
            return sequence[0] if sequence else None
        except Exception:
            return None


# Register the constructor for numpy scalar types
NumpySafeLoader.add_constructor(
    'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar',
    numpy_scalar_constructor
)


def load_config(config_path: Path):
    """Load experiment config from YAML file."""
    # Try loading with custom loader first, fallback to FullLoader if it fails
    try:
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=NumpySafeLoader)
    except (yaml.constructor.ConstructorError, yaml.YAMLError) as e:
        # If custom loader fails (e.g., other numpy types), try FullLoader
        print(f"Warning: Custom loader failed, trying FullLoader: {e}")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def ensure_weight_stats_exist(manifest_path: Path, column_name: str, output_dir: Path,
                              rare_threshold_percentile: float = 10.0,
                              min_samples_threshold: int = 50,
                              weighting_method: str = "inverse_frequency",
                              max_weight: float = None,
                              min_weight: float = 1.0,
                              filters: dict = None):
    """
    Ensure weight stats JSON exists for a column. If not, generate it automatically.
    
    Args:
        manifest_path: Path to manifest CSV
        column_name: Column name to analyze
        output_dir: Directory to save stats (will create subdirectory for column)
        rare_threshold_percentile: Percentile for rare class detection
        min_samples_threshold: Minimum samples threshold
        weighting_method: Weighting method
        max_weight: Max weight cap
        min_weight: Min weight
        filters: Optional filters dict (same format as dataset filters) to apply before computing weights.
                 This ensures weights are computed on the same filtered dataset that will be used for training.
    
    Returns:
        Path to stats JSON file
    """
    from analysis.analyze_column_distribution import analyze_column_distribution
    import pandas as pd
    
    output_dir = Path(output_dir)
    column_output_dir = output_dir / "weight_stats" / column_name
    
    # Create a filter signature for the stats filename to ensure different filters get different stats
    filter_suffix = ""
    if filters:
        # Create a simple hash/signature from filters
        filter_str = "_".join([f"{k}_{v}" for k, v in sorted(filters.items())])
        # Sanitize for filename (remove special chars, limit length)
        filter_str = "".join(c if c.isalnum() or c in "_-" else "_" for c in filter_str)[:50]
        filter_suffix = f"_{filter_str}"
    
    # First, check if stats file exists directly in experiment directory (user-provided)
    stats_path_experiment = output_dir / f"{column_name}_distribution_stats.json"
    if stats_path_experiment.exists():
        print(f"Using existing weight stats from experiment directory: {stats_path_experiment}")
        return stats_path_experiment
    
    # Then check in the subdirectory (auto-generated location)
    stats_path = column_output_dir / f"{column_name}_distribution_stats{filter_suffix}.json"
    
    # Check if stats already exist
    if stats_path.exists():
        print(f"Using existing weight stats: {stats_path}")
        return stats_path
    
    # Load manifest and apply filters if provided
    df = pd.read_csv(manifest_path, low_memory=False)
    original_size = len(df)
    
    if filters:
        print(f"Applying filters before computing weights (original size: {len(df)})...")
        # Apply same filtering logic as ManifestDataset._apply_filters
        for key, value in filters.items():
            if "__lt" in key:
                col = key.replace("__lt", "")
                df = df[df[col] < value]
            elif "__gt" in key:
                col = key.replace("__gt", "")
                df = df[df[col] > value]
            elif "__le" in key:
                col = key.replace("__le", "")
                df = df[df[col] <= value]
            elif "__ge" in key:
                col = key.replace("__ge", "")
                df = df[df[col] >= value]
            elif "__ne" in key:
                col = key.replace("__ne", "")
                df = df[df[col] != value]
            else:
                if isinstance(value, (list, tuple, set)):
                    df = df[df[key].isin(value)]
                else:
                    df = df[df[key] == value]
        df = df.reset_index(drop=True)
        print(f"After filtering: {len(df)} samples (removed {original_size - len(df)})")
    
    # Save filtered manifest temporarily for analysis
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        temp_manifest = Path(tmp_file.name)
        df.to_csv(temp_manifest, index=False)
    
    # Generate stats automatically
    print(f"\n{'='*60}")
    print(f"Generating weight stats for column: {column_name}")
    print(f"{'='*60}")
    print(f"Manifest: {manifest_path}")
    if filters:
        print(f"Using filtered manifest: {len(df)} samples (filters: {filters})")
    print(f"Output: {column_output_dir}")
    print(f"Weighting method: {weighting_method}")
    if max_weight:
        print(f"Max weight cap: {max_weight}")
    print()
    
    try:
        analyze_column_distribution(
            manifest_path=temp_manifest,  # Use filtered manifest
            column_name=column_name,
            output_dir=column_output_dir,
            rare_threshold_percentile=rare_threshold_percentile,
            min_samples_threshold=min_samples_threshold,
            weighting_method=weighting_method,
            max_weight=max_weight,
            min_weight=min_weight
        )
        # Rename the generated stats file to include filter suffix
        generated_stats = column_output_dir / f"{column_name}_distribution_stats.json"
        if generated_stats.exists() and filter_suffix:
            generated_stats.rename(stats_path)
        print(f"\nWeight stats generated: {stats_path}")
        return stats_path
    except Exception as e:
        print(f"Error generating weight stats: {e}")
        raise
    finally:
        # Clean up temporary manifest
        if temp_manifest.exists():
            temp_manifest.unlink()


def build_model(config):
    """Build autoencoder from config."""
    ae_cfg = config["autoencoder"].copy() if isinstance(config["autoencoder"], dict) else config["autoencoder"]
    # Pass save_path from experiment config so model can write statistics
    exp_cfg = config.get("experiment", {})
    if isinstance(ae_cfg, dict) and exp_cfg.get("save_path"):
        ae_cfg["save_path"] = exp_cfg["save_path"]
    model = Autoencoder.from_config(ae_cfg)
    return model




def build_dataset(config):
    """Build dataset from config."""
    ds_cfg = config["dataset"]
    dataset = ManifestDataset(**ds_cfg)
    return dataset


def build_loss(config):
    """Build loss function from config.
    
    The loss config should have a "type" key specifying the loss class.
    For multiple losses, use CompositeLoss with a "losses" list.
    """
    from models.losses.base_loss import LOSS_REGISTRY
    
    loss_cfg = config["training"]["loss"]
    
    if "type" not in loss_cfg:
        raise ValueError(f"Loss config missing 'type' key: {loss_cfg}")
    
    loss_type = loss_cfg["type"]
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(LOSS_REGISTRY.keys())}")
    
    loss_fn = LOSS_REGISTRY[loss_type].from_config(loss_cfg)
    return loss_fn


def build_optimizer(model, config):
    """Build optimizer from config using model's parameter_groups for trainable params."""
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"].get("weight_decay", 0.0)
    optimizer_type = config["training"].get("optimizer", "AdamW").lower()
    
    # Get parameter groups from model (respects frozen params and per-module LRs)
    param_groups = model.parameter_groups()
    
    # If no groups returned (all frozen or no per-module LRs), use trainable params with base LR
    if not param_groups:
        trainable_params = list(model.trainable_parameters())
        if not trainable_params:
            raise ValueError("No trainable parameters found in model!")
        param_groups = [{"params": trainable_params, "lr": lr}]
    else:
        # Set default LR for groups that don't have one specified
        for group in param_groups:
            if "lr" not in group:
                group["lr"] = lr
    
    # Add weight_decay to all groups
    for group in param_groups:
        group["weight_decay"] = weight_decay
    
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(param_groups)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(param_groups)
    elif optimizer_type == "sgd":
        for group in param_groups:
            group["momentum"] = 0.9
        optimizer = torch.optim.SGD(param_groups)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    return optimizer


def get_device(config):
    """Get device from config or default."""
    device = config.get("training", {}).get("device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def to_device(device):
    """Convert device string to device object if needed.
    
    Args:
        device: Device string (e.g., "cuda", "cpu") or torch.device object
    
    Returns:
        torch.device object
    """
    return torch.device(device) if isinstance(device, str) else device


def move_batch_to_device(batch, device, non_blocking=None):
    """Move batch to device with optional non_blocking transfer.
    
    Args:
        batch: Dictionary of tensors and other data
        device: Device to move to (string or torch.device)
        non_blocking: Whether to use non_blocking transfer. If None, auto-detects based on device type.
    
    Returns:
        Batch with tensors moved to device
    """
    device_obj = to_device(device)
    if non_blocking is None:
        non_blocking = device_obj.type == "cuda" if hasattr(device_obj, 'type') else False
    
    return {
        k: v.to(device_obj, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def split_dataset(dataset, config):
    """Split dataset into train and validation sets based on config.
    
    Args:
        dataset: Dataset to split
        config: Training config dict
    
    Returns:
        (train_dataset, val_dataset) tuple. val_dataset may be None if train_split >= 1.0
    """
    train_split = config.get("train_split", 0.8)
    split_seed = config.get("split_seed", 42)
    
    if train_split < 1.0:
        train_dataset, val_dataset = dataset.split(train_split=train_split, random_seed=split_seed)
    else:
        train_dataset = dataset
        val_dataset = None
    
    return train_dataset, val_dataset


def create_grad_scaler(use_amp, device):
    """Create gradient scaler for mixed precision training.
    
    Args:
        use_amp: Whether to use mixed precision
        device: Device (string or torch.device)
    
    Returns:
        GradScaler instance or None
    """
    if not use_amp:
        return None
    
    device_obj = to_device(device)
    if device_obj.type == "cuda":
        # Use the newer torch.amp.GradScaler API
        return torch.amp.GradScaler('cuda')
    return None


def save_metrics_csv(training_history, metrics_csv_path, phase_metrics_path=None):
    """Save training history to CSV file(s).
    
    Args:
        training_history: List of dict records (one per epoch)
        metrics_csv_path: Path to main metrics CSV
        phase_metrics_path: Optional path to phase metrics CSV
    """
    import pandas as pd
    
    df = pd.DataFrame(training_history)
    df.to_csv(metrics_csv_path, index=False)
    
    if phase_metrics_path:
        df.to_csv(phase_metrics_path, index=False)


def build_scheduler(optimizer, config, last_epoch=-1, max_steps=None):
    """Build learning rate scheduler from config.
    
    Args:
        optimizer: Optimizer to attach scheduler to
        config: Training config dict
        last_epoch: Last epoch number (for resuming training). Default -1 means start from beginning.
        max_steps: Optional max steps for steps-based training. If provided, overrides epoch-based calculation.
    """
    training_cfg = config.get("training", {})
    
    scheduler_type = training_cfg.get("scheduler", {}).get("type", None)
    if scheduler_type is None:
        return None
    
    # Determine T_max: use max_steps if provided (steps-based), otherwise use epochs (epoch-based)
    if max_steps is not None:
        T_max = max_steps
    else:
        T_max = config.get("experiment", {}).get("epochs_target", training_cfg.get("epochs", 100))
    
    # When resuming (last_epoch >= 0), ensure optimizer param_groups have initial_lr
    # This is required by PyTorch schedulers when resuming
    if last_epoch >= 0:
        for param_group in optimizer.param_groups:
            if "initial_lr" not in param_group:
                # Set initial_lr to current LR (from config) so scheduler can resume correctly
                param_group["initial_lr"] = param_group.get("lr", 0.001)
    
    if scheduler_type.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, last_epoch=last_epoch)
    elif scheduler_type.lower() == "linear":
        start_factor = training_cfg.get("scheduler", {}).get("start_factor", 1.0)
        end_factor = training_cfg.get("scheduler", {}).get("end_factor", 0.0)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=T_max, last_epoch=last_epoch)
    elif scheduler_type.lower() == "step":
        step_size = training_cfg.get("scheduler", {}).get("step_size", T_max // 3)
        gamma = training_cfg.get("scheduler", {}).get("gamma", 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)
    else:
        print(f"  Warning: Unknown scheduler type '{scheduler_type}', not using scheduler")
        return None
    
    return scheduler

