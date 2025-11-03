"""Test that noise schedulers properly destroy data at final timestep."""
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from models.components.scheduler import SCHEDULER_REGISTRY


def test_scheduler_destroys_data():
    """Verify that all schedulers destroy data (alpha_bar_T ≈ 0) at final timestep."""
    num_steps = 1000
    
    for scheduler_name, SchedulerClass in SCHEDULER_REGISTRY.items():
        scheduler = SchedulerClass(num_steps=num_steps)
        
        # Get final timestep alpha_bar (index is num_steps - 1)
        final_alpha_bar = scheduler.alpha_bars[-1].item()
        data_coeff = final_alpha_bar ** 0.5  # sqrt(alpha_bar) is the data coefficient
        
        print(f"\n{scheduler_name}:")
        print(f"  Final alpha_bar: {final_alpha_bar:.6e}")
        print(f"  Data coefficient: {data_coeff:.6e} ({data_coeff * 100:.4f}%)")
        
        # For proper noise destruction, alpha_bar_T should be very small (< 1e-3)
        # This means < 0.1% of original signal remains
        assert final_alpha_bar < 1e-3, (
            f"{scheduler_name}: Final alpha_bar ({final_alpha_bar:.6e}) is too large. "
            f"Data destruction incomplete!"
        )
        
        # Verify noise addition at final timestep
        x0 = torch.randn(2, 4, 8, 8)
        noise = torch.randn_like(x0)
        t_final = torch.tensor([num_steps - 1] * 2)
        
        x_t = scheduler.add_noise(x0, noise, t_final)
        
        # At final timestep, x_t should be mostly noise
        # Check that it's close to noise (within small tolerance)
        noise_ratio = (x_t - noise).abs().mean() / noise.abs().mean()
        print(f"  Noise ratio (lower is better): {noise_ratio:.6f}")
        
        # x_t should be very close to pure noise
        assert noise_ratio < 0.1, (
            f"{scheduler_name}: Noise addition incomplete. "
            f"x_t differs from noise by {noise_ratio * 100:.2f}%"
        )


if __name__ == "__main__":
    test_scheduler_destroys_data()
    print("\n✓ All schedulers properly destroy data at final timestep!")

