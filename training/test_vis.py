import torch
from pathlib import Path
from sampling_utils import _save_grid_with_titles_matplotlib

# Create fake data
num_samples = 8
channels = 3
height = 64
width = 64

output_dir = Path("test_outputs")
output_dir.mkdir(exist_ok=True)

# Create 3 rows of fake images
row1 = [torch.rand(channels, height, width) for _ in range(num_samples)]
row2 = [torch.rand(channels, height, width) for _ in range(num_samples)]
row3 = [torch.rand(channels, height, width) for _ in range(num_samples)]

# Test RGB images
_save_grid_with_titles_matplotlib(
    rows_of_tensors=[row1, row2, row3],
    row_titles=["Unconditioned", "Conditioned", "Target"],
    save_path=output_dir / "test_comparison_images.png",
    is_grayscale=False
)

# Test grayscale images
row1_gray = [torch.rand(1, height, width) for _ in range(num_samples)]
row2_gray = [torch.rand(1, height, width) for _ in range(num_samples)]
row3_gray = [torch.rand(1, height, width) for _ in range(num_samples)]

_save_grid_with_titles_matplotlib(
    rows_of_tensors=[row1_gray, row2_gray, row3_gray],
    row_titles=["|Cond - Tgt|", "|Uncond - Tgt|", "|Cond - Uncond|"],
    save_path=output_dir / "test_comparison_diffs.png",
    is_grayscale=True
)

print(f"✓ Test images saved to {output_dir}")
print("✓ Adjust VIZ_* parameters at top of sampling_utils.py")