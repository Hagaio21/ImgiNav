#!/usr/bin/env python3

import argparse
import sys
import yaml
from pathlib import Path
import torch
from torchvision.utils import save_image
import numpy as np
from PIL import Image
try:
    import imageio
except ImportError:
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise ImportError("imageio is required. Install with: pip install imageio")

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.diffusion import LatentDiffusion
from visualization.lifting_utils import lift_layout, plot_point_cloud_3d_to_image


def main():
    parser = argparse.ArgumentParser(
        description="Generate diffusion samples and create 3D GIF on point cloud"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/diffusion_3d.gif",
        help="Output path for GIF"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ddim", "ddpm"],
        default="ddim",
        help="Sampling mode"
    )
    parser.add_argument(
        "--zmap",
        type=str,
        required=True,
        help="Path to zmap.json for 3D lifting"
    )
    parser.add_argument(
        "--point-density",
        type=float,
        default=1.0,
        help="Point density for 3D point cloud (0.0-1.0)"
    )
    parser.add_argument(
        "--height-samples",
        type=int,
        default=10,
        help="Number of height samples per pixel"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="FPS for output GIF"
    )
    
    args = parser.parse_args()
    
    config_path = "config/inference_config.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        inference_cfg = yaml.safe_load(f)
    
    diff_config = inference_cfg["diffusion"]["config"]
    diff_checkpoint = inference_cfg["diffusion"]["checkpoint"]
    ae_config = inference_cfg["autoencoder"]["config"]
    ae_checkpoint = inference_cfg["autoencoder"]["checkpoint"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with open(diff_config, "r", encoding="utf-8") as f:
        diffusion_cfg = yaml.safe_load(f)
    
    if "model" in diffusion_cfg and "diffusion" in diffusion_cfg["model"]:
        diffusion_cfg = diffusion_cfg["model"]["diffusion"]
    
    diffusion_cfg["autoencoder"] = {"config": ae_config, "checkpoint": ae_checkpoint}
    
    diffusion = LatentDiffusion.from_config(diffusion_cfg, device=device)
    
    state = torch.load(diff_checkpoint, map_location=device)
    loaded_state = state.get("state_dict", state.get("model", state))
    if loaded_state and list(loaded_state.keys())[0].startswith('module.'):
        loaded_state = {k[7:]: v for k, v in loaded_state.items()}
    diffusion.backbone.load_state_dict(loaded_state, strict=False)
    
    diffusion.eval()
    
    method = args.mode
    if method == "ddim":
        num_steps = 50
    else:
        num_steps = diffusion.scheduler.num_steps
    
    batch_size = 1
    
    print(f"[Sampling] Generating with {method.upper()}, {num_steps} steps")
    print(f"[3D] Will create GIF with zmap: {args.zmap}")
    
    with torch.no_grad():
        samples, history = diffusion.sample(
            batch_size=batch_size,
            image=False,
            cond=None,
            num_steps=num_steps,
            method=method,
            eta=0.0,
            device=device,
            guidance_scale=1.0,
            return_full_history=True,
            verbose=True
        )
        
        latents_history = history.get("latents", [])
        print(f"\n[History] Processing {len(latents_history)} intermediate steps...")
        
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        # Generate both room scale and scene scale GIFs
        output_base = Path(args.output).stem
        output_dir = Path(args.output).parent
        output_room = output_dir / f"{output_base}_room.gif"
        output_scene = output_dir / f"{output_base}_scene.gif"
        
        for scale_name, scale_type in [("room", "room"), ("scene", "scene")]:
            if scale_type == "room":
                print(f"\n[Processing {scale_name.upper()} scale (3m x 3m)")
            else:
                print(f"\n[Processing {scale_name.upper()} scale (30m x 30m)")
            
            frame_files = []
            
            # First, compute final point cloud to get fixed axis limits
            print(f"[{scale_name.upper()}] Computing final point cloud for fixed axis limits...")
            final_latent = latents_history[-1]
            final_rgb, _ = diffusion.autoencoder.decode(final_latent, from_latent=True)
            final_rgb = final_rgb[0].cpu().numpy()
            final_rgb = (final_rgb * 255).astype(np.uint8)
            final_rgb = np.transpose(final_rgb, (1, 2, 0))
            
            final_layout_img = Image.fromarray(final_rgb)
            temp_final_path = frames_dir / f"temp_final_layout_{scale_name}.png"
            final_layout_img.save(temp_final_path)
            
            final_points = lift_layout(
                layout_path=temp_final_path,
                zmap_path=args.zmap,
                point_density=args.point_density,
                height_samples=args.height_samples,
                force_scale=scale_type
            )
            
            # Calculate fixed axis limits from final point cloud
            if final_points.shape[0] > 0:
                x_final = final_points[:, 0]
                y_final = final_points[:, 1]
                z_final = final_points[:, 2]
                max_range = np.array([x_final.max()-x_final.min(), y_final.max()-y_final.min(), z_final.max()-z_final.min()]).max() / 2.0
                mid_x = (x_final.max()+x_final.min()) * 0.5
                mid_y = (y_final.max()+y_final.min()) * 0.5
                mid_z = (z_final.max()+z_final.min()) * 0.5
                axis_limits = (
                    (mid_x - max_range, mid_x + max_range),
                    (mid_y - max_range, mid_y + max_range),
                    (mid_z - max_range, mid_z + max_range)
                )
            else:
                axis_limits = None
            
            temp_final_path.unlink()
            
            for step_idx, latent in enumerate(latents_history):
                print(f"[{scale_name.upper()}] Step {step_idx}/{len(latents_history)-1} Decoding and lifting to 3D...")
                
                rgb_out, _ = diffusion.autoencoder.decode(latent, from_latent=True)
                rgb_out = rgb_out[0].cpu().numpy()
                
                rgb_out = (rgb_out * 255).astype(np.uint8)
                rgb_out = np.transpose(rgb_out, (1, 2, 0))
                
                layout_img = Image.fromarray(rgb_out)
                
                temp_layout_path = frames_dir / f"layout_step_{step_idx:04d}_{scale_name}.png"
                layout_img.save(temp_layout_path)
                
                lifted_points = lift_layout(
                    layout_path=temp_layout_path,
                    zmap_path=args.zmap,
                    point_density=args.point_density,
                    height_samples=args.height_samples,
                    force_scale=scale_type
                )
                
                if lifted_points.shape[0] > 0:
                    frame_path = frames_dir / f"frame_{step_idx:04d}_{scale_name}.png"
                    plot_point_cloud_3d_to_image(
                        points=lifted_points,
                        output_path=frame_path,
                        title=f"Diffusion Step {step_idx}/{len(latents_history)-1} ({scale_name.upper()})",
                        axis_limits=axis_limits,
                        layout_image=rgb_out
                    )
                    frame_files.append(frame_path)
                else:
                    print(f"  Warning: No points generated for step {step_idx}")
            
            print(f"\n[{scale_name.upper()}] Creating GIF from {len(frame_files)} frames...")
            
            frames = []
            for frame_path in sorted(frame_files):
                frames.append(imageio.imread(frame_path))
            
            output_path = output_room if scale_type == "room" else output_scene
            imageio.mimsave(output_path, frames, fps=args.fps)
            print(f"[{scale_name.upper()}] GIF saved to: {output_path}")
        
        print(f"\n[Complete] Both GIFs generated:")
        print(f"  Room scale (3m x 3m): {output_room}")
        print(f"  Scene scale (30m x 30m): {output_scene}")


if __name__ == "__main__":
    main()

