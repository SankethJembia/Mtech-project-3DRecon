import argparse
import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
import os
import gc

from src.dataset import MedicalVolumeDataset
from src.gaussian_model import MedGSModel, SimpleCamera
from src.losses import medgs_loss
from src.mesh_extraction import extract_mesh_from_gaussians

def calculate_psnr(img1, img2):
    """Calculates PSNR between two image tensors."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(100.0, device=img1.device)
    max_pixel = 1.0 # Images are normalized to [0, 1]
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

def train(args):
    print(f"Loading data from {args.input_path}...")
    dataset = MedicalVolumeDataset(args.input_path)
    
    H, W = dataset.H, dataset.W
    camera = SimpleCamera(H, W)
    bg_color = torch.zeros(3, device="cuda")

    initial_points = 30_000 
    model = MedGSModel(num_points=initial_points, poly_degree=2).cuda()
    model.create_from_pcd(num_pts=initial_points)

    lrs = [
        {'params': [model._xyz], 'lr': 0.00016},
        {'params': [model._features_dc], 'lr': 0.0025},
        {'params': [model._opacity], 'lr': 0.05},
        {'params': [model._scaling], 'lr': 0.005},
        {'params': [model._rotation], 'lr': 0.001},
        {'params': [model._time_params], 'lr': 0.0001}
    ]
    optimizer = torch.optim.Adam(lrs, eps=1e-15)

    # ... previous setup code ...
    # --- Create visualization folders ---
    vis_dir = os.path.join(os.path.dirname(args.output_path), "training_vis")
    pred_dir = os.path.join(vis_dir, "predicted_slices")
    gt_dir = os.path.join(vis_dir, "ground_truth_slices")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    print(f"Starting optimization for {args.iterations} iterations...")
    iter_bar = tqdm(range(args.iterations), desc="Training MedGS")
    
    ema_loss = 0.0
    ema_psnr = 0.0 # Initialize PSNR tracker
    
    for iteration in iter_bar:
        optimizer.zero_grad(set_to_none=True)
        
        idx = np.random.randint(0, len(dataset))
        gt_image_tensor, t_value = dataset[idx]
        gt_image = gt_image_tensor.cuda().repeat(3, 1, 1)
        
        try:
            pred_image, _ = model.render(camera, t_value, bg_color)
        except Exception as e:
            torch.cuda.empty_cache() 
            continue
            
        # C. Calculate Loss
        loss = medgs_loss(pred_image, gt_image, model._scaling, len(dataset))
        
        # Calculate PSNR purely for tracking (no gradients)
        with torch.no_grad():
            current_psnr = calculate_psnr(pred_image, gt_image)
            
        # D. Backpropagation
        try:
            loss.backward()
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                continue
            else:
                raise e

        optimizer.step()
        
        # E. Progress Tracking & Image Saving
        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            ema_psnr = 0.4 * current_psnr.item() + 0.6 * ema_psnr 
            
            # Update terminal every 100 steps
            if iteration % 100 == 0:
                iter_bar.set_postfix({
                    "Loss": f"{ema_loss:.5f}", 
                    "PSNR": f"{ema_psnr:.2f}dB"
                })
                torch.cuda.empty_cache()
                gc.collect()
                
            # Save visual snapshots every 250 steps
            if iteration % 250 == 0:
                filename = f"iter_{iteration:04d}_depth_{t_value:.2f}.png"
                save_image(pred_image, os.path.join(pred_dir, filename))
                save_image(gt_image, os.path.join(gt_dir, filename))

    # Export PLY (The one that worked)
    ply_path = args.output_path.replace(".obj", ".ply")
    print(f"\nOptimization complete. Exporting assets...")
    model.save_ply(ply_path, num_time_samples=dataset.D)
    
    # --- NEW: Pass dataset.zooms ---
    extract_mesh_from_gaussians(
        model=model, 
        camera=camera, 
        bg_color=bg_color, 
        zooms=dataset.zooms, # Passing the physical dimensions here
        resolution_z=dataset.D, 
        threshold=0.5, 
        output_path=args.output_path
    )
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output/liver.obj")
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    train(args)