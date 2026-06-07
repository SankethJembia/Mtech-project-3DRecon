import argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree

def compute_metrics(pred_mesh_path, gt_mesh_path, num_samples=100000):
    print(f"Loading predicted mesh: {pred_mesh_path}")
    pred_mesh = trimesh.load(pred_mesh_path)
    
    print(f"Loading ground truth mesh: {gt_mesh_path}")
    gt_mesh = trimesh.load(gt_mesh_path)

    # 1. Sample points uniformly from the surface of both meshes
    print(f"Sampling {num_samples} points from both surfaces for unbiased evaluation...")
    pred_pts, _ = trimesh.sample.sample_surface(pred_mesh, num_samples)
    gt_pts, _ = trimesh.sample.sample_surface(gt_mesh, num_samples)

    # 2. Build KD-Trees for O(log N) nearest-neighbor lookups
    pred_tree = cKDTree(pred_pts)
    gt_tree = cKDTree(gt_pts)

    # 3. Calculate Point-to-Point Distances
    # Find the distance from every GT point to the nearest Predicted point
    dists_gt_to_pred, _ = pred_tree.query(gt_pts)
    
    # Find the distance from every Predicted point to the nearest GT point
    dists_pred_to_gt, _ = gt_tree.query(pred_pts)

    # --- Chamfer Distance (CD) ---
    # The average point-wise distance between the two surfaces.
    cd = (np.mean(dists_gt_to_pred) + np.mean(dists_pred_to_gt)) / 2.0

    # --- Hausdorff Distance (HD) ---
    # The maximum deviation (worst-case error) between the two surfaces.
    hd = max(np.max(dists_gt_to_pred), np.max(dists_pred_to_gt))

    # --- 95th Percentile Hausdorff Distance (HD95) ---
    # Captures extreme deviations while ignoring the top 5% of outliers (e.g., stray noise).
    all_dists = np.concatenate([dists_gt_to_pred, dists_pred_to_gt])
    hd95 = np.percentile(all_dists, 95)

    print("\n" + "="*30)
    print("      MEDGS EVALUATION      ")
    print("="*30)
    print(f"Chamfer Distance (CD)   : {cd:.5f}")
    print(f"Hausdorff Distance (HD) : {hd:.5f}")
    print(f"95th Percentile (HD95)  : {hd95:.5f}")
    print("="*30)
    
    # Lower values indicate better performance.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Mesh Reconstruction Accuracy")
    parser.add_argument("--pred", type=str, required=True, help="Predicted mesh (.obj)")
    parser.add_argument("--gt", type=str, required=True, help="Ground Truth mesh (.obj)")
    parser.add_argument("--samples", type=int, default=100000, help="Number of surface points to sample")
    args = parser.parse_args()
    
    compute_metrics(args.pred, args.gt, args.samples)