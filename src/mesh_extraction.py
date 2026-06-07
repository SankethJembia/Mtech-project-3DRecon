import torch
import numpy as np
from skimage.measure import marching_cubes
import trimesh

def extract_mesh_from_gaussians(model, camera, bg_color, zooms, resolution_z=128, threshold=0.5, output_path="output/liver.obj"):
    print(f"\n[Mesh Extraction] Rendering {resolution_z} dense 2D slices...")
    
    H, W = int(camera.image_height), int(camera.image_width)
    volume = np.zeros((resolution_z, H, W), dtype=np.float32)
    
    with torch.no_grad():
        for i in range(resolution_z):
            t_val = i / (resolution_z - 1.0)
            pred_image, _ = model.render(camera, t_val, bg_color)
            volume[i, :, :] = pred_image[0].cpu().numpy()

    print("[Mesh Extraction] Stack complete. Running Marching Cubes...")
    try:
        verts, faces, normals, values = marching_cubes(volume, level=threshold)
        
        # Scale X and Y coordinates to match image bounds
        verts[:, 0] = verts[:, 0] / resolution_z * W 
        verts[:, 1] = verts[:, 1] / resolution_z * H
        
        # --- NEW: Un-squash the Z-axis using real-world MRI dimensions ---
        z_spacing = zooms[0] # physical depth between slices
        x_spacing = zooms[1] # physical width of pixel
        stretch_factor = z_spacing / x_spacing
        
        print(f"[Mesh Extraction] Applying Z-axis stretch factor: {stretch_factor:.4f}")
        verts[:, 2] = verts[:, 2] * stretch_factor
        
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        mesh.export(output_path)
        print(f"[Mesh Extraction] Success! Saved physically scaled mesh to {output_path}")
        
    except ValueError as e:
        print(f"[Error] Marching Cubes failed: {e}")