import torch
import torch.nn as nn
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import math

class SimpleCamera:
    def __init__(self, H, W, fov=60.0):
        self.image_height = H
        self.image_width = W
        self.FoVx = math.radians(fov)
        self.FoVy = math.radians(fov)
        self.camera_center = torch.tensor([0.0, 0.0, -2.0], device="cuda") 
        self.world_view_transform = torch.eye(4, device="cuda").transpose(0, 1)
        self.world_view_transform[2, 3] = 2.0 
        self.full_proj_transform = self.getProjectionMatrix(znear=0.01, zfar=100.0).transpose(0, 1).cuda()

    def getProjectionMatrix(self, znear, zfar):
        tanHalfFovY = math.tan(self.FoVy / 2)
        tanHalfFovX = math.tan(self.FoVx / 2)
        P = torch.zeros(4, 4)
        P[0, 0] = 1 / tanHalfFovX
        P[1, 1] = 1 / tanHalfFovY
        P[2, 2] = -(zfar + znear) / (zfar - znear)
        P[3, 2] = -1.0
        P[2, 3] = -(2 * zfar * znear) / (zfar - znear)
        return P

class MedGSModel(nn.Module):
    def __init__(self, num_points=100000, poly_degree=2):
        super().__init__()
        self.poly_degree = poly_degree
        self.setup_functions()
        self._xyz = None
        self._features_dc = None
        self._scaling = None
        self._rotation = None
        self._opacity = None
        self._time_params = None

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = lambda x: torch.log(x/(1-x))
        self.rotation_activation = torch.nn.functional.normalize

    def create_from_pcd(self, num_pts, spatial_lr_scale=1.0):
        print(f"Initializing {num_pts} Gaussians...")
        pts = (torch.rand((num_pts, 3), device="cuda") * 2.0 - 1.0) * spatial_lr_scale
        self._xyz = nn.Parameter(pts.requires_grad_(True))
        
        # --- FIXED: Use 3 Channels (RGB) for compatibility ---
        self._features_dc = nn.Parameter(torch.zeros((num_pts, 3), device="cuda").requires_grad_(True))
        
        self._opacity = nn.Parameter(self.inverse_opacity_activation(0.1 * torch.ones((num_pts, 1), device="cuda")).requires_grad_(True))
        self._scaling = nn.Parameter(self.scaling_inverse_activation(0.05 * torch.ones((num_pts, 3), device="cuda")).requires_grad_(True))
        self._rotation = nn.Parameter(torch.zeros((num_pts, 4), device="cuda").requires_grad_(True))
        self._rotation.data[:, 0] = 1 
        self._time_params = nn.Parameter(torch.zeros((num_pts, self.poly_degree * 3), device="cuda").requires_grad_(True))

    def get_time_deformed_xyz(self, t):
        xyz = self._xyz
        deformation = torch.zeros_like(xyz)
        for i in range(1, self.poly_degree + 1):
             coeffs = self._time_params[:, (i-1)*3 : i*3] 
             deformation = deformation + coeffs * (t ** i)
        return xyz + deformation
    
    def save_ply(self, path, num_time_samples=50):
        """
        Bakes the dynamic 'Folded Gaussians' into a static 3D PLY file.
        We sample the model at 'num_time_samples' intervals along the Z-axis (t).
        """
        from plyfile import PlyData, PlyElement
        
        print(f"Baking MedGS model to {path} with {num_time_samples} temporal slices...")
        
        xyz_all = []
        features_all = []
        opacity_all = []
        scale_all = []
        rot_all = []
        
        # 1. Sample the trajectory (Baking)
        with torch.no_grad():
            for i in range(num_time_samples):
                t = i / float(num_time_samples)
                
                # Get position at this slice (Deformation)
                deformed_xyz = self.get_time_deformed_xyz(torch.tensor(t).cuda())
                
                # Filter invisible points to save space (Optimization)
                mask = self._opacity.squeeze() > 0.01
                
                xyz_all.append(deformed_xyz[mask].cpu().numpy())
                features_all.append(self._features_dc[mask].cpu().numpy())
                opacity_all.append(self._opacity[mask].cpu().numpy())
                scale_all.append(self._scaling[mask].cpu().numpy())
                rot_all.append(self._rotation[mask].cpu().numpy())

        # 2. Concatenate all slices
        xyz = np.concatenate(xyz_all, axis=0)
        f_dc = np.concatenate(features_all, axis=0)
        opacity = np.concatenate(opacity_all, axis=0)
        scale = np.concatenate(scale_all, axis=0)
        rotation = np.concatenate(rot_all, axis=0)

        # 3. Construct PLY Structure
        dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                      ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                      ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
                      ('opacity', 'f4'),
                      ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
                      ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        # Fill data
        elements['x'] = xyz[:, 0]
        elements['y'] = xyz[:, 1]
        elements['z'] = xyz[:, 2]
        elements['nx'] = np.zeros_like(xyz[:, 0]) # Normals (placeholder)
        elements['ny'] = np.zeros_like(xyz[:, 1])
        elements['nz'] = np.zeros_like(xyz[:, 2])
        
        # Color (f_dc)
        elements['f_dc_0'] = f_dc[:, 0]
        elements['f_dc_1'] = f_dc[:, 1]
        elements['f_dc_2'] = f_dc[:, 2]
        
        elements['opacity'] = opacity[:, 0]
        elements['scale_0'] = scale[:, 0]
        elements['scale_1'] = scale[:, 1]
        elements['scale_2'] = scale[:, 2]
        elements['rot_0'] = rotation[:, 0]
        elements['rot_1'] = rotation[:, 1]
        elements['rot_2'] = rotation[:, 2]
        elements['rot_3'] = rotation[:, 3]

        # 4. Save
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        print(f"Saved {len(xyz)} Gaussians to {path}")
        
    def render(self, camera, t_value, bg_color):
        deformed_xyz = self.get_time_deformed_xyz(t_value)
        
        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.image_height),
            image_width=int(camera.image_width),
            tanfovx=math.tan(camera.FoVx * 0.5),
            tanfovy=math.tan(camera.FoVy * 0.5),
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=0,
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
            antialiasing=False 
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        outputs = rasterizer(
            means3D=deformed_xyz,
            means2D=torch.zeros_like(self._xyz, dtype=torch.float32, device="cuda"),
            shs=None,
            colors_precomp=self._features_dc, # Now shape [N, 3]
            opacities=self.opacity_activation(self._opacity),
            scales=self.scaling_activation(self._scaling),
            rotations=self.rotation_activation(self._rotation),
            cov3D_precomp=None
        )
        
        # Unpack result (Output[0] is image, Output[1] is radii)
        return outputs[0], outputs[1]