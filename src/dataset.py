import nibabel as nib
import torch
import numpy as np
from torch.utils.data import Dataset

class MedicalVolumeDataset(Dataset):
    def __init__(self, nii_path):
        self.nii = nib.load(nii_path)
        self.volume = self.nii.get_fdata()
        
        # --- NEW: Extract physical voxel dimensions (dx, dy, dz) ---
        self.zooms = self.nii.header.get_zooms()
        
        # Transpose to (Slices, W, H) to match our rendering orientation
        if self.volume.shape[2] < self.volume.shape[0]:
            self.volume = self.volume.transpose(2, 0, 1)
            # Swap zooms to match the transposed dimensions: (dz, dx, dy)
            self.zooms = (self.zooms[2], self.zooms[0], self.zooms[1])
            
        print(f"[Dataset] Physical Voxel Spacing (Z, X, Y) in mm: {self.zooms}")

        # Normalize Intensity
        self.volume = (self.volume - np.min(self.volume)) / (np.max(self.volume) - np.min(self.volume) + 1e-8)
        
        self.D, self.H, self.W = self.volume.shape
        self.slices = []
        self.z_coords = []
        
        for i in range(self.D):
            slice_data = self.volume[i, :, :]
            if np.sum(slice_data) > 0.01: 
                self.slices.append(torch.tensor(slice_data, dtype=torch.float32).unsqueeze(0))
                self.z_coords.append(i / (self.D - 1))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        return self.slices[idx], self.z_coords[idx]