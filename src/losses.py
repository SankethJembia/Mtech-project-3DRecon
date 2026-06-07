import torch
import torch.nn.functional as F

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def ssim(img1, img2, window_size=11, size_average=True):
    # Standard SSIM implementation (omitted for brevity, use kornia or pytorch_msssim in prod)
    # Placeholder:
    return 1.0 - F.mse_loss(img1, img2) # Simplification for example

def sigma_regularization(scaling, N_frames):
    """
    Equation (7): Penalize temporal spread if too small or too large[cite: 176].
    """
    # Assuming one of the scaling dimensions corresponds to time/Z-spread
    sigma_t = scaling[:, 2] # Assuming 3rd dim is Z
    
    lower_bound = 2.0 / N_frames
    upper_bound = 1.0
    
    loss_low = torch.relu(lower_bound - sigma_t)
    loss_high = torch.relu(sigma_t - upper_bound)
    
    return torch.mean(loss_low + loss_high)

def medgs_loss(pred_img, gt_img, scaling, N_frames, lambda1=0.8, lambda4=0.1):
    """
    Equation (8) for Mesh Reconstruction: L1 + Sigma Reg [cite: 179]
    """
    Ll1 = l1_loss(pred_img, gt_img)
    Lsigma = sigma_regularization(scaling, N_frames)
    
    return (1.0 - lambda1) * Ll1 + lambda4 * Lsigma
def calculate_psnr(img1, img2):
    """Calculates PSNR between two image tensors."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(100.0, device=img1.device)
    max_pixel = 1.0 # Images are normalized to [0, 1]
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr