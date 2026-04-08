"""Evaluation metrics: PSNR and SSIM."""
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(img1, img2):
    """Compute PSNR between two images.

    Args:
        img1, img2: numpy arrays (H,W,C) in [0,1] or torch tensors (C,H,W) in [0,1]
    Returns:
        PSNR value in dB
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)
    img1 = np.clip(img1, 0, 1).astype(np.float64)
    img2 = np.clip(img2, 0, 1).astype(np.float64)
    return peak_signal_noise_ratio(img1, img2, data_range=1.0)


def compute_ssim(img1, img2):
    """Compute SSIM between two images.

    Args:
        img1, img2: numpy arrays (H,W,C) in [0,1] or torch tensors (C,H,W) in [0,1]
    Returns:
        SSIM value
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)
    img1 = np.clip(img1, 0, 1).astype(np.float64)
    img2 = np.clip(img2, 0, 1).astype(np.float64)
    return structural_similarity(img1, img2, channel_axis=2, data_range=1.0)
