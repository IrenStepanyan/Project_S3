"""
Image Quality Metrics

Implements standard quality assessment metrics for compressed images:
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)
- Mean Absolute Error (MAE)

These complement information theory metrics by measuring perceptual quality.
"""

import numpy as np
from typing import Tuple
import cv2


def mean_squared_error(original: np.ndarray, compressed: np.ndarray) -> float:
    """Calculate Mean Squared Error (MSE)."""
    if original.shape != compressed.shape:
        raise ValueError("Images must have the same dimensions")
    
    mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
    return mse


def peak_signal_to_noise_ratio(original: np.ndarray, compressed: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR)."""
    mse = mean_squared_error(original, compressed)
    max_pixel = 255.0 if original.dtype == np.uint8 else np.max(original)

    if mse == 0:
        return float('inf')
    
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr


def mean_absolute_error(original: np.ndarray, compressed: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE)."""
    if original.shape != compressed.shape:
        raise ValueError("Images must have the same dimensions")
    
    mae = np.mean(np.abs(original.astype(float) - compressed.astype(float)))
    return mae


def structural_similarity_index(
    original: np.ndarray,
    compressed: np.ndarray,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03
) -> float:
    """Calculate Structural Similarity Index (SSIM)."""
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        compressed_gray = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        compressed_gray = compressed

    img1 = original_gray.astype(float)
    img2 = compressed_gray.astype(float)

    C1 = (k1 * 255) ** 2
    C2 = (k2 * 255) ** 2

    window = cv2.getGaussianKernel(window_size, 1.5)
    window = np.outer(window, window.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    ssim = np.mean(ssim_map)
    return ssim


def multi_scale_ssim(original: np.ndarray, compressed: np.ndarray, scales: int = 5) -> float:
    """Calculate Multi-Scale SSIM (MS-SSIM)."""
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = []

    for i in range(scales):
        ssim_val = structural_similarity_index(original, compressed)
        levels.append(ssim_val)

        if i < scales - 1:
            original = cv2.pyrDown(original)
            compressed = cv2.pyrDown(compressed)

    ms_ssim = np.prod(np.power(levels, weights[:scales]))
    return ms_ssim


def calculate_all_metrics(original: np.ndarray, compressed: np.ndarray) -> dict:
    """Calculate all quality metrics at once."""
    metrics = {
        'mse': mean_squared_error(original, compressed),
        'psnr': peak_signal_to_noise_ratio(original, compressed),
        'mae': mean_absolute_error(original, compressed),
        'ssim': structural_similarity_index(original, compressed)
    }
    return metrics


def rate_distortion_point(original: np.ndarray, compressed: np.ndarray, rate: float) -> Tuple[float, float]:
    """Calculate a single point on the rate-distortion curve."""
    mse = mean_squared_error(original, compressed)
    return rate, mse
