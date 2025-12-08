"""
Image Processing Utilities

Helper functions for image I/O, preprocessing, and manipulation.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt


def load_image(path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load image from file.
    
    Parameters:
    -----------
    path : str
        Path to image file
    target_size : tuple, optional
        (width, height) to resize image to
    
    Returns:
    --------
    image : np.ndarray
        Loaded image in BGR format
    """
    image = cv2.imread(path)
    
    if image is None:
        raise ValueError(f"Could not load image from {path}")
    
    if target_size is not None:
        image = cv2.resize(image, target_size)
    
    return image


def save_image(image: np.ndarray, path: str):
    """
    Save image to file.
    
    Parameters:
    -----------
    image : np.ndarray
        Image to save
    path : str
        Output path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, image)


def resize_image(image: np.ndarray, max_size: int = 512) -> np.ndarray:
    """
    Resize image maintaining aspect ratio.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    max_size : int
        Maximum dimension (width or height)
    
    Returns:
    --------
    resized : np.ndarray
        Resized image
    """
    h, w = image.shape[:2]
    
    if max(h, w) <= max_size:
        return image
    
    if h > w:
        new_h = max_size
        new_w = int(w * (max_size / h))
    else:
        new_w = max_size
        new_h = int(h * (max_size / w))
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    
    Returns:
    --------
    normalized : np.ndarray
        Normalized image
    """
    return image.astype(np.float32) / 255.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Denormalize image from [0, 1] to [0, 255] range.
    
    Parameters:
    -----------
    image : np.ndarray
        Normalized image
    
    Returns:
    --------
    denormalized : np.ndarray
        Denormalized image
    """
    return (image * 255).clip(0, 255).astype(np.uint8)


def rgb_to_ycbcr(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB to YCbCr color space.
    
    YCbCr is better for compression as Y (luminance) contains most info.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)


def ycbcr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert YCbCr to RGB color space."""
    return cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)


def add_noise(image: np.ndarray, noise_type: str = 'gaussian', 
              amount: float = 0.01) -> np.ndarray:
    """
    Add noise to image for testing robustness.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    noise_type : str
        'gaussian', 'salt_pepper', or 'poisson'
    amount : float
        Noise intensity
    
    Returns:
    --------
    noisy : np.ndarray
        Image with added noise
    """
    if noise_type == 'gaussian':
        noise = np.random.normal(0, amount * 255, image.shape)
        noisy = image.astype(float) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
    elif noise_type == 'salt_pepper':
        noisy = image.copy()
        # Salt
        num_salt = int(amount * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
        noisy[coords[0], coords[1]] = 255
        # Pepper
        num_pepper = int(amount * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
        noisy[coords[0], coords[1]] = 0
        
    elif noise_type == 'poisson':
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return noisy


def create_test_images() -> dict:
    """
    Create synthetic test images with known properties.
    
    Returns:
    --------
    images : dict
        Dictionary of test images with different characteristics
    """
    size = 256
    
    images = {}
    
    # High entropy (random noise)
    images['random'] = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    
    # Low entropy (solid color)
    images['solid'] = np.full((size, size, 3), 128, dtype=np.uint8)
    
    # Medium entropy (gradient)
    gradient = np.linspace(0, 255, size)
    images['gradient'] = np.stack([gradient] * size).astype(np.uint8)
    images['gradient'] = cv2.cvtColor(images['gradient'], cv2.COLOR_GRAY2BGR)
    
    # Structured (checkerboard)
    checker = np.zeros((size, size), dtype=np.uint8)
    square_size = 32
    checker[::square_size*2, ::square_size*2] = 255
    checker[square_size::square_size*2, square_size::square_size*2] = 255
    images['checker'] = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)
    
    # Natural-like (Perlin noise simulation)
    x = np.linspace(0, 10, size)
    y = np.linspace(0, 10, size)
    X, Y = np.meshgrid(x, y)
    natural = (np.sin(X) + np.cos(Y)) * 127.5 + 127.5
    images['natural'] = natural.astype(np.uint8)
    images['natural'] = cv2.cvtColor(images['natural'], cv2.COLOR_GRAY2BGR)
    
    return images


def batch_process_images(image_paths: List[str], 
                         process_fn, 
                         output_dir: str) -> List[str]:
    """
    Process multiple images with a function.
    
    Parameters:
    -----------
    image_paths : list
        List of input image paths
    process_fn : callable
        Function that takes and returns an image
    output_dir : str
        Directory to save processed images
    
    Returns:
    --------
    output_paths : list
        List of output image paths
    """
    output_paths = []
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for path in image_paths:
        image = load_image(path)
        processed = process_fn(image)
        
        output_path = Path(output_dir) / Path(path).name
        save_image(processed, str(output_path))
        output_paths.append(str(output_path))
    
    return output_paths
