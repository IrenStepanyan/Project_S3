"""
Information Theory Metrics for Image Compression

Implements key information-theoretic measures:
- Shannon Entropy
- Mutual Information
- Kullback-Leibler Divergence
- Rate-Distortion Trade-off
- Cross-Entropy

These metrics quantify information content, preservation, and loss.

Mathematical Foundations:
------------------------

1. Entropy: H(X) = -Σ p(x) log₂ p(x)
   Measures average information content in bits

2. Joint Entropy: H(X,Y) = -Σ p(x,y) log₂ p(x,y)
   Information in two variables together

3. Mutual Information: I(X;Y) = H(X) + H(Y) - H(X,Y)
   How much knowing Y tells us about X

4. KL Divergence: D_KL(P||Q) = Σ p(x) log₂(p(x)/q(x))
   How different distribution Q is from P

5. Cross-Entropy: H(P,Q) = -Σ p(x) log₂ q(x)
   Expected bits using wrong distribution
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.signal import convolve2d
from typing import Tuple, Optional
import warnings


def calculate_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Calculate normalized histogram (probability distribution).
    
    Parameters:
    -----------
    image : np.ndarray
        Input image (grayscale or will be converted)
    bins : int
        Number of histogram bins
    
    Returns:
    --------
    hist : np.ndarray
        Normalized histogram (probability distribution)
    """
    if len(image.shape) == 3:
        # Convert to grayscale
        image = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
    
    # Flatten image
    pixels = image.flatten()
    
    # Calculate histogram
    hist, _ = np.histogram(pixels, bins=bins, range=(0, 255), density=False)
    
    # Normalize to probability distribution
    hist = hist.astype(float) / np.sum(hist)
    
    # Remove zeros (for log calculations)
    hist = hist[hist > 0]
    
    return hist


def shannon_entropy(image: np.ndarray, bins: int = 256) -> float:
    """
    Calculate Shannon entropy of an image.
    
    H(X) = -Σ p(x) log₂ p(x)
    
    Interpretation:
    ---------------
    - Higher entropy = more random, harder to compress
    - Lower entropy = more predictable, easier to compress
    - Maximum entropy for 8-bit image: 8 bits/pixel
    - Uniform distribution achieves maximum entropy
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    bins : int
        Number of histogram bins
    
    Returns:
    --------
    entropy : float
        Shannon entropy in bits
    """
    hist = calculate_histogram(image, bins)
    
    # Calculate entropy: -Σ p(x) log₂ p(x)
    entropy = -np.sum(hist * np.log2(hist))
    
    return entropy


def joint_entropy(image1: np.ndarray, image2: np.ndarray, bins: int = 256) -> float:
    """
    Calculate joint entropy H(X,Y) of two images.
    
    H(X,Y) = -Σ p(x,y) log₂ p(x,y)
    
    Measures total information in both images together.
    If images are independent: H(X,Y) = H(X) + H(Y)
    If images are identical: H(X,Y) = H(X) = H(Y)
    
    Parameters:
    -----------
    image1, image2 : np.ndarray
        Input images (must be same size)
    bins : int
        Number of histogram bins
    
    Returns:
    --------
    joint_ent : float
        Joint entropy in bits
    """
    if len(image1.shape) == 3:
        image1 = 0.299 * image1[:,:,0] + 0.587 * image1[:,:,1] + 0.114 * image1[:,:,2]
    if len(image2.shape) == 3:
        image2 = 0.299 * image2[:,:,0] + 0.587 * image2[:,:,1] + 0.114 * image2[:,:,2]
    
    # Flatten images
    pixels1 = image1.flatten().astype(int)
    pixels2 = image2.flatten().astype(int)
    
    # Calculate 2D histogram
    hist_2d, _, _ = np.histogram2d(
        pixels1, pixels2,
        bins=bins,
        range=[[0, 255], [0, 255]]
    )
    
    # Normalize to joint probability distribution
    hist_2d = hist_2d / np.sum(hist_2d)
    
    # Remove zeros
    hist_2d = hist_2d[hist_2d > 0]
    
    # Calculate joint entropy
    joint_ent = -np.sum(hist_2d * np.log2(hist_2d))
    
    return joint_ent


def mutual_information(original: np.ndarray, compressed: np.ndarray, bins: int = 256) -> float:
    """
    Calculate mutual information between original and compressed images.
    
    I(X;Y) = H(X) + H(Y) - H(X,Y)
    
    Alternative formulation:
    I(X;Y) = H(X) - H(X|Y)  (reduction in uncertainty about X given Y)
    
    Interpretation:
    ---------------
    - I(X;Y) = 0: Images are independent (no information preserved)
    - I(X;Y) = H(X): Y tells us everything about X (perfect preservation)
    - Higher MI = more information preserved in compression
    - MI is symmetric: I(X;Y) = I(Y;X)
    - MI ≥ 0 always (can't have negative information)
    
    Parameters:
    -----------
    original : np.ndarray
        Original image
    compressed : np.ndarray
        Compressed/reconstructed image
    bins : int
        Number of histogram bins
    
    Returns:
    --------
    mi : float
        Mutual information in bits
    """
    # Calculate individual entropies
    h_original = shannon_entropy(original, bins)
    h_compressed = shannon_entropy(compressed, bins)
    
    # Calculate joint entropy
    h_joint = joint_entropy(original, compressed, bins)
    
    # Mutual information
    mi = h_original + h_compressed - h_joint
    
    # Ensure non-negative (numerical errors can cause small negatives)
    mi = max(0, mi)
    
    return mi


def kl_divergence(original: np.ndarray, compressed: np.ndarray, bins: int = 256) -> float:
    """
    KL divergence D_KL(P || Q) between the intensity distributions
    of the original (P) and compressed (Q) images.
    """

    import numpy as np
    from scipy.stats import entropy as scipy_entropy

    # Flatten images
    original = np.asarray(original, dtype=np.float32).flatten()
    compressed = np.asarray(compressed, dtype=np.float32).flatten()

    # Shared dynamic range
    data_min = min(original.min(), compressed.min())
    data_max = max(original.max(), compressed.max())

    # Compute histograms with identical bin edges!
    hist_original, _ = np.histogram(
        original, bins=bins, range=(data_min, data_max), density=True
    )
    hist_compressed, _ = np.histogram(
        compressed, bins=bins, range=(data_min, data_max), density=True
    )

    # Add small epsilon to avoid log(0) and division by zero
    eps = 1e-12
    hist_original = hist_original + eps
    hist_compressed = hist_compressed + eps

    # Normalize to proper probability distributions
    hist_original /= hist_original.sum()
    hist_compressed /= hist_compressed.sum()

    # Compute KL divergence in bits
    kl = scipy_entropy(hist_original, hist_compressed, base=2)

    return float(kl)


def cross_entropy(original: np.ndarray, compressed: np.ndarray, bins: int = 256) -> float:
    """
    Calculate cross-entropy H(P,Q).
    
    H(P,Q) = -Σ p(x) log₂ q(x)
    
    Relationship to KL divergence:
    H(P,Q) = H(P) + D_KL(P||Q)
    
    Interpretation:
    ---------------
    - Expected bits needed to encode P using code optimized for Q
    - H(P,Q) ≥ H(P) with equality iff P = Q
    - Minimum when Q = P (optimal coding for P)
    
    Parameters:
    -----------
    original : np.ndarray
        Original image (true distribution P)
    compressed : np.ndarray
        Compressed image (model distribution Q)
    bins : int
        Number of histogram bins
    
    Returns:
    --------
    ce : float
        Cross-entropy in bits
    """
    h_original = shannon_entropy(original, bins)
    kl_div = kl_divergence(original, compressed, bins)
    
    ce = h_original + kl_div
    
    return ce


def conditional_entropy(original: np.ndarray, compressed: np.ndarray, bins: int = 256) -> float:
    """
    Calculate conditional entropy H(X|Y).
    
    H(X|Y) = H(X,Y) - H(Y)
    
    Interpretation:
    ---------------
    - Average uncertainty about X given we know Y
    - H(X|Y) = 0: Y completely determines X
    - H(X|Y) = H(X): Y tells us nothing about X (independent)
    - Measures information lost in compression
    
    Parameters:
    -----------
    original : np.ndarray
        Original image X
    compressed : np.ndarray
        Compressed image Y
    bins : int
        Number of histogram bins
    
    Returns:
    --------
    cond_ent : float
        Conditional entropy in bits
    """
    h_joint = joint_entropy(original, compressed, bins)
    h_compressed = shannon_entropy(compressed, bins)
    
    cond_ent = h_joint - h_compressed
    
    return max(0, cond_ent)  # Ensure non-negative


def compression_efficiency(original: np.ndarray, compressed: np.ndarray, bins: int = 256) -> dict:
    """
    Calculate comprehensive compression efficiency metrics.
    
    Returns dictionary with multiple information-theoretic measures
    that together characterize the compression quality.
    
    Returns:
    --------
    metrics : dict
        Dictionary containing:
        - entropy_original: Original entropy
        - entropy_compressed: Compressed entropy
        - entropy_reduction: H(X) - H(Y)
        - mutual_information: I(X;Y)
        - information_loss: H(X) - I(X;Y) = H(X|Y)
        - kl_divergence: D_KL(P||Q)
        - cross_entropy: H(P,Q)
        - compression_rate: H(Y) / H(X)
        - information_preservation: I(X;Y) / H(X)
    """
    h_original = shannon_entropy(original, bins)
    h_compressed = shannon_entropy(compressed, bins)
    mi = mutual_information(original, compressed, bins)
    kl_div = kl_divergence(original, compressed, bins)
    ce = cross_entropy(original, compressed, bins)
    
    metrics = {
        'entropy_original': h_original,
        'entropy_compressed': h_compressed,
        'entropy_reduction': h_original - h_compressed,
        'mutual_information': mi,
        'information_loss': h_original - mi,
        'kl_divergence': kl_div,
        'cross_entropy': ce,
        'compression_rate': h_compressed / h_original if h_original > 0 else 0,
        'information_preservation': mi / h_original if h_original > 0 else 0
    }
    
    return metrics


def estimate_compressibility(image: np.ndarray) -> dict:
    """
    Estimate how compressible an image is using information theory.
    
    Analyzes:
    - Entropy (information content)
    - Redundancy (predictability)
    - Spatial correlation (neighboring pixels)
    
    Returns:
    --------
    analysis : dict
        Compressibility analysis
    """
    h = shannon_entropy(image)
    
    # Maximum entropy for 8-bit image
    max_entropy = 8.0
    
    # Redundancy: how far from maximum entropy
    redundancy = max_entropy - h
    redundancy_percent = (redundancy / max_entropy) * 100
    
    # Spatial correlation: entropy of differences
    if len(image.shape) == 3:
        gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
    else:
        gray = image
    
    # Horizontal differences
    diff_h = np.abs(np.diff(gray, axis=1))
    h_diff_h = shannon_entropy(diff_h)
    
    # Vertical differences
    diff_v = np.abs(np.diff(gray, axis=0))
    h_diff_v = shannon_entropy(diff_v)
    
    # Average differential entropy
    h_diff_avg = (h_diff_h + h_diff_v) / 2
    
    # Lower differential entropy = higher spatial correlation = more compressible
    spatial_predictability = max(0, (h - h_diff_avg) / h * 100) if h > 0 else 0
    
    analysis = {
        'entropy': h,
        'max_entropy': max_entropy,
        'redundancy': redundancy,
        'redundancy_percent': redundancy_percent,
        'differential_entropy': h_diff_avg,
        'spatial_predictability': spatial_predictability,
        'compressibility_score': (redundancy_percent + spatial_predictability) / 2
    }
    
    return analysis
