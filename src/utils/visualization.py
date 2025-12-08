"""
Visualization Utilities

Functions for plotting compression results, metrics, and comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
import cv2


sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_compression_comparison(original: np.ndarray, 
                               compressed_images: Dict[str, np.ndarray],
                               titles: Optional[List[str]] = None,
                               save_path: Optional[str] = None):
    """
    Plot original and multiple compressed versions side by side.
    
    Parameters:
    -----------
    original : np.ndarray
        Original image
    compressed_images : dict
        Dictionary of compressed images {method_name: image}
    titles : list, optional
        Custom titles for each subplot
    save_path : str, optional
        Path to save figure
    """
    n_images = len(compressed_images) + 1
    fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
    
    if n_images == 2:
        axes = [axes]
    
    # Plot original
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Plot compressed versions
    for idx, (method, image) in enumerate(compressed_images.items(), 1):
        axes[idx].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        title = titles[idx] if titles and idx < len(titles) else f'{method.upper()}'
        axes[idx].set_title(title, fontsize=14, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_rate_distortion_curve(rates: List[float], 
                               distortions: List[float],
                               method_name: str = '',
                               save_path: Optional[str] = None):
    """
    Plot rate-distortion curve.
    
    Shows fundamental trade-off: higher rate → lower distortion
    
    Parameters:
    -----------
    rates : list
        Compression rates (bits per pixel)
    distortions : list
        Distortion values (MSE)
    method_name : str
        Name of compression method
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(rates, distortions, 'o-', linewidth=2, markersize=8, label=method_name)
    
    ax.set_xlabel('Rate (bits per pixel)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distortion (MSE)', fontsize=12, fontweight='bold')
    ax.set_title('Rate-Distortion Curve', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add annotations
    for r, d in zip(rates[::2], distortions[::2]):
        ax.annotate(f'{r:.2f} bpp\n{d:.1f} MSE', 
                   xy=(r, d), xytext=(10, 10),
                   textcoords='offset points',
                   fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_multiple_rd_curves(rd_data: Dict[str, Tuple[List, List]],
                           save_path: Optional[str] = None):
    """
    Plot rate-distortion curves for multiple methods.
    
    Parameters:
    -----------
    rd_data : dict
        Dictionary {method_name: (rates, distortions)}
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(rd_data)))
    
    for (method, (rates, distortions)), color in zip(rd_data.items(), colors):
        ax.plot(rates, distortions, 'o-', linewidth=2.5, markersize=8, 
               label=method, color=color, alpha=0.8)
    
    ax.set_xlabel('Rate (bits per pixel)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Distortion (MSE)', fontsize=13, fontweight='bold')
    ax.set_title('Rate-Distortion Comparison', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    ax.set_yscale('log')  # Log scale often clearer for distortion
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_metrics_radar(metrics: Dict[str, float],
                      save_path: Optional[str] = None):
    """
    Plot metrics as radar chart.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of normalized metrics [0, 1]
    save_path : str, optional
        Path to save figure
    """
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    # Number of variables
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]
    angles += angles[:1]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4')
    ax.fill(angles, values, alpha=0.25, color='#1f77b4')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Compression Metrics', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_entropy_histogram(original: np.ndarray, compressed: np.ndarray,
                          save_path: Optional[str] = None):
    """
    Plot histograms showing entropy change.
    
    Parameters:
    -----------
    original : np.ndarray
        Original image
    compressed : np.ndarray
        Compressed image
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Convert to grayscale for histogram
    if len(original.shape) == 3:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        comp_gray = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray = original
        comp_gray = compressed
    
    # Plot original histogram
    axes[0].hist(orig_gray.ravel(), bins=256, range=(0, 256), 
                density=True, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_title('Original Image Histogram', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Pixel Value', fontsize=11)
    axes[0].set_ylabel('Probability', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot compressed histogram
    axes[1].hist(comp_gray.ravel(), bins=256, range=(0, 256), 
                density=True, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_title('Compressed Image Histogram', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Pixel Value', fontsize=11)
    axes[1].set_ylabel('Probability', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]],
                           save_path: Optional[str] = None):
    """
    Plot bar chart comparing metrics across methods.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary {method: {metric: value}}
    save_path : str, optional
        Path to save figure
    """
    methods = list(metrics_dict.keys())
    metric_names = list(next(iter(metrics_dict.values())).keys())
    
    n_metrics = len(metric_names)
    n_methods = len(methods)
    
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(16, 10))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metric_names):
        values = [metrics_dict[method][metric] for method in methods]
        
        axes[idx].bar(methods, values, color=plt.cm.tab10(np.linspace(0, 1, n_methods)))
        axes[idx].set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Value', fontsize=10)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_compression_artifacts(original: np.ndarray, compressed: np.ndarray,
                               save_path: Optional[str] = None):
    """
    Visualize compression artifacts by showing difference.
    
    Parameters:
    -----------
    original : np.ndarray
        Original image
    compressed : np.ndarray
        Compressed image
    save_path : str, optional
        Path to save figure
    """
    # Calculate difference
    diff = cv2.absdiff(original, compressed)
    
    # Amplify difference for visualization
    diff_amplified = cv2.convertScaleAbs(diff, alpha=5.0)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    # Compressed
    axes[1].imshow(cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Compressed', fontsize=13, fontweight='bold')
    axes[1].axis('off')

    # Difference
    axes[2].imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Difference (Artifacts)', fontsize=13, fontweight='bold')
    axes[2].axis('off')

    # Amplified difference
    axes[3].imshow(cv2.cvtColor(diff_amplified, cv2.COLOR_BGR2RGB))
    axes[3].set_title('Difference (5x Amplified)', fontsize=13, fontweight='bold')
    axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()