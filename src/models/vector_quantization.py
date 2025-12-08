"""
Vector Quantization (VQ) Image Compression

Implements codebook-based compression using K-means clustering.
This is the foundation of VQ-VAE and similar methods.

Mathematical Foundation:
-----------------------
VQ approximates data distribution using finite codebook:
C = {c₁, c₂, ..., cₖ}

For each input vector x, find nearest codebook entry:
q(x) = argmin ||x - cᵢ||²
         i

Information Theory Connection:
-----------------------------
- Codebook size K determines rate: R ≤ log₂(K) bits/vector
- Distortion from quantization error: D = E[||x - q(x)||²]
- Optimal codebook minimizes expected distortion
- Related to Lloyd-Max quantization and rate-distortion theory
"""

import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from typing import Tuple, Optional
from scipy.spatial.distance import cdist


class VectorQuantizer:
    """
    Vector Quantization for image compression.
    
    Uses K-means clustering to learn a codebook of representative
    color patterns. Each image patch is encoded as an index into
    this codebook.
    
    Parameters:
    -----------
    codebook_size : int
        Number of entries in codebook (K). Larger = higher quality, lower compression.
    block_size : int
        Size of image blocks to quantize (typically 4x4 or 8x8)
    """
    
    def __init__(self, codebook_size: int = 256, block_size: int = 4):
        self.codebook_size = codebook_size
        self.block_size = block_size
        self.codebook = None
        self.is_trained = False
        
    def _extract_blocks(self, image: np.ndarray) -> np.ndarray:
        """
        Extract non-overlapping blocks from image.
        
        Returns:
        --------
        blocks : np.ndarray, shape (num_blocks, block_size*block_size*channels)
            Flattened image blocks as vectors
        """
        h, w, c = image.shape
        
        # Pad image to multiple of block_size
        pad_h = (self.block_size - h % self.block_size) % self.block_size
        pad_w = (self.block_size - w % self.block_size) % self.block_size
        
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        
        h_pad, w_pad, _ = image.shape
        
        # Extract blocks
        blocks = []
        for i in range(0, h_pad, self.block_size):
            for j in range(0, w_pad, self.block_size):
                block = image[i:i+self.block_size, j:j+self.block_size, :]
                blocks.append(block.flatten())
        
        return np.array(blocks)
    
    def train_codebook(self, images: list, max_samples: int = 100000):
        """
        Train codebook using K-means on image patches.
        
        This learns the optimal set of representative vectors that
        minimize quantization error.
        
        Parameters:
        -----------
        images : list of np.ndarray
            Training images
        max_samples : int
            Maximum number of samples for training (for efficiency)
        """
        print(f"Training codebook with {self.codebook_size} entries...")
        
        # Extract blocks from all training images
        all_blocks = []
        for img in images:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            blocks = self._extract_blocks(img)
            all_blocks.append(blocks)
        
        all_blocks = np.vstack(all_blocks)
        
        # Subsample if too many blocks
        if len(all_blocks) > max_samples:
            indices = np.random.choice(len(all_blocks), max_samples, replace=False)
            all_blocks = all_blocks[indices]
        
        # Learn codebook using K-means
        # MiniBatchKMeans is faster for large datasets
        kmeans = MiniBatchKMeans(
            n_clusters=self.codebook_size,
            random_state=42,
            batch_size=1000,
            max_iter=100,
            verbose=0
        )
        
        kmeans.fit(all_blocks)
        self.codebook = kmeans.cluster_centers_
        self.is_trained = True
        
        # Calculate training statistics
        distances = cdist(all_blocks, self.codebook, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        avg_distortion = np.mean(min_distances ** 2)
        
        print(f"Codebook training complete!")
        print(f"Average quantization distortion: {avg_distortion:.2f}")
        
        return avg_distortion
    
    def _find_nearest_codewords(self, blocks: np.ndarray) -> np.ndarray:
        """
        Find nearest codebook entry for each block.
        
        This is the quantization step: continuous vectors → discrete indices
        
        Returns:
        --------
        indices : np.ndarray
            Codebook indices for each block
        """
        # Compute distances to all codebook entries
        distances = cdist(blocks, self.codebook, metric='euclidean')
        
        # Find nearest (minimum distance)
        indices = np.argmin(distances, axis=1)
        
        return indices
    
    def compress(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Compress image using vector quantization.
        
        Returns:
        --------
        indices : np.ndarray
            Codebook indices (compressed representation)
        metadata : dict
            Information needed for decompression
        """
        if not self.is_trained:
            raise ValueError("Codebook not trained! Call train_codebook() first.")
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        original_shape = image.shape
        
        # Extract and quantize blocks
        blocks = self._extract_blocks(image)
        indices = self._find_nearest_codewords(blocks)
        
        # Calculate compression statistics
        h, w, c = original_shape
        h_pad = ((h + self.block_size - 1) // self.block_size) * self.block_size
        w_pad = ((w + self.block_size - 1) // self.block_size) * self.block_size
        num_blocks_h = h_pad // self.block_size
        num_blocks_w = w_pad // self.block_size
        
        # Reshape indices to 2D grid
        indices_2d = indices.reshape(num_blocks_h, num_blocks_w)
        
        metadata = {
            'original_shape': original_shape,
            'codebook': self.codebook,
            'block_size': self.block_size,
            'codebook_size': self.codebook_size,
            'grid_shape': (num_blocks_h, num_blocks_w)
        }
        
        return indices_2d, metadata
    
    def decompress(self, indices: np.ndarray, metadata: dict) -> np.ndarray:
        """
        Decompress image from codebook indices.
        
        Reconstruction: look up each index in codebook and arrange blocks.
        """
        codebook = metadata['codebook']
        block_size = metadata['block_size']
        original_shape = metadata['original_shape']
        grid_shape = metadata['grid_shape']
        
        # Flatten indices
        indices_flat = indices.flatten()
        
        # Look up codebook entries
        reconstructed_blocks = codebook[indices_flat]
        
        # Reshape blocks back to 2D
        num_blocks_h, num_blocks_w = grid_shape
        h_reconstructed = num_blocks_h * block_size
        w_reconstructed = num_blocks_w * block_size
        
        # Reconstruct image
        reconstructed = np.zeros((h_reconstructed, w_reconstructed, original_shape[2]))
        
        block_idx = 0
        for i in range(0, h_reconstructed, block_size):
            for j in range(0, w_reconstructed, block_size):
                block = reconstructed_blocks[block_idx].reshape(
                    block_size, block_size, original_shape[2]
                )
                reconstructed[i:i+block_size, j:j+block_size, :] = block
                block_idx += 1
        
        # Crop to original size
        reconstructed = reconstructed[:original_shape[0], :original_shape[1], :]
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        
        return reconstructed
    
    def compress_decompress(self, image: np.ndarray) -> np.ndarray:
        """Convenience method for full compression-decompression cycle."""
        indices, metadata = self.compress(image)
        return self.decompress(indices, metadata)
    
    def calculate_theoretical_rate(self) -> float:
        """
        Calculate theoretical compression rate in bits per pixel.
        
        Rate = (log₂(K)) / (block_size²)
        
        where K is codebook size and block_size² is pixels per block.
        """
        bits_per_index = np.log2(self.codebook_size)
        pixels_per_block = self.block_size ** 2
        bits_per_pixel = bits_per_index / pixels_per_block
        
        return bits_per_pixel


class ProductVectorQuantizer(VectorQuantizer):
    """
    Product Vector Quantization (PVQ) for improved compression.
    
    Splits vectors into sub-vectors and quantizes each independently.
    This increases codebook capacity exponentially with linear cost.
    
    If we have M subspaces with K codewords each:
    - Total codebook entries: K^M (exponential!)
    - Storage cost: M*K (linear)
    - This is the principle behind modern quantization methods
    """
    
    def __init__(self, codebook_size: int = 256, block_size: int = 4, num_subspaces: int = 2):
        super().__init__(codebook_size, block_size)
        self.num_subspaces = num_subspaces
        self.subspace_codebooks = []
    
    def train_codebook(self, images: list, max_samples: int = 100000):
        """Train separate codebooks for each subspace."""
        print(f"Training Product VQ with {self.num_subspaces} subspaces...")
        
        # Extract all blocks
        all_blocks = []
        for img in images:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            blocks = self._extract_blocks(img)
            all_blocks.append(blocks)
        
        all_blocks = np.vstack(all_blocks)
        
        if len(all_blocks) > max_samples:
            indices = np.random.choice(len(all_blocks), max_samples, replace=False)
            all_blocks = all_blocks[indices]
        
        # Split into subspaces
        vector_dim = all_blocks.shape[1]
        subspace_dim = vector_dim // self.num_subspaces
        
        # Train codebook for each subspace
        self.subspace_codebooks = []
        for i in range(self.num_subspaces):
            start_idx = i * subspace_dim
            end_idx = start_idx + subspace_dim if i < self.num_subspaces - 1 else vector_dim
            
            subspace_blocks = all_blocks[:, start_idx:end_idx]
            
            kmeans = MiniBatchKMeans(
                n_clusters=self.codebook_size,
                random_state=42 + i,
                batch_size=1000,
                max_iter=100,
                verbose=0
            )
            
            kmeans.fit(subspace_blocks)
            self.subspace_codebooks.append(kmeans.cluster_centers_)
        
        self.is_trained = True
        
        effective_codebook_size = self.codebook_size ** self.num_subspaces
        print(f"Effective codebook size: {effective_codebook_size}")
        print(f"Storage requirement: {self.codebook_size * self.num_subspaces} entries")


def calculate_rate_distortion_curve(
    image: np.ndarray,
    codebook_sizes: list = [16, 32, 64, 128, 256, 512],
    block_size: int = 4
) -> Tuple[list, list]:
    """
    Generate rate-distortion curve for VQ compression.
    
    This demonstrates the fundamental trade-off:
    - Higher rate (more bits) → Lower distortion (better quality)
    - Lower rate (fewer bits) → Higher distortion (worse quality)
    
    Returns:
    --------
    rates : list
        Bits per pixel for each codebook size
    distortions : list
        MSE for each codebook size
    """
    rates = []
    distortions = []
    
    for K in codebook_sizes:
        vq = VectorQuantizer(codebook_size=K, block_size=block_size)
        vq.train_codebook([image])
        
        reconstructed = vq.compress_decompress(image)
        
        # Calculate rate
        rate = vq.calculate_theoretical_rate()
        
        # Calculate distortion (MSE)
        mse = np.mean((image.astype(float) - reconstructed.astype(float)) ** 2)
        
        rates.append(rate)
        distortions.append(mse)
        
        print(f"K={K}: Rate={rate:.3f} bpp, Distortion={mse:.2f}")
    
    return rates, distortions
