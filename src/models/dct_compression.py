"""
DCT-based Image Compression

Implements JPEG-style compression using Discrete Cosine Transform.
This is a classical frequency-domain compression technique.

Mathematical Foundation:
-----------------------
The 2D DCT transforms spatial domain to frequency domain:

F(u,v) = (2/√N) C(u)C(v) Σ Σ f(x,y) cos[(2x+1)uπ/2N] cos[(2y+1)vπ/2N]

where C(0) = 1/√2, C(u) = 1 for u > 0

Information Theory Connection:
-----------------------------
- Concentrates energy in low frequencies (high entropy → low entropy)
- Quantization introduces controlled information loss
- Rate determined by quantization table
"""
import numpy as np
import cv2
from scipy.fftpack import dct, idct
from typing import Tuple, Optional

class DCTCompression:
    """
    DCT-based image compression with configurable quality.
    
    Parameters:
    -----------
    quality : int (1-100)
        Compression quality. Higher = better quality, lower compression.
    block_size : int
        Size of DCT blocks (typically 8x8 for JPEG)
    """
    
    def __init__(self, quality: int = 75, block_size: int = 8):
        self.quality = max(1, min(100, quality))
        self.block_size = block_size
        self.quantization_table = self._generate_quantization_table()
        
    def _generate_quantization_table(self) -> np.ndarray:
        """
        Generate quantization table based on quality factor.
        
        Lower quality = more aggressive quantization = more compression
        This is where information is intentionally discarded!
        """
        # Standard JPEG luminance quantization table
        base_table = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ])
        
        # Scale based on quality (JPEG standard formula)
        if self.quality < 50:
            scale = 5000 / self.quality
        else:
            scale = 200 - 2 * self.quality
            
        quantization = np.floor((base_table * scale + 50) / 100)
        quantization[quantization == 0] = 1  # Avoid division by zero
        
        return quantization.astype(np.int32)
    
    def _apply_dct_block(self, block: np.ndarray) -> np.ndarray:
        """Apply 2D DCT to a single block."""
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    def _apply_idct_block(self, block: np.ndarray) -> np.ndarray:
        """Apply inverse 2D DCT to a single block."""
        return idct(idct(block.T, norm='ortho').T, norm='ortho')
    
    def _process_channel(self, channel: np.ndarray, compress: bool = True) -> np.ndarray:
        """
        Process a single color channel.
        
        Parameters:
        -----------
        channel : np.ndarray
            Single channel image
        compress : bool
            If True, apply DCT+quantization. If False, apply dequantization+IDCT
        """
        h, w = channel.shape
        # Pad to multiple of block_size
        pad_h = (self.block_size - h % self.block_size) % self.block_size
        pad_w = (self.block_size - w % self.block_size) % self.block_size
        
        if pad_h > 0 or pad_w > 0:
            channel = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='edge')
        
        h_pad, w_pad = channel.shape
        result = np.zeros_like(channel, dtype=np.float32)
        
        # Process blocks
        for i in range(0, h_pad, self.block_size):
            for j in range(0, w_pad, self.block_size):
                block = channel[i:i+self.block_size, j:j+self.block_size].astype(np.float32)
                
                if compress:
                    # Forward DCT
                    dct_block = self._apply_dct_block(block)
                    # Quantize (information loss happens here!)
                    quantized = np.round(dct_block / self.quantization_table)
                    result[i:i+self.block_size, j:j+self.block_size] = quantized
                else:
                    # Dequantize
                    dequantized = block * self.quantization_table
                    # Inverse DCT
                    reconstructed = self._apply_idct_block(dequantized)
                    result[i:i+self.block_size, j:j+self.block_size] = reconstructed
        
        # Remove padding
        return result[:h, :w]
    
    def compress(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Compress image using DCT.
        
        Returns:
        --------
        compressed_data : np.ndarray
            Quantized DCT coefficients
        metadata : dict
            Information needed for decompression
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Convert to YCbCr (better compression than RGB)
        ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        
        # Process each channel
        compressed_channels = []
        for channel in cv2.split(ycbcr):
            compressed = self._process_channel(channel, compress=True)
            compressed_channels.append(compressed)
        
        compressed_data = np.stack(compressed_channels, axis=-1)
        
        metadata = {
            'original_shape': image.shape,
            'quality': self.quality,
            'block_size': self.block_size,
            'quantization_table': self.quantization_table
        }
        
        return compressed_data, metadata
    
    def decompress(self, compressed_data: np.ndarray, metadata: dict) -> np.ndarray:
        """
        Decompress DCT-compressed image.
        
        Returns:
        --------
        reconstructed : np.ndarray
            Reconstructed image in BGR format
        """
        channels = cv2.split(compressed_data)
        
        # Process each channel
        reconstructed_channels = []
        for channel in channels:
            reconstructed = self._process_channel(channel, compress=False)
            reconstructed_channels.append(reconstructed)
        
        # Merge channels
        ycbcr_reconstructed = cv2.merge(reconstructed_channels)
        ycbcr_reconstructed = np.clip(ycbcr_reconstructed, 0, 255).astype(np.uint8)
        
        # Convert back to BGR
        bgr_reconstructed = cv2.cvtColor(ycbcr_reconstructed, cv2.COLOR_YCrCb2BGR)
        
        return bgr_reconstructed
    
    def compress_decompress(self, image: np.ndarray) -> np.ndarray:
        """Convenience method for full compression-decompression cycle."""
        compressed, metadata = self.compress(image)
        return self.decompress(compressed, metadata)


def calculate_compression_ratio(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Calculate actual compression ratio.
    
    In practice, we'd use entropy coding (Huffman, arithmetic) after quantization.
    This gives theoretical ratio based on non-zero coefficients.
    """
    # Count non-zero coefficients (sparse representation)
    non_zero = np.count_nonzero(compressed)
    total = compressed.size
    
    # Theoretical bits needed (assuming optimal entropy coding)
    sparsity_ratio = non_zero / total
    compression_ratio = 1 / sparsity_ratio if sparsity_ratio > 0 else 1
    
    return compression_ratio
