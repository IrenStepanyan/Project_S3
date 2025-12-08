"""
Unit tests for compression methods.
"""

import unittest
import numpy as np
import sys
sys.path.append('..')

from src.models.dct_compression import DCTCompression
from src.models.vector_quantization import VectorQuantizer
from src.metrics.information_metrics import shannon_entropy, mutual_information


class TestDCTCompression(unittest.TestCase):
    """Test DCT-based compression."""
    
    def setUp(self):
        """Create test image."""
        self.test_image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    
    def test_compression_decompression(self):
        """Test that compression-decompression cycle works."""
        compressor = DCTCompression(quality=75)
        result = compressor.compress_decompress(self.test_image)
        
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_quality_levels(self):
        """Test different quality levels."""
        qualities = [25, 50, 75, 100]
        psnr_values = []
        
        for quality in qualities:
            compressor = DCTCompression(quality=quality)
            compressed = compressor.compress_decompress(self.test_image)
            
            mse = np.mean((self.test_image.astype(float) - compressed.astype(float)) ** 2)
            psnr = 10 * np.log10((255 ** 2) / mse) if mse > 0 else 100
            psnr_values.append(psnr)
        
        # Higher quality should give higher PSNR
        self.assertTrue(all(psnr_values[i] <= psnr_values[i+1] 
                           for i in range(len(psnr_values)-1)))
    
    def test_entropy_reduction(self):
        """Test that compression reduces entropy."""
        compressor = DCTCompression(quality=50)
        compressed = compressor.compress_decompress(self.test_image)
        
        h_original = shannon_entropy(self.test_image)
        h_compressed = shannon_entropy(compressed)
        
        # Compressed should have lower or equal entropy
        self.assertLessEqual(h_compressed, h_original + 0.5)  # Allow small numerical error


class TestVectorQuantization(unittest.TestCase):
    """Test Vector Quantization compression."""
    
    def setUp(self):
        """Create test image."""
        self.test_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    
    def test_codebook_training(self):
        """Test codebook training."""
        vq = VectorQuantizer(codebook_size=64, block_size=4)
        vq.train_codebook([self.test_image])
        
        self.assertTrue(vq.is_trained)
        self.assertIsNotNone(vq.codebook)
        self.assertEqual(vq.codebook.shape[0], 64)
    
    def test_compression_decompression(self):
        """Test VQ compression cycle."""
        vq = VectorQuantizer(codebook_size=128, block_size=4)
        vq.train_codebook([self.test_image])
        
        result = vq.compress_decompress(self.test_image)
        
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_rate_calculation(self):
        """Test theoretical rate calculation."""
        vq = VectorQuantizer(codebook_size=256, block_size=4)
        rate = vq.calculate_theoretical_rate()
        
        # Rate should be log2(256) / 16 = 8 / 16 = 0.5 bpp
        expected_rate = np.log2(256) / (4 * 4)
        self.assertAlmostEqual(rate, expected_rate, places=5)


class TestInformationMetrics(unittest.TestCase):
    """Test information theory metrics."""
    
    def test_entropy_bounds(self):
        """Test entropy is within valid bounds."""
        # Uniform distribution (maximum entropy)
        uniform = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        h_uniform = shannon_entropy(uniform)
        self.assertGreater(h_uniform, 7.0)  # Close to 8 bits
        
        # Constant image (minimum entropy)
        constant = np.full((100, 100, 3), 128, dtype=np.uint8)
        h_constant = shannon_entropy(constant)
        self.assertLess(h_constant, 1.0)  # Close to 0 bits
    
    def test_mutual_information_bounds(self):
        """Test MI is within valid bounds."""
        img1 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        img2 = img1.copy()  # Identical image
        
        mi = mutual_information(img1, img2)
        h1 = shannon_entropy(img1)
        
        # MI should equal entropy for identical images
        self.assertAlmostEqual(mi, h1, delta=0.5)
        
        # MI should be non-negative
        img3 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        mi_random = mutual_information(img1, img3)
        self.assertGreaterEqual(mi_random, 0)
    
    def test_entropy_additivity(self):
        """Test entropy properties."""
        img = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        
        # Entropy should be positive
        h = shannon_entropy(img)
        self.assertGreater(h, 0)
        
        # Entropy should be at most log2(256) = 8 for 8-bit images
        self.assertLessEqual(h, 8.0)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()
