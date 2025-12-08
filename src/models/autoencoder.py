"""
Autoencoder-based Image Compression

Implements learned compression using convolutional autoencoders.
This demonstrates how neural networks learn optimal representations.

Mathematical Foundation:
-----------------------
Encoder: z = f_θ(x)  maps image x to latent representation z
Decoder: x̂ = g_φ(z)  reconstructs image from latent code

Loss function combines reconstruction error and rate term:
L = D(x, x̂) + λ·R(z)

where D is distortion (MSE) and R is rate (entropy of z)

Information Bottleneck Principle:
---------------------------------
The latent space z acts as an information bottleneck that:
- Minimizes I(X; Z) (compress information)
- Maximizes I(Y; Z) (preserve relevant information)
- Achieves optimal rate-distortion trade-off
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for image compression.
    
    Architecture:
    - Encoder: Progressively downsamples using strided convolutions
    - Latent: Bottleneck layer (compressed representation)
    - Decoder: Progressively upsamples using transposed convolutions
    """
    
    def __init__(self, latent_dim: int = 128, num_channels: int = 3):
        super(ConvAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder - reduces spatial dimensions while increasing features
        self.encoder = nn.Sequential(
            # 256x256x3 -> 128x128x64
            nn.Conv2d(num_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 128x128x64 -> 64x64x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 64x64x128 -> 32x32x256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 32x32x256 -> 16x16x512
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Latent bottleneck - the compressed representation!
        # This is where maximum compression happens
        self.fc_encode = nn.Linear(512 * 16 * 16, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * 16 * 16)
        
        # Decoder - mirror of encoder
        self.decoder = nn.Sequential(
            # 16x16x512 -> 32x32x256
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 32x32x256 -> 64x64x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 64x64x128 -> 128x128x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 128x128x64 -> 256x256x3
            nn.ConvTranspose2d(64, num_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent representation.
        
        This is the compression step - we go from high-dimensional image
        space to low-dimensional latent space.
        """
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        z = self.fc_encode(x)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to image.
        
        This is the decompression step - we reconstruct the image
        from the compressed latent code.
        """
        batch_size = z.size(0)
        x = self.fc_decode(z)
        x = x.view(batch_size, 512, 16, 16)
        x = self.decoder(x)
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full compression-decompression pipeline."""
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z


class RateDistortionAutoencoder(ConvAutoencoder):
    """
    Enhanced autoencoder with explicit rate-distortion optimization.
    
    This implements the information-theoretic rate-distortion trade-off:
    L = D + λ·R
    
    where:
    - D: Distortion (reconstruction error)
    - R: Rate (entropy of latent representation)
    - λ: Lagrange multiplier controlling trade-off
    """
    
    def __init__(self, latent_dim: int = 128, num_channels: int = 3, lambda_rate: float = 0.01):
        super().__init__(latent_dim, num_channels)
        self.lambda_rate = lambda_rate
    
    def compute_rate(self, z: torch.Tensor) -> torch.Tensor:
        """
        Estimate rate (bits needed to encode z).
        
        We approximate entropy using the continuous approximation:
        H(Z) ≈ 0.5 * log(2πe * σ²)
        
        Lower entropy = more compressible = lower rate
        """
        # Estimate entropy from latent statistics
        mean = z.mean(dim=0)
        std = z.std(dim=0) + 1e-8  # Add epsilon for numerical stability
        
        # Continuous entropy approximation
        entropy = 0.5 * torch.log(2 * np.pi * np.e * std ** 2)
        rate = entropy.sum()
        
        return rate
    
    def compute_distortion(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """
        Compute distortion (reconstruction error).
        
        Uses MSE as distortion metric, but could also use:
        - SSIM (Structural Similarity Index)
        - Perceptual loss (VGG features)
        - MS-SSIM (Multi-Scale SSIM)
        """
        mse = F.mse_loss(x_recon, x, reduction='mean')
        return mse
    
    def loss_function(self, x: torch.Tensor, x_recon: torch.Tensor, z: torch.Tensor) -> dict:
        """
        Rate-Distortion loss function.
        
        Returns dictionary with individual components for monitoring.
        """
        distortion = self.compute_distortion(x, x_recon)
        rate = self.compute_rate(z)
        
        # Total rate-distortion loss
        total_loss = distortion + self.lambda_rate * rate
        
        return {
            'total_loss': total_loss,
            'distortion': distortion,
            'rate': rate,
            'bpp': rate / (x.size(2) * x.size(3))  # Bits per pixel
        }


class AutoencoderCompressor:
    """
    Wrapper class for using trained autoencoder for compression.
    """
    
    def __init__(self, model_path: Optional[str] = None, latent_dim: int = 128, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = RateDistortionAutoencoder(latent_dim=latent_dim).to(self.device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()
    
    def compress(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Compress image using autoencoder.
        
        Returns:
        --------
        latent_code : np.ndarray
            Compressed latent representation
        metadata : dict
            Information for decompression
        """
        # Preprocess
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image_tensor = image_tensor.to(self.device)
        
        # Encode
        with torch.no_grad():
            latent_code = self.model.encode(image_tensor)
        
        latent_np = latent_code.cpu().numpy()
        
        metadata = {
            'original_shape': image.shape,
            'latent_dim': self.model.latent_dim,
            'device': str(self.device)
        }
        
        return latent_np, metadata
    
    def decompress(self, latent_code: np.ndarray, metadata: dict) -> np.ndarray:
        """Decompress latent code back to image."""
        latent_tensor = torch.from_numpy(latent_code).float().to(self.device)
        
        with torch.no_grad():
            reconstructed = self.model.decode(latent_tensor)
        
        # Postprocess
        reconstructed_np = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
        reconstructed_np = (reconstructed_np * 255).astype(np.uint8)
        
        return reconstructed_np
    
    def compress_decompress(self, image: np.ndarray) -> np.ndarray:
        """Convenience method for full cycle."""
        latent, metadata = self.compress(image)
        return self.decompress(latent, metadata)
