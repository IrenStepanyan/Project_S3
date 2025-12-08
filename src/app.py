"""
Flask Web Application for InfoTheory Image Compression Explorer

Provides a web interface for:
- Uploading images
- Selecting compression methods
- Adjusting compression parameters
- Viewing metrics in real-time
- Comparing different methods
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import cv2
import io
import base64
from PIL import Image
import json
import os

from models.dct_compression import DCTCompression
from models.autoencoder import AutoencoderCompressor
from models.vector_quantization import VectorQuantizer
from metrics.information_metrics import (
    shannon_entropy, mutual_information, kl_divergence,
    compression_efficiency, estimate_compressibility
)
from metrics.quality_metrics import calculate_all_metrics

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def decode_image(image_data: str) -> np.ndarray:
    """Decode base64 image data to numpy array."""
    # Remove data URL prefix if present
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    
    # Decode base64
    image_bytes = base64.b64decode(image_data)
    
    # Convert to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return image


def encode_image(image: np.ndarray) -> str:
    """Encode numpy array to base64 string."""
    # Encode image
    _, buffer = cv2.imencode('.png', image)
    
    # Convert to base64
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/png;base64,{image_base64}"


@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('index.html')


@app.route('/api/compress', methods=['POST'])
def compress():
    """
    Compress an image using specified method.
    
    Request JSON:
    {
        "image": "base64_encoded_image",
        "method": "dct|autoencoder|vq",
        "quality": 1-100
    }
    
    Response JSON:
    {
        "compressed_image": "base64_encoded_image",
        "metrics": {
            "entropy_original": float,
            "entropy_compressed": float,
            "mutual_information": float,
            "kl_divergence": float,
            "psnr": float,
            "ssim": float,
            ...
        }
    }
    """
    try:
        data = request.get_json()
        
        # Decode image
        image = decode_image(data['image'])
        method = data.get('method', 'dct')
        quality = int(data.get('quality', 75))
        
        # Apply compression
        if method == 'dct':
            compressor = DCTCompression(quality=quality)
            compressed_image = compressor.compress_decompress(image)
            
        elif method == 'autoencoder':
            # For demo, use simplified autoencoder
            # In production, load pre-trained model
            latent_dim = int(128 * (quality / 100))
            compressor = AutoencoderCompressor(latent_dim=latent_dim, device='cpu')
            compressed_image = compressor.compress_decompress(image)
            
        elif method == 'vq':
            codebook_size = int(256 * (quality / 100))
            codebook_size = max(16, min(512, codebook_size))
            vq = VectorQuantizer(codebook_size=codebook_size, block_size=4)
            # Train on the image itself (for demo)
            vq.train_codebook([image])
            compressed_image = vq.compress_decompress(image)
            
        else:
            return jsonify({'error': 'Invalid compression method'}), 400
        
        # Calculate metrics
        info_metrics = compression_efficiency(image, compressed_image)
        quality_metrics = calculate_all_metrics(image, compressed_image)
        
        # Combine all metrics
        all_metrics = {**info_metrics, **quality_metrics}
        
        # Add derived metrics
        compression_ratio = (image.size * 8) / (info_metrics['entropy_compressed'] * image.size)
        all_metrics['compression_ratio'] = compression_ratio
        all_metrics['bit_rate'] = info_metrics['entropy_compressed']
        
        # Encode compressed image
        compressed_base64 = encode_image(compressed_image)
        
        response = {
            'compressed_image': compressed_base64,
            'metrics': {k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                       for k, v in all_metrics.items()}
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Analyze image compressibility without compression.
    
    Request JSON:
    {
        "image": "base64_encoded_image"
    }
    
    Response JSON:
    {
        "analysis": {
            "entropy": float,
            "redundancy": float,
            "compressibility_score": float,
            ...
        }
    }
    """
    try:
        data = request.get_json()
        image = decode_image(data['image'])
        
        analysis = estimate_compressibility(image)
        
        response = {
            'analysis': {k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                        for k, v in analysis.items()}
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare', methods=['POST'])
def compare():
    """
    Compare multiple compression methods on the same image.
    
    Request JSON:
    {
        "image": "base64_encoded_image",
        "quality": 75,
        "methods": ["dct", "autoencoder", "vq"]
    }
    
    Response JSON:
    {
        "comparisons": {
            "dct": {
                "compressed_image": "...",
                "metrics": {...}
            },
            "autoencoder": {...},
            "vq": {...}
        }
    }
    """
    try:
        data = request.get_json()
        image = decode_image(data['image'])
        quality = int(data.get('quality', 75))
        methods = data.get('methods', ['dct', 'autoencoder', 'vq'])
        
        comparisons = {}
        
        for method in methods:
            # Compress with each method
            if method == 'dct':
                compressor = DCTCompression(quality=quality)
                compressed = compressor.compress_decompress(image)
            elif method == 'autoencoder':
                latent_dim = int(128 * (quality / 100))
                compressor = AutoencoderCompressor(latent_dim=latent_dim)
                compressed = compressor.compress_decompress(image)
            elif method == 'vq':
                codebook_size = int(256 * (quality / 100))
                codebook_size = max(16, min(512, codebook_size))
                vq = VectorQuantizer(codebook_size=codebook_size)
                vq.train_codebook([image])
                compressed = vq.compress_decompress(image)
            else:
                continue
            
            # Calculate metrics
            info_metrics = compression_efficiency(image, compressed)
            quality_metrics = calculate_all_metrics(image, compressed)
            
            comparisons[method] = {
                'compressed_image': encode_image(compressed),
                'metrics': {k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                           for k, v in {**info_metrics, **quality_metrics}.items()}
            }
        
        return jsonify({'comparisons': comparisons})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/rate-distortion', methods=['POST'])
def rate_distortion_curve():
    """
    Generate rate-distortion curve for an image.
    
    Request JSON:
    {
        "image": "base64_encoded_image",
        "method": "dct|autoencoder|vq",
        "points": 10
    }
    
    Response JSON:
    {
        "rates": [float, ...],
        "distortions": [float, ...],
        "psnr_values": [float, ...]
    }
    """
    try:
        data = request.get_json()
        image = decode_image(data['image'])
        method = data.get('method', 'dct')
        num_points = int(data.get('points', 10))
        
        rates = []
        distortions = []
        psnr_values = []
        
        # Generate curve with different quality settings
        quality_values = np.linspace(10, 100, num_points)
        
        for quality in quality_values:
            if method == 'dct':
                compressor = DCTCompression(quality=int(quality))
                compressed = compressor.compress_decompress(image)
            elif method == 'autoencoder':
                latent_dim = int(128 * (quality / 100))
                compressor = AutoencoderCompressor(latent_dim=max(16, latent_dim))
                compressed = compressor.compress_decompress(image)
            elif method == 'vq':
                codebook_size = int(256 * (quality / 100))
                codebook_size = max(16, min(512, codebook_size))
                vq = VectorQuantizer(codebook_size=codebook_size)
                vq.train_codebook([image])
                compressed = vq.compress_decompress(image)
            else:
                continue
            
            # Calculate rate and distortion
            rate = shannon_entropy(compressed)
            distortion = np.mean((image.astype(float) - compressed.astype(float)) ** 2)
            psnr = 10 * np.log10((255 ** 2) / distortion) if distortion > 0 else 100
            
            rates.append(float(rate))
            distortions.append(float(distortion))
            psnr_values.append(float(psnr))
        
        return jsonify({
            'rates': rates,
            'distortions': distortions,
            'psnr_values': psnr_values,
            'quality_values': quality_values.tolist()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
