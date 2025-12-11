# InfoTheory-Driven Image Compression Explorer

A comprehensive educational project exploring the intersection of **Information Theory** and **Machine Learning** for image compression. This project implements multiple compression algorithms and provides real-time analysis of key information-theoretic metrics.

## 🎯 Project Overview

This project demonstrates how information theory principles guide machine learning approaches to image compression. It includes:

- **Three compression methods**: DCT (JPEG-style), Autoencoder-based, and Vector Quantization
- **Real-time metrics**: Entropy, Mutual Information, KL Divergence, PSNR, and more
- **Interactive visualization**: Compare methods and see information loss quantitatively
- **Educational focus**: Detailed mathematical explanations of each concept

## 📊 Key Features

### Compression Methods
1. **DCT (Discrete Cosine Transform)**: Classical frequency-domain compression
2. **Autoencoder**: Neural network-based learned compression
3. **Vector Quantization**: Codebook-based compression with clustering

### Information Theory Metrics
- **Shannon Entropy**: Measures information content (bits/symbol)
- **Mutual Information**: Quantifies information preservation
- **KL Divergence**: Measures distribution difference
- **Rate-Distortion**: Plots compression efficiency
- **PSNR/MSE**: Quality metrics

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
pip
virtualenv (recommended)
```

### Setup
```bash
# Clone the repository
git clone https://github.com/StepanyanIren/infotheory-compression.git
cd infotheory-compression

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure
```
infotheory-compression/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dct_compression.py      # DCT-based compression
│   │   ├── autoencoder.py          # Neural network compression
│   │   └── vector_quantization.py  # VQ-based compression
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── information_metrics.py  # Entropy, MI, KL divergence
│   │   └── quality_metrics.py      # PSNR, SSIM, MSE
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_processing.py     # Image I/O and preprocessing
│   │   └── visualization.py        # Plotting and visualization
│   └── app.py                      # Flask web application
├── notebooks/
│   ├── 01_theory_introduction.ipynb
│   ├── 02_compression_methods.ipynb
│   └── 03_experiments.ipynb
├── docs/
│   ├── THEORY.md                   # Mathematical foundations
│   ├── ARCHITECTURE.md             # System design
│   └── EXPERIMENTS.md              # Results and analysis
├── tests/
│   ├── test_compression.py
│   └── test_metrics.py
├── requirements.txt
├── setup.py
└── README.md
```

## 🎓 Usage

### Command Line Interface
```bash
# Basic compression
python -m src.app compress --input image.jpg --method dct --quality 50

# Compare methods
python -m src.app compare --input image.jpg --output comparison.png

# Generate metrics report
python -m src.app analyze --input image.jpg --report report.json
```

### Web Interface
```bash
# Start Flask server
python src/app.py

# Open browser to http://localhost:5000
```

### Python API
```python
from src.models import DCTCompression, Autoencoder
from src.metrics import calculate_entropy, mutual_information

# Load and compress image
compressor = DCTCompression(quality=75)
compressed = compressor.compress('image.jpg')

# Calculate metrics
original_entropy = calculate_entropy(original_image)
compressed_entropy = calculate_entropy(compressed)
mi = mutual_information(original_image, compressed)

print(f"Entropy reduction: {original_entropy - compressed_entropy:.2f} bits")
```

## 📖 Theoretical Background

### Shannon Entropy
Entropy H(X) measures the average information content:
```
H(X) = -Σ p(x) log₂ p(x)
```

Where p(x) is the probability distribution of pixel values.

### Mutual Information
MI measures how much knowing one variable tells us about another:
```
I(X;Y) = H(X) + H(Y) - H(X,Y)
```

### Rate-Distortion Theory
Optimal trade-off between compression rate R and distortion D:
```
R(D) = min I(X;Y) subject to E[d(X,Y)] ≤ D
```

See [THEORY.md](docs/THEORY.md) for complete mathematical derivations.

## 🧪 Experiments

### Benchmark Results

| Method | PSNR (dB) | Compression Ratio | Encoding Time (ms) |
|--------|-----------|-------------------|-------------------|
| DCT    | 32.5      | 10.2:1           | 45                |
| Autoencoder | 34.1 | 15.3:1          | 120               |
| VQ     | 30.8      | 8.5:1            | 35                |

### Rate-Distortion Curves
![Rate-Distortion](docs/images/rate_distortion.png)

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🔗 References

1. Shannon, C. E. (1948). "A Mathematical Theory of Communication"
2. Cover, T. M., & Thomas, J. A. (2006). "Elements of Information Theory"
3. Ballé, J., et al. (2018). "Variational image compression with a scale hyperprior"

## 👥 Authors

- Iren Stepanyan - Initial work

## 🙏 Acknowledgments

- University Professor/Advisor: **Mariam Harutyunyan**
- Information Theory course materials
- PyTorch and TensorFlow communities