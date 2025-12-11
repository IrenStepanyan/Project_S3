# Information Theory and Image Compression: Mathematical Foundations

## Table of Contents
1. [Introduction](#introduction)
2. [Shannon's Information Theory](#shannons-information-theory)
3. [Rate-Distortion Theory](#rate-distortion-theory)
4. [Compression Methods](#compression-methods)
5. [Information-Theoretic Metrics](#information-theoretic-metrics)
6. [Applications to Machine Learning](#applications-to-machine-learning)

---

## Introduction

This document provides a comprehensive mathematical foundation for understanding image compression through the lens of information theory. We explore how fundamental concepts from Claude Shannon's work enable efficient data compression while quantifying information loss.

### Key Questions Addressed:
- What is information, and how do we measure it?
- What is the theoretical limit of compression?
- How do we balance compression rate and quality?
- How do machine learning methods approach these limits?

---

## Shannon's Information Theory

### 1. Shannon Entropy

**Definition**: The entropy H(X) of a discrete random variable X measures the average amount of information (in bits) needed to describe it.
```
H(X) = -∑ p(xᵢ) log₂ p(xᵢ)
       i
```

where p(xᵢ) is the probability of outcome xᵢ.

**Intuition**: 
- High entropy → High uncertainty → Hard to predict → Hard to compress
- Low entropy → Low uncertainty → Easy to predict → Easy to compress

**Properties**:
1. H(X) ≥ 0 (non-negative)
2. H(X) = 0 iff X is deterministic
3. H(X) is maximized when X is uniformly distributed
4. For n equally likely outcomes: H(X) = log₂(n)

**Example for Images**:
```python
# 8-bit grayscale image
# Maximum entropy: H_max = log₂(256) = 8 bits/pixel

# Uniform distribution (white noise)
p(x) = 1/256 for all x
H = -∑(1/256)log₂(1/256) = log₂(256) = 8 bits/pixel

# Constant image (all pixels same value)
p(x₀) = 1, p(x) = 0 for x ≠ x₀
H = -1·log₂(1) = 0 bits/pixel
```

### 2. Joint Entropy

For two random variables X and Y:
```
H(X,Y) = -∑∑ p(x,y) log₂ p(x,y)
          x y
```

**Interpretation**: Total information in both variables together.

**Properties**:
- If X and Y are independent: H(X,Y) = H(X) + H(Y)
- In general: H(X,Y) ≤ H(X) + H(Y)
- Chain rule: H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)

### 3. Conditional Entropy

The entropy of X given Y:
```
H(X|Y) = H(X,Y) - H(Y)
       = -∑∑ p(x,y) log₂ p(x|y)
         x y
```

**Interpretation**: Average uncertainty about X after observing Y.

**Application to Compression**: 
- H(X|Y) represents information lost when compressing X into Y
- Perfect compression: H(X|Y) = 0 (Y completely determines X)
- No compression: H(X|Y) = H(X) (Y tells nothing about X)

### 4. Mutual Information

The mutual information between X and Y:
```
I(X;Y) = H(X) - H(X|Y)
       = H(Y) - H(Y|X)
       = H(X) + H(Y) - H(X,Y)
```

**Interpretation**: 
- How much knowing Y reduces uncertainty about X
- Information shared between X and Y
- Reduction in entropy due to knowledge of the other variable

**Properties**:
1. I(X;Y) ≥ 0 (non-negative)
2. I(X;Y) = 0 iff X and Y are independent
3. I(X;Y) = I(Y;X) (symmetric)
4. I(X;X) = H(X) (self-information)

**Venn Diagram Representation**:
```
     H(X)           H(Y)
   ┌─────────┐   ┌─────────┐
   │         │   │         │
   │  H(X|Y) │I(X;Y)│ H(Y|X) │
   │         │   │         │
   └─────────┘   └─────────┘
```

**Application to Image Compression**:
```
Original image: X
Compressed image: Y

I(X;Y) = Information preserved in compression
H(X|Y) = Information lost in compression
```

### 5. Kullback-Leibler Divergence

The KL divergence measures how one probability distribution differs from another:
```
D_KL(P||Q) = ∑ p(x) log₂(p(x)/q(x))
             x
```

**Interpretation**:
- "Distance" from distribution Q to P (not a true metric!)
- Extra bits needed if we use code optimized for Q instead of P
- D_KL(P||Q) = 0 iff P = Q
- D_KL(P||Q) ≥ 0 always (Gibbs' inequality)

**Properties**:
1. Not symmetric: D_KL(P||Q) ≠ D_KL(Q||P)
2. Not a metric: doesn't satisfy triangle inequality
3. Related to cross-entropy: H(P,Q) = H(P) + D_KL(P||Q)

**Application to Compression**:
```
P = Original pixel distribution
Q = Compressed pixel distribution

D_KL(P||Q) quantifies how much the compression
changed the statistical properties of the image
```

### 6. Cross-Entropy
```
H(P,Q) = -∑ p(x) log₂ q(x)
         x
```

**Interpretation**: Expected number of bits needed to encode samples from P using code optimized for Q.

**Relationship to Other Metrics**:
```
H(P,Q) = H(P) + D_KL(P||Q)
```

- H(P,Q) ≥ H(P), with equality iff P = Q
- Measures coding inefficiency

---

## Rate-Distortion Theory

**Central Problem**: What is the minimum rate R (bits per pixel) needed to encode a source with distortion at most D?

### 1. Rate-Distortion Function
```
R(D) = min I(X;Y̅)
       Y̅:E[d(X,Y̅)]≤D
```

where:
- X: Source (original image)
- Y̅: Reconstruction (compressed image)
- d(X,Y̅): Distortion measure (e.g., MSE)
- I(X;Y̅): Mutual information

**Interpretation**: 
- R(D) gives the theoretical minimum rate for target distortion D
- Trade-off curve: lower rate → higher distortion
- Cannot achieve both low rate AND low distortion simultaneously

### 2. Shannon's Source Coding Theorem

For lossless compression:
```
R ≥ H(X)
```

**Interpretation**: Cannot compress below the entropy (on average).

For lossy compression with distortion D:
```
R ≥ R(D)
```

### 3. Distortion Measures

Common distortion measures for images:

**Mean Squared Error (MSE)**:
```
d(x,y) = ||x - y||² = (1/N)∑(xᵢ - yᵢ)²
```

**Mean Absolute Error (MAE)**:
```
d(x,y) = ||x - y||₁ = (1/N)∑|xᵢ - yᵢ|
```

**Perceptual Distortion** (e.g., SSIM):
```
SSIM(x,y) = [l(x,y)ᵅ][c(x,y)ᵝ][s(x,y)ᵞ]
```

where l, c, s measure luminance, contrast, and structure.

### 4. Rate-Distortion Curve

The R-D curve characterizes the trade-off:
```
     Rate (R)
       ^
       |     
   Rmax|●────────────
       |  ╲
       |    ╲
       |      ╲  
       |        ╲
   R(D)|          ●
       |            ╲
       |              ╲
       |                ●
       |__________________╲___> Distortion (D)
       0                  Dmax
```

**Key Points**:
- Point (0, Dmax): No compression, maximum distortion
- Point (Rmax, 0): Lossless, no distortion, maximum rate
- Optimal operation: on the curve
- Suboptimal: above the curve

### 5. Information Bottleneck

A framework for optimal lossy compression:
```
min I(X;T) - βI(T;Y)
 T
```

where:
- X: Input (image)
- T: Compressed representation (bottleneck)
- Y: Target (what we want to preserve)
- β: Trade-off parameter

**Interpretation**:
- Minimize I(X;T): compress as much as possible
- Maximize I(T;Y): preserve relevant information
- β controls the trade-off

This principle underlies many modern compression methods including autoencoders.

---

## Compression Methods

### 1. Transform Coding (DCT)

**Principle**: Transform image to domain where energy is concentrated.

**Discrete Cosine Transform (2D)**:
```
F(u,v) = (2/N)C(u)C(v) ∑∑ f(x,y) cos[(2x+1)uπ/2N] cos[(2y+1)vπ/2N]
                        x y
```

**Information Theory Connection**:
1. DCT decorrelates pixels (reduces redundancy)
2. Energy compaction: Most energy in low frequencies
3. Quantization: Controlled information loss
4. Entropy coding: Approach entropy limit

**Rate-Distortion Trade-off**:
- Quantization step size controls rate and distortion
- Larger step → Lower rate, higher distortion
- Smaller step → Higher rate, lower distortion

**Entropy Analysis**:
```
Before DCT: H(spatial) ≈ 6-7 bits/pixel
After DCT + Quantization: H(frequency) ≈ 1-3 bits/pixel
After Entropy Coding: R ≈ 0.5-2 bpp
```

### 2. Vector Quantization (VQ)

**Principle**: Approximate continuous distribution with discrete codebook.

**Codebook**: C = {c₁, c₂, ..., cₖ}

**Encoding**:
```
q(x) = argmin ||x - cᵢ||²
        i
```

**Rate**:
```
R = log₂(K) bits per vector
```

For block size B×B:
```
R_pixel = log₂(K) / B² bits per pixel
```

**Distortion**:
```
D = E[||X - q(X)||²]
```

**Lloyd-Max Algorithm** (optimal quantizer):
1. Nearest neighbor rule: assign x to nearest cᵢ
2. Centroid condition: cᵢ = E[X | q(X) = cᵢ]

**Information Theory**:
- Codebook size K determines maximum entropy: H ≤ log₂(K)
- Optimal codebook minimizes expected distortion
- Related to rate-distortion theory

### 3. Autoencoder Compression

**Architecture**:
```
Encoder: x → z = fₑ(x; θₑ)    [Image → Latent]
Decoder: z → x̂ = fₐ(z; θₐ)    [Latent → Reconstruction]
```

**Loss Function**:
```
L = D(x, x̂) + λR(z)
```

where:
- D: Distortion (e.g., MSE, perceptual loss)
- R: Rate term (entropy of latent code)
- λ: Lagrange multiplier

**Information Bottleneck Interpretation**:
```
Minimize: I(X;Z) [compression]
Maximize: I(Z;X̂) [reconstruction quality]
```

**Variational Autoencoder (VAE)**:

Assume latent distribution:
```
q(z|x) ≈ p(z|x)
```

ELBO (Evidence Lower BOund):
```
L = E_q[log p(x|z)] - D_KL(q(z|x)||p(z))
```

First term: Reconstruction quality
Second term: Rate constraint (KL to prior)

**Rate-Distortion in VAE**:
```
Rate: R = I(X;Z) ≈ D_KL(q(z|x)||p(z))
Distortion: D = E[d(X, X̂)]
```

**Advantages**:
- Learned representations (optimal for training data)
- End-to-end optimization
- Can incorporate perceptual losses
- Flexible architecture

### 4. Entropy Coding

After quantization, use entropy coding to approach H(X):

**Huffman Coding**:
- Optimal for symbol-by-symbol coding
- Achieves H(X) ≤ R < H(X) + 1

**Arithmetic Coding**:
- Can achieve R arbitrarily close to H(X)
- Better for adaptive probability models

**Example**:
```
Symbol  | Probability | Huffman Code | Bits
--------|-------------|--------------|------
A       | 0.5         | 0            | 1
B       | 0.25        | 10           | 2
C       | 0.125       | 110          | 3
D       | 0.125       | 111          | 3

Expected length: 0.5(1) + 0.25(2) + 0.125(3) + 0.125(3) = 1.75 bits
Entropy: H = -[0.5log₂0.5 + 0.25log₂0.25 + 2(0.125log₂0.125)] = 1.75 bits
```

Perfect match! (Rare case where Huffman is optimal)

---

## Information-Theoretic Metrics

### 1. Entropy Reduction
```
ΔH = H(X) - H(Y)
```

**Interpretation**: 
- Information removed by compression
- Positive ΔH → Compression achieved
- ΔH = 0 → No compression
- Maximum ΔH = H(X) (extreme compression to constant)

### 2. Compression Ratio
```
CR = H(X) / H(Y)
```

or in terms of file sizes:
```
CR = Size(original) / Size(compressed)
```

**Typical Values**:
- Lossless: 2:1 to 4:1
- Lossy (high quality): 10:1 to 20:1
- Lossy (medium quality): 20:1 to 50:1
- Lossy (low quality): 50:1 to 100:1

### 3. Information Preservation Ratio
```
IPR = I(X;Y) / H(X)
```

**Interpretation**:
- IPR = 1: Perfect information preservation
- IPR = 0: No information preserved
- Measures how much original information remains

### 4. Efficiency Metrics

**Coding Efficiency**:
```
η = H(X) / R_actual
```

where R_actual is actual bits used.

**Rate-Distortion Efficiency**:
```
η_RD = R(D) / R_actual
```

Measures how close we are to theoretical limit.

### 5. Redundancy
```
Redundancy = R_max - H(X)
```

For 8-bit images:
```
Redundancy = 8 - H(X) bits/pixel
```

**Interpretation**: 
- How much "extra" information beyond minimum
- High redundancy → High compression potential
- Low redundancy → Already efficient, hard to compress

---

## Applications to Machine Learning

### 1. Neural Network Compression

**Weight Quantization**:

Information theory guides quantization strategies:
```
Q: R^d → {c₁, ..., cₖ}
R = log₂(K) bits per weight
```

Trade-off:
- Fewer bits (low K) → Smaller model, possibly worse accuracy
- More bits (high K) → Larger model, better accuracy

### 2. Feature Learning

**Information Bottleneck in Deep Learning**:

Each layer i can be viewed as:
```
X → T₁ → T₂ → ... → Tₙ → Y
```

Optimal: maximize I(Tᵢ;Y) while minimizing I(X;Tᵢ)

**Analysis**:
- Early layers: high I(X;Tᵢ), preserve details
- Middle layers: balance compression and information
- Late layers: low I(X;Tᵢ), high I(Tᵢ;Y), compressed but task-relevant

### 3. Generative Models

**VAE Information Theory**:
```
ELBO = E[log p(x|z)] - D_KL(q(z|x)||p(z))
     = -Reconstruction Loss - Rate
```

**GAN Connection**:

Implicit density modeling:
- Generator learns to compress noise → image
- Discriminator estimates divergence between distributions

### 4. Optimal Compression with Neural Networks

**Learned Image Compression**:

Modern approach:
```
Encoder: x → ŷ → quantize → ỹ
Decoder: ỹ → x̂

Optimize: D(x, x̂) + λR(ỹ)
```

where R(ỹ) is estimated entropy.

**Advantages over Traditional**:
- Learned transforms (vs. fixed DCT)
- Nonlinear (vs. linear transform coding)
- Context-adaptive
- End-to-end rate-distortion optimization

**State-of-the-art Results**:
- Ballé et al. (2018): Outperforms JPEG, comparable to JPEG2000
- Minnen et al. (2020): Surpasses H.264/5 for images

---

## Summary

### Key Takeaways:

1. **Entropy measures information**: 
   - H(X) = theoretical minimum bits needed
   - Cannot compress below entropy (losslessly)

2. **Rate-Distortion trade-off is fundamental**:
   - R(D): minimum rate for distortion D
   - Cannot achieve arbitrarily low rate AND distortion

3. **Mutual Information quantifies preservation**:
   - I(X;Y): information kept in compression
   - Maximize I(X;Y) while minimizing rate

4. **Transform coding exploits redundancy**:
   - Decorrelate data
   - Concentrate energy
   - Quantize less important components

5. **ML approaches learn optimal representations**:
   - Autoencoders: end-to-end optimization
   - VAE: explicit rate-distortion trade-off
   - Modern methods outperform traditional codecs

### Fundamental Limits:
```
Lossless: R ≥ H(X)
Lossy: R ≥ R(D)
```

No compression method can beat these limits (on average).

### Practical Compression:
```
Original → Transform → Quantize → Entropy Code → Compressed
   (H≈7 bpp)    (decorrelate)  (lose info)  (approach H)  (1-2 bpp)
```

### The Future:

Neural compression methods are approaching theoretical limits while providing better perceptual quality through learned representations and end-to-end optimization.

---

## References

1. Shannon, C. E. (1948). "A Mathematical Theory of Communication"
2. Cover, T. M., & Thomas, J. A. (2006). "Elements of Information Theory"
3. Ballé, J., et al. (2018). "Variational image compression with a scale hyperprior"
4. Tishby, N., & Zaslavsky, N. (2015). "Deep learning and the information bottleneck principle"
5. MacKay, D. J. C. (2003). "Information Theory, Inference, and Learning Algorithms"