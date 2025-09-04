# Self-Attention Profiling

This repository contains code for profiling the computational complexity, memory usage, and wall clock time of self-attention mechanisms as a function of input sequence length.

## Repository Structure

```
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── self_attention_profiler.py         # Main profiling script
```

## Overview

The profiling script implements a multi-headed self-attention mechanism and measures:

1. **Computational Complexity (FLOPs)**: Theoretical floating-point operations count
2. **Memory Usage**: Peak memory consumption during computation
3. **Wall Clock Time**: Actual execution time

These metrics are measured for sequence lengths: 10, 100, 1K, 10K on both CPU and GPU devices.

## Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy
- Matplotlib
- Seaborn
- psutil (for memory monitoring)

## Installation & Setup

### For Google Colab

1. **Upload the files**:

   - Upload `self_attention_profiler.py` and `requirements.txt` to your Colab environment

2. **Install required packages** (run this in the first cell):

```python
!pip install -r requirements.txt
```

3. **Run the profiling**:

```python
# In a new cell, run:
!python self_attention_profiler.py
```

### For Local Environment

1. **Clone or download** this repository
2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run the script**:

```bash
python self_attention_profiler.py
```

## Usage Instructions

### Step-by-Step for Google Colab

1. **Create a new Colab notebook**
2. **Cell 1**: Install dependencies

```python
!pip install torch torchvision torchaudio matplotlib seaborn psutil numpy
```

3. **Cell 2**: Upload and run the script

```python
# Upload self_attention_profiler.py to Colab, then run:
!python self_attention_profiler.py
```

4. **Cell 3**: Display results

```python
from IPython.display import Image, display
import matplotlib.pyplot as plt

# Display the generated plots
display(Image('self_attention_profiling.png'))
display(Image('flops_comparison.png'))
display(Image('time_comparison.png'))
display(Image('memory_comparison.png'))
```

### Alternative: Run directly in Colab cells

You can also copy the script content directly into Colab cells:

1. **Cell 1**: Install packages (as above)
2. **Cell 2**: Copy the entire `self_attention_profiler.py` content
3. **Cell 3**: Run `main()` function

## Expected Output

The script will generate:

- **Console output**: Progress updates and summary statistics
- **Plot files**: High-quality PNG and PDF plots suitable for academic papers
- **Performance metrics**: Detailed timing, memory, and FLOP measurements

### Sample Console Output

```
Self-Attention Profiling Experiment
===================================
GPU available: Tesla T4
Profiling on CPU...
  Testing sequence length: 10
  Testing sequence length: 100
  Testing sequence length: 1000
  Testing sequence length: 10000
Profiling on GPU...
  Testing sequence length: 10
  Testing sequence length: 100
  Testing sequence length: 1000
  Testing sequence length: 10000
Creating plots...
```

## Key Features

- **Multi-headed self-attention**: Standard transformer attention mechanism
- **Comprehensive profiling**: FLOPs, memory, and time measurements
- **Error bars**: Standard error calculations for statistical significance
- **Publication-ready plots**: High-quality figures with proper formatting
- **CPU/GPU comparison**: Performance analysis across devices
- **Scalability analysis**: Performance trends with sequence length

## Technical Details

### Self-Attention Implementation

- Model dimension: 512
- Number of heads: 8
- Batch size: 1
- No bias in linear layers (standard practice)

### Profiling Methodology

- **Multiple runs**: 10 runs per configuration for statistical significance
- **Standard error**: Error bars represent standard error (std/√n)
- **Memory tracking**: Peak memory usage during forward pass
- **FLOP counting**: Theoretical operation count for matrix multiplications
- **Warm-up**: GPU synchronization for accurate timing

### Plot Characteristics

- **Log-log scales**: For better visualization of scaling trends
- **Error bars**: Standard error representation
- **Academic styling**: Serif fonts, proper grid, high DPI
- **Multiple formats**: PNG for viewing, PDF for LaTeX inclusion

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or maximum sequence length
2. **Missing packages**: Ensure all requirements are installed
3. **Slow execution**: Expected for large sequence lengths, especially on CPU

### Performance Notes

- **CPU timing**: May be slow for sequence length 10K
- **GPU requirement**: CUDA-compatible GPU recommended for best results
- **Memory usage**: Large sequence lengths require significant memory

## Expected Results Interpretation

The profiling should show:

1. **Quadratic scaling**: Time and memory scale as O(n²) with sequence length
2. **GPU acceleration**: Significant speedup for larger sequence lengths
3. **Memory efficiency**: GPU typically more memory-efficient for large sequences
4. **FLOP consistency**: Similar FLOP counts across devices (algorithm unchanged)

## Repository Information

- **Purpose**: Academic assignment for NLP Advanced course
- **Focus**: Transformer self-attention profiling and analysis
- **Output**: Research-quality plots and performance metrics
- **Compatibility**: Google Colab and local Python environments

## Contact

For questions or issues with this profiling code, please contact the author via email or linkedin.

---

**Note**: This code is designed for educational purposes as part of a homework assignment. The implementations prioritize clarity and correctness over maximum optimization.
