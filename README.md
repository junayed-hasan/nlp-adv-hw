# NLP Advanced Homework

This repository contains code for various NLP experiments and analyses including self-attention profiling, transformer fine-tuning, and information retrieval systems.

**Author:** Mohammad Junayed Hasan  
**Date:** September 2025  
**GitHub:** https://github.com/junayed-hasan/self-attention-profiling

## Repository Structure

```
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── self_attention_profiler.py         # Self-attention profiling script
├── perplexity_sampling_analysis.py    # Perplexity and sampling analysis script
├── sgd_visualization.py               # SGD behavior visualization script
├── modernbert_classifier.py           # ModernBERT head-only fine-tuning classifier script
├── modernbert_lora_classifier.py      # ModernBERT LoRA fine-tuning classifier script
├── scifact_ir_system.py               # SciFact Information Retrieval system using OpenAI embeddings
```

## Experiments

### 1. Self-Attention Profiling

Profiles the computational complexity, memory usage, and wall clock time of self-attention mechanisms.

**Metrics measured:**
- Computational Complexity (FLOPs)
- Memory Usage (MB)
- Wall Clock Time (ms)

**Sequence lengths tested:** 10, 100, 1K, 10K on both CPU and GPU

### 2. Perplexity and Sampling Analysis

Analyzes perplexity of coherent vs shuffled text and compares different text generation sampling strategies.

**Components:**
- Perplexity analysis of original vs shuffled text
- Text generation with different temperature settings
- Sampling strategy comparison (greedy vs temperature sampling)

### 3. SGD Behavior Visualization

Visualizes the optimization behavior of SGD with different parameter settings on 2D functions.

**Components:**
- Momentum effect analysis (0.0, 0.5, 0.9)
- Weight decay impact visualization
- Minimization vs maximization comparison
- Convergence speed analysis

### 4. ModernBERT Head-Only Fine-tuning

Fine-tunes only the classification head of ModernBERT on the StrategyQA dataset for binary classification while keeping the pre-trained backbone frozen.

**Features:**
- **Head-only fine-tuning**: Only classification head trainable (~1,538 parameters vs 139M total)
- **StrategyQA dataset**: Binary classification task (True/False answers to strategy questions)  
- **Modern best practices**: Latest transformers library, mixed precision, proper initialization
- **Early stopping**: Prevents overfitting with patience-based monitoring
- **Comprehensive evaluation**: Accuracy, precision, recall, F1-score, confusion matrix
- **Training visualization**: Loss curves, accuracy plots, learning rate schedule
- **Publication-quality plots**: High-resolution figures suitable for academic papers
- **Reproducible results**: Fixed seeds and deterministic training
- **Google Colab optimized**: T4 GPU compatible with memory-efficient settings

### 5. ModernBERT LoRA Fine-tuning

Implements Low-Rank Adaptation (LoRA) fine-tuning of ModernBERT on the StrategyQA dataset with the same number of trainable parameters as head-only training for fair comparison.

**Features:**
- **LoRA fine-tuning**: Low-rank adaptation with rank=1 applied to Wo modules in layer 0
- **Parameter matching**: Exactly 1,538 trainable parameters (same as head-only training)
- **Fair comparison**: Same dataset, evaluation metrics, and early stopping as head-only
- **Advanced configuration**: Smart layer targeting and parameter count validation
- **Comprehensive evaluation**: Same metrics as head-only for direct comparison
- **Publication-ready visualizations**: Green-themed plots to distinguish from head-only
- **Theory implementation**: Follows Hu et al. (2021) LoRA methodology

### 6. SciFact Information Retrieval System

Implements an information retrieval system for the SciFact dataset using pre-computed OpenAI embeddings and FAISS for efficient similarity search.

**Features:**
- **FAISS-based search**: Efficient vector similarity search with GPU acceleration
- **OpenAI embeddings**: Uses pre-computed text-embedding-ada-002 embeddings
- **Comprehensive evaluation**: MAP and MRR metrics at multiple cutoffs (@1, @10, @50)
- **SciFact dataset**: Scientific claim verification with 1.4K claims and evidence pairs
- **Flexible architecture**: Support for different FAISS index types and similarity metrics
- **Performance optimization**: GPU acceleration and normalized embeddings for cosine similarity
- **Statistical analysis**: Error bars and significance testing
- **Publication-ready results**: Formatted tables and comprehensive evaluation reports

## Requirements

- **Python 3.8+** (Recommended: 3.9 or 3.10)
- **PyTorch 2.0+** (with CUDA support for GPU acceleration)
- **Transformers 4.20+** (for ModernBERT support)
- **HuggingFace Datasets 2.0+** (for StrategyQA and SciFact datasets)
- **PEFT 0.7+** (for LoRA implementation)
- **FAISS 1.7+** (for efficient vector search)
- **NumPy 1.19+**
- **Matplotlib 3.3+** (for publication-quality plots)
- **Seaborn 0.11+** (for statistical visualizations)
- **scikit-learn 1.0+** (for evaluation metrics)
- **tqdm 4.64+** (for progress bars)
- **accelerate 0.20+** (for optimized training)
- **tokenizers 0.13+** (for fast tokenization)
- **psutil 5.8+** (for memory monitoring)

## Installation & Setup

### For Google Colab (Recommended)

#### Quick Start - All Experiments

1. **Create a new Colab notebook** and enable GPU runtime:
   - Runtime → Change runtime type → Hardware accelerator: GPU (T4 recommended)

2. **Install dependencies** (run in first cell):

```python
# Install all required packages
!pip install torch>=2.0.0 torchvision torchaudio
!pip install transformers>=4.20.0 datasets>=2.0.0 accelerate>=0.20.0
!pip install peft>=0.7.0 matplotlib seaborn scikit-learn numpy tqdm tokenizers huggingface-hub psutil scipy

# Install FAISS (try GPU version first, fallback to CPU)
try:
    !pip install faiss-gpu>=1.7.4
    print("FAISS GPU installed successfully")
except:
    !pip install faiss-cpu>=1.7.4
    print("FAISS CPU installed (GPU version not available)")
```

3. **Run specific experiments**:

```python
# For self-attention profiling:
!python self_attention_profiler.py

# For perplexity and sampling analysis:
!python perplexity_sampling_analysis.py

# For SGD behavior visualization:
!python sgd_visualization.py

# For ModernBERT head-only fine-tuning:
!python modernbert_classifier.py

# For ModernBERT LoRA fine-tuning:
!python modernbert_lora_classifier.py

# For SciFact Information Retrieval:
!python scifact_ir_system.py
```

### For Local Environment

1. **Clone or download** this repository
2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run the scripts**:

```bash
# For self-attention profiling:
python self_attention_profiler.py

# For perplexity and sampling analysis:
python perplexity_sampling_analysis.py

# For SGD behavior visualization:
python sgd_visualization.py

# For ModernBERT head-only fine-tuning:
python modernbert_classifier.py

# For ModernBERT LoRA fine-tuning:
python modernbert_lora_classifier.py

# For SciFact Information Retrieval:
python scifact_ir_system.py
```

## Expected Output

### ModernBERT Head-Only Fine-tuning

- **Test Accuracy**: ~59.3% on StrategyQA
- **Best Validation Accuracy**: ~70.0%
- **Trainable Parameters**: 1,538
- **Training Time**: ~5-6 minutes on T4 GPU
- **Outputs**: Training curves, confusion matrix, performance metrics

### ModernBERT LoRA Fine-tuning

- **Test Accuracy**: ~53.8% on StrategyQA  
- **Best Validation Accuracy**: ~62.0%
- **Trainable Parameters**: 1,538 (matching head-only)
- **Training Time**: ~12 minutes on T4 GPU
- **Outputs**: LoRA training curves, confusion matrix, comparative analysis

### SciFact Information Retrieval

- **MAP@1**: To be determined based on actual embeddings
- **MRR@1**: To be determined based on actual embeddings
- **Dataset**: 1.4K scientific claims with evidence pairs
- **Search Speed**: Sub-second retrieval with FAISS
- **Outputs**: Comprehensive IR metrics table, performance analysis

## Key Features

- **Multi-modal analysis**: Self-attention profiling, language model analysis, fine-tuning, and information retrieval
- **Comprehensive evaluation**: FLOPs, memory, timing measurements, and IR metrics (MAP, MRR)
- **Modern architectures**: ModernBERT with both head-only and LoRA fine-tuning approaches
- **Efficient search**: FAISS-based vector similarity search with GPU acceleration
- **Publication-ready outputs**: High-quality figures, formatted tables, and detailed reports
- **Reproducible results**: Fixed seeds and deterministic training across all experiments
- **Google Colab optimized**: T4 GPU compatible with memory-efficient settings

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or maximum sequence length
2. **Missing packages**: Ensure all requirements are installed via `pip install -r requirements.txt`
3. **Missing embeddings**: Download OpenAI embeddings from provided Google Drive links for IR system
4. **Dataset download**: First run may take time to download datasets (StrategyQA, SciFact)
5. **Model loading**: ModernBERT model download may take several minutes initially

## Repository Information

- **Author**: Mohammad Junayed Hasan
- **Date**: September 2025
- **GitHub**: https://github.com/junayed-hasan/self-attention-profiling
- **Purpose**: Academic assignments for NLP Advanced course
- **Focus**: Transformer analysis, fine-tuning strategies, and information retrieval systems
- **Output**: Research-quality analysis, performance metrics, and comparative evaluations

---

**Note**: This code is designed for educational purposes as part of homework assignments. The implementations prioritize clarity, correctness, and reproducibility.
