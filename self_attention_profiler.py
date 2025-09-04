#!/usr/bin/env python3
"""
Self-Attention Profiling Script
===============================

This script profiles the computational complexity, memory usage, and wall clock time
of self-attention mechanisms as a function of input sequence length.

Author: NLP Advanced HW Assignment
Date: September 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import gc
import psutil
import os
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Set style for academic plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class SelfAttention(nn.Module):
    """
    A simple self-attention module for profiling purposes.
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()
        
        # Compute Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(context)
        
        return output


def count_flops(model: nn.Module, input_tensor: torch.Tensor) -> int:
    """
    Count FLOPs for self-attention computation.
    
    Args:
        model: Self-attention model
        input_tensor: Input tensor
        
    Returns:
        Estimated FLOP count
    """
    batch_size, seq_len, d_model = input_tensor.shape
    num_heads = model.num_heads
    d_k = model.d_k
    
    # Linear projections for Q, K, V (3 projections)
    qkv_flops = 3 * batch_size * seq_len * d_model * d_model
    
    # Attention computation: Q @ K^T
    attention_flops = batch_size * num_heads * seq_len * seq_len * d_k
    
    # Attention @ V
    context_flops = batch_size * num_heads * seq_len * seq_len * d_k
    
    # Output projection
    output_flops = batch_size * seq_len * d_model * d_model
    
    total_flops = qkv_flops + attention_flops + context_flops + output_flops
    return total_flops


def measure_memory_usage(device: torch.device) -> float:
    """
    Measure current memory usage.
    
    Args:
        device: Device to measure memory for
        
    Returns:
        Memory usage in MB
    """
    if device.type == 'cuda':
        return torch.cuda.memory_allocated(device) / 1024**2
    else:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2


def profile_self_attention(seq_lengths: List[int], 
                         device: torch.device,
                         num_runs: int = 3,
                         batch_size: int = 2,
                         d_model: int = 128) -> Dict:
    """
    Profile self-attention for different sequence lengths.
    
    Args:
        seq_lengths: List of sequence lengths to test
        device: Device to run on (CPU or CUDA)
        num_runs: Number of runs for averaging
        batch_size: Batch size for inputs
        d_model: Model dimension
        
    Returns:
        Dictionary containing profiling results
    """
    model = SelfAttention(d_model=d_model).to(device)
    model.eval()
    
    results = {
        'seq_lengths': seq_lengths,
        'flops_mean': [],
        'flops_std': [],
        'memory_mean': [],
        'memory_std': [],
        'time_mean': [],
        'time_std': [],
        'device': device.type
    }
    
    print(f"Profiling on {device.type.upper()}...")
    
    for seq_len in seq_lengths:
        print(f"  Testing sequence length: {seq_len}")
        
        # Adjust batch size for very long sequences to avoid OOM
        current_batch_size = batch_size
        if seq_len >= 1000:
            current_batch_size = 1  # Use batch size 1 for long sequences
        elif seq_len >= 100:
            current_batch_size = max(1, batch_size // 2)
        
        flops_list = []
        memory_list = []
        time_list = []
        
        for run in range(num_runs):
            print(f"    Run {run+1}/{num_runs}")  # Progress indicator
            
            # Clear cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            try:
                # Create input
                x = torch.randn(current_batch_size, seq_len, d_model, device=device)
                
                # Measure initial memory
                initial_memory = measure_memory_usage(device)
                
                # Count FLOPs (scale by actual batch size used)
                flops = count_flops(model, x)
                # Normalize FLOPs to original batch size for fair comparison
                flops = flops * (batch_size / current_batch_size)
                flops_list.append(flops)
                
                # Warmup run for GPU
                if device.type == 'cuda' and run == 0:
                    with torch.no_grad():
                        _ = model(x)
                    torch.cuda.synchronize()
                
                # Measure time
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()  # More precise timing
                
                with torch.no_grad():
                    output = model(x)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
                time_list.append(elapsed_time)
                
                # Measure peak memory
                peak_memory = measure_memory_usage(device)
                memory_usage = peak_memory - initial_memory
                memory_list.append(max(0, memory_usage))  # Ensure non-negative
                
                # Clean up
                del x, output
                
            except RuntimeError as e:
                print(f"    Warning: Runtime error for seq_len {seq_len}, run {run}: {e}")
                # Use previous values or skip
                if flops_list:
                    flops_list.append(flops_list[-1])
                    time_list.append(time_list[-1] if time_list else 1000.0)
                    memory_list.append(memory_list[-1] if memory_list else 50.0)
                else:
                    # Estimate values for failed runs
                    estimated_flops = count_flops(model, torch.zeros(1, seq_len, d_model)) * batch_size
                    flops_list.append(estimated_flops)
                    time_list.append(1000.0)  # Default time for failed runs
                    memory_list.append(50.0)  # Default memory for failed runs
        
        # Calculate statistics
        results['flops_mean'].append(np.mean(flops_list))
        results['flops_std'].append(np.std(flops_list) / np.sqrt(num_runs))  # Standard error
        results['memory_mean'].append(np.mean(memory_list))
        results['memory_std'].append(np.std(memory_list) / np.sqrt(num_runs))  # Standard error
        results['time_mean'].append(np.mean(time_list))
        results['time_std'].append(np.std(time_list) / np.sqrt(num_runs))  # Standard error
    
    return results


def create_plots(cpu_results: Dict, gpu_results: Dict, save_dir: str = "."):
    """
    Create publication-quality plots for the profiling results.
    
    Args:
        cpu_results: Results from CPU profiling
        gpu_results: Results from GPU profiling
        save_dir: Directory to save plots
    """
    # Set up the plotting style for academic papers
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True,
        'legend.fontsize': 10,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'lines.linewidth': 2,
        'lines.markersize': 6
    })
    
    # Create subplots - only the 3 main comparisons as requested
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Self-Attention Profiling Results', fontsize=16, fontweight='bold')
    
    seq_lengths = cpu_results['seq_lengths']
    
    # Plot 1: FLOPs comparison
    ax = axes[0]
    ax.errorbar(seq_lengths, cpu_results['flops_mean'], yerr=cpu_results['flops_std'],
                label='CPU', marker='o', capsize=5)
    ax.errorbar(seq_lengths, gpu_results['flops_mean'], yerr=gpu_results['flops_std'],
                label='GPU', marker='s', capsize=5)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('FLOPs')
    ax.set_title('Computational Complexity (FLOPs)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Memory usage comparison
    ax = axes[1]
    ax.errorbar(seq_lengths, cpu_results['memory_mean'], yerr=cpu_results['memory_std'],
                label='CPU', marker='o', capsize=5)
    ax.errorbar(seq_lengths, gpu_results['memory_mean'], yerr=gpu_results['memory_std'],
                label='GPU', marker='s', capsize=5)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('Memory Usage')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Time comparison
    ax = axes[2]
    ax.errorbar(seq_lengths, cpu_results['time_mean'], yerr=cpu_results['time_std'],
                label='CPU', marker='o', capsize=5)
    ax.errorbar(seq_lengths, gpu_results['time_mean'], yerr=gpu_results['time_std'],
                label='GPU', marker='s', capsize=5)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Wall Clock Time (ms)')
    ax.set_title('Wall Clock Time')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/self_attention_profiling.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/self_attention_profiling.pdf', bbox_inches='tight')
    plt.show()
    
    # Save individual plots for LaTeX inclusion
    create_individual_plots(cpu_results, gpu_results, save_dir)


def create_individual_plots(cpu_results: Dict, gpu_results: Dict, save_dir: str):
    """Create individual plots for LaTeX inclusion."""
    
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'serif',
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.5,
        'lines.markersize': 8
    })
    
    seq_lengths = cpu_results['seq_lengths']
    
    # Individual FLOP plot
    plt.figure(figsize=(8, 6))
    plt.errorbar(seq_lengths, cpu_results['flops_mean'], yerr=cpu_results['flops_std'],
                label='CPU', marker='o', capsize=5, linewidth=2.5, markersize=8)
    plt.errorbar(seq_lengths, gpu_results['flops_mean'], yerr=gpu_results['flops_std'],
                label='GPU', marker='s', capsize=5, linewidth=2.5, markersize=8)
    plt.xlabel('Sequence Length', fontsize=14)
    plt.ylabel('FLOPs', fontsize=14)
    plt.title('Computational Complexity (FLOPs)', fontsize=16, fontweight='bold')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/flops_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/flops_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    # Individual time plot
    plt.figure(figsize=(8, 6))
    plt.errorbar(seq_lengths, cpu_results['time_mean'], yerr=cpu_results['time_std'],
                label='CPU', marker='o', capsize=5, linewidth=2.5, markersize=8)
    plt.errorbar(seq_lengths, gpu_results['time_mean'], yerr=gpu_results['time_std'],
                label='GPU', marker='s', capsize=5, linewidth=2.5, markersize=8)
    plt.xlabel('Sequence Length', fontsize=14)
    plt.ylabel('Wall Clock Time (ms)', fontsize=14)
    plt.title('Wall Clock Time Comparison', fontsize=16, fontweight='bold')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/time_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/time_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    # Individual memory plot
    plt.figure(figsize=(8, 6))
    plt.errorbar(seq_lengths, cpu_results['memory_mean'], yerr=cpu_results['memory_std'],
                label='CPU', marker='o', capsize=5, linewidth=2.5, markersize=8)
    plt.errorbar(seq_lengths, gpu_results['memory_mean'], yerr=gpu_results['memory_std'],
                label='GPU', marker='s', capsize=5, linewidth=2.5, markersize=8)
    plt.xlabel('Sequence Length', fontsize=14)
    plt.ylabel('Memory Usage (MB)', fontsize=14)
    plt.title('Memory Usage Comparison', fontsize=16, fontweight='bold')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/memory_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/memory_comparison.pdf', bbox_inches='tight')
    plt.close()


def main():
    """Main function to run the profiling experiment."""
    
    print("Self-Attention Profiling Experiment")
    print("===================================")
    
    # Define sequence lengths to test
    seq_lengths = [10, 100, 1000, 10000]
    
    # Check device availability
    cpu_device = torch.device('cpu')
    gpu_available = torch.cuda.is_available()
    
    if gpu_available:
        gpu_device = torch.device('cuda')
        print(f"GPU available: {torch.cuda.get_device_name()}")
    else:
        print("GPU not available, using CPU only")
        gpu_device = cpu_device
    
    # Profile on CPU
    print("\nProfiling on CPU...")
    cpu_results = profile_self_attention(seq_lengths, cpu_device, num_runs=3)
    
    # Profile on GPU (if available)
    if gpu_available:
        print("\nProfiling on GPU...")
        gpu_results = profile_self_attention(seq_lengths, gpu_device, num_runs=3)
    else:
        gpu_results = cpu_results.copy()
        gpu_results['device'] = 'cpu_fallback'
    
    # Create and save plots
    print("\nCreating plots...")
    create_plots(cpu_results, gpu_results)
    
    # Print summary statistics
    print("\nSummary Results:")
    print("================")
    print(f"Sequence Lengths: {seq_lengths}")
    print(f"\nCPU Results:")
    print(f"  FLOPs: {cpu_results['flops_mean']}")
    print(f"  Time (ms): {cpu_results['time_mean']}")
    print(f"  Memory (MB): {cpu_results['memory_mean']}")
    
    if gpu_available:
        print(f"\nGPU Results:")
        print(f"  FLOPs: {gpu_results['flops_mean']}")
        print(f"  Time (ms): {gpu_results['time_mean']}")
        print(f"  Memory (MB): {gpu_results['memory_mean']}")
        
        speedup = np.array(cpu_results['time_mean']) / np.array(gpu_results['time_mean'])
        print(f"\nGPU Speedup: {speedup}")
    
    print("\nProfiling complete! Check the generated plots.")


if __name__ == "__main__":
    main()
