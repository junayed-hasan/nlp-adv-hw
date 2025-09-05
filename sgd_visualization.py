#!/usr/bin/env python3
"""
SGD Behavior Visualization Script
=================================

This script visualizes the behavior of SGD optimization with different parameters
including momentum, weight decay, and maximization vs minimization.

Author: NLP Advanced HW Assignment
Date: September 2025
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set publication-quality plotting parameters
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
    'lines.markersize': 4,
    'figure.dpi': 150
})

def f_minimize(x, y):
    """Minimization function: f(x,y) = x^2 + y^2"""
    return x**2 + y**2

def f_maximize(x, y):
    """Maximization function: f(x,y) = -x^2 - y^2"""
    return -x**2 - y**2

def compute_gradient_minimize(x, y):
    """Gradient of f(x,y) = x^2 + y^2"""
    return 2*x, 2*y

def compute_gradient_maximize(x, y):
    """Gradient of f(x,y) = -x^2 - y^2"""
    return -2*x, -2*y

def optimize_function(func_type='minimize', momentum=0.0, weight_decay=0.0, 
                     maximize=False, lr=0.1, num_steps=50, start_point=(2.0, 1.5)):
    """
    Optimize a 2D function using SGD and track the trajectory.
    
    Args:
        func_type: 'minimize' or 'maximize'
        momentum: Momentum coefficient
        weight_decay: Weight decay coefficient
        maximize: Whether to maximize the function
        lr: Learning rate
        num_steps: Number of optimization steps
        start_point: Starting point (x, y)
        
    Returns:
        trajectory: List of (x, y) points
        losses: List of function values
    """
    # Initialize parameters
    x = torch.tensor(start_point[0], requires_grad=True, dtype=torch.float32)
    y = torch.tensor(start_point[1], requires_grad=True, dtype=torch.float32)
    
    # Initialize optimizer
    optimizer = optim.SGD([x, y], lr=lr, momentum=momentum, 
                         weight_decay=weight_decay, maximize=maximize)
    
    trajectory = []
    losses = []
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Compute loss
        if func_type == 'minimize':
            loss = f_minimize(x, y)
        else:
            loss = f_maximize(x, y)
        
        # Store current position and loss
        trajectory.append((x.item(), y.item()))
        losses.append(loss.item())
        
        # Backward pass
        loss.backward()
        
        # Optimization step
        optimizer.step()
    
    return trajectory, losses

def create_contour_plot(func_type='minimize', x_range=(-3, 3), y_range=(-3, 3), num_points=100):
    """Create contour plot for the function."""
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    
    if func_type == 'minimize':
        Z = f_minimize(X, Y)
        levels = np.logspace(-1, 1.5, 20)  # Logarithmic levels for better visualization
    else:
        Z = f_maximize(X, Y)
        levels = np.linspace(-10, 0, 20)
    
    return X, Y, Z, levels

def plot_momentum_comparison():
    """Plot comparison of different momentum values."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    momentum_values = [0.0, 0.5, 0.9]
    colors = ['red', 'blue', 'green']
    
    for idx, (momentum, color) in enumerate(zip(momentum_values, colors)):
        ax = axes[idx]
        
        # Create contour plot
        X, Y, Z, levels = create_contour_plot('minimize')
        contour = ax.contour(X, Y, Z, levels=levels, colors='gray', alpha=0.6, linewidths=0.8)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
        
        # Optimize and plot trajectory
        trajectory, losses = optimize_function(
            func_type='minimize',
            momentum=momentum,
            lr=0.1,
            num_steps=30
        )
        
        x_traj = [point[0] for point in trajectory]
        y_traj = [point[1] for point in trajectory]
        
        # Plot trajectory
        ax.plot(x_traj, y_traj, color=color, linewidth=2, marker='o', 
                markersize=3, alpha=0.8, label=f'Momentum={momentum}')
        
        # Mark start and end points
        ax.plot(x_traj[0], y_traj[0], 'ko', markersize=8, label='Start')
        ax.plot(x_traj[-1], y_traj[-1], 'k*', markersize=10, label='End')
        ax.plot(0, 0, 'r*', markersize=12, label='Optimum')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Momentum = {momentum}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
    
    plt.suptitle('SGD Optimization: Effect of Momentum\n$f(x,y) = x^2 + y^2$ (Minimization)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sgd_momentum_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('sgd_momentum_comparison.pdf', bbox_inches='tight')
    plt.show()

def plot_weight_decay_comparison():
    """Plot comparison with and without weight decay."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    weight_decay_values = [0.0, 0.1]
    colors = ['blue', 'orange']
    titles = ['Without Weight Decay', 'With Weight Decay (0.1)']
    
    for idx, (weight_decay, color, title) in enumerate(zip(weight_decay_values, colors, titles)):
        ax = axes[idx]
        
        # Create contour plot
        X, Y, Z, levels = create_contour_plot('minimize')
        contour = ax.contour(X, Y, Z, levels=levels, colors='gray', alpha=0.6, linewidths=0.8)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
        
        # Optimize with momentum=0.5 and varying weight decay
        trajectory, losses = optimize_function(
            func_type='minimize',
            momentum=0.5,
            weight_decay=weight_decay,
            lr=0.1,
            num_steps=30
        )
        
        x_traj = [point[0] for point in trajectory]
        y_traj = [point[1] for point in trajectory]
        
        # Plot trajectory
        ax.plot(x_traj, y_traj, color=color, linewidth=2, marker='o', 
                markersize=3, alpha=0.8, label=f'Weight Decay={weight_decay}')
        
        # Mark start and end points
        ax.plot(x_traj[0], y_traj[0], 'ko', markersize=8, label='Start')
        ax.plot(x_traj[-1], y_traj[-1], 'k*', markersize=10, label='End')
        ax.plot(0, 0, 'r*', markersize=12, label='Optimum')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
    
    plt.suptitle('SGD Optimization: Effect of Weight Decay\n$f(x,y) = x^2 + y^2$, Momentum=0.5', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sgd_weight_decay_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('sgd_weight_decay_comparison.pdf', bbox_inches='tight')
    plt.show()

def plot_maximize_comparison():
    """Plot comparison of minimization vs maximization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    settings = [
        {'func_type': 'minimize', 'maximize': False, 'title': 'Minimization: $f(x,y) = x^2 + y^2$'},
        {'func_type': 'maximize', 'maximize': True, 'title': 'Maximization: $f(x,y) = -x^2 - y^2$'}
    ]
    colors = ['blue', 'red']
    
    for idx, (setting, color) in enumerate(zip(settings, colors)):
        ax = axes[idx]
        
        # Create contour plot
        X, Y, Z, levels = create_contour_plot(setting['func_type'])
        
        if setting['func_type'] == 'minimize':
            contour = ax.contour(X, Y, Z, levels=levels, colors='gray', alpha=0.6, linewidths=0.8)
        else:
            contour = ax.contour(X, Y, Z, levels=levels, colors='gray', alpha=0.6, linewidths=0.8)
        
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
        
        # Optimize
        trajectory, losses = optimize_function(
            func_type=setting['func_type'],
            momentum=0.5,
            weight_decay=0.0,
            maximize=setting['maximize'],
            lr=0.1,
            num_steps=30,
            start_point=(1.5, 1.0)  # Closer start for maximization case
        )
        
        x_traj = [point[0] for point in trajectory]
        y_traj = [point[1] for point in trajectory]
        
        # Plot trajectory
        ax.plot(x_traj, y_traj, color=color, linewidth=2, marker='o', 
                markersize=3, alpha=0.8, label='Optimization Path')
        
        # Mark start and end points
        ax.plot(x_traj[0], y_traj[0], 'ko', markersize=8, label='Start')
        ax.plot(x_traj[-1], y_traj[-1], 'k*', markersize=10, label='End')
        
        if setting['func_type'] == 'minimize':
            ax.plot(0, 0, 'g*', markersize=12, label='Global Minimum')
        else:
            ax.plot(0, 0, 'g*', markersize=12, label='Global Maximum')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(setting['title'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
    
    plt.suptitle('SGD Optimization: Minimization vs Maximization\nMomentum=0.5, Learning Rate=0.1', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sgd_min_max_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('sgd_min_max_comparison.pdf', bbox_inches='tight')
    plt.show()

def plot_loss_curves():
    """Plot loss curves for different momentum values."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    momentum_values = [0.0, 0.3, 0.6, 0.9]
    colors = ['red', 'blue', 'green', 'purple']
    
    for momentum, color in zip(momentum_values, colors):
        trajectory, losses = optimize_function(
            func_type='minimize',
            momentum=momentum,
            lr=0.1,
            num_steps=50
        )
        
        steps = range(len(losses))
        ax.semilogy(steps, losses, color=color, linewidth=2, 
                   marker='o', markersize=3, alpha=0.8, 
                   label=f'Momentum={momentum}')
    
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Function Value (log scale)')
    ax.set_title('Convergence Speed: Effect of Momentum')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sgd_convergence_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig('sgd_convergence_curves.pdf', bbox_inches='tight')
    plt.show()

def main():
    """Main function to run SGD visualization experiments."""
    
    print("SGD Behavior Visualization")
    print("==========================")
    
    print("\nGenerating momentum comparison plot...")
    plot_momentum_comparison()
    
    print("Generating weight decay comparison plot...")
    plot_weight_decay_comparison()
    
    print("Generating minimization vs maximization comparison plot...")
    plot_maximize_comparison()
    
    print("Generating convergence curves plot...")
    plot_loss_curves()
    
    print("\nAnalysis Summary:")
    print("=================")
    
    print("\n1. Momentum Effects:")
    print("   - Momentum=0.0: Direct gradient descent, can oscillate")
    print("   - Momentum=0.5: Smoother trajectory, faster convergence")
    print("   - Momentum=0.9: Very smooth, can overshoot but recovers quickly")
    
    print("\n2. Weight Decay Effects:")
    print("   - Without weight decay: Standard optimization trajectory")
    print("   - With weight decay: Adds regularization, slightly different path")
    print("   - Weight decay acts as a force pulling parameters toward zero")
    
    print("\n3. Maximization vs Minimization:")
    print("   - Minimization (maximize=False): Moves toward minimum at (0,0)")
    print("   - Maximization (maximize=True): Moves toward maximum at (0,0)")
    print("   - The maximize parameter effectively negates gradients")
    
    print("\nAll plots have been saved as PNG and PDF files.")
    print("Repository: https://github.com/junayedhs/nlp-advanced-homework")

if __name__ == "__main__":
    main()
