#!/usr/bin/env python3
"""
Results Analysis Script for Deep Learning Project 2
Creates training curves and performance visualizations from saved training histories.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_training_history(model_name, checkpoints_dir="checkpoints"):
    """Load training history from JSON file"""
    history_file = os.path.join(checkpoints_dir, f"{model_name}_training_history.json")
    
    if not os.path.exists(history_file):
        print(f"Warning: {history_file} not found")
        return None
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    return history

def plot_training_curves(histories, save_dir="plots"):
    """Plot training and validation curves for all models"""
    
    # Create plots directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the plot style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves Comparison', fontsize=16, fontweight='bold')
    
    # Colors for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    model_names = list(histories.keys())
    
    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    for i, (model_name, history) in enumerate(histories.items()):
        if history is not None:
            epochs = range(1, len(history['train_losses']) + 1)
            ax1.plot(epochs, history['train_losses'], 
                    label=model_name, color=colors[i % len(colors)], 
                    linewidth=2, marker='o', markersize=4)
    
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax2 = axes[0, 1]
    for i, (model_name, history) in enumerate(histories.items()):
        if history is not None:
            epochs = range(1, len(history['val_losses']) + 1)
            ax2.plot(epochs, history['val_losses'], 
                    label=model_name, color=colors[i % len(colors)], 
                    linewidth=2, marker='s', markersize=4)
    
    ax2.set_title('Validation Loss', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training Accuracy
    ax3 = axes[1, 0]
    for i, (model_name, history) in enumerate(histories.items()):
        if history is not None:
            epochs = range(1, len(history['train_accs']) + 1)
            ax3.plot(epochs, history['train_accs'], 
                    label=model_name, color=colors[i % len(colors)], 
                    linewidth=2, marker='o', markersize=4)
    
    ax3.set_title('Training Accuracy', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Validation Accuracy
    ax4 = axes[1, 1]
    for i, (model_name, history) in enumerate(histories.items()):
        if history is not None:
            epochs = range(1, len(history['val_accs']) + 1)
            ax4.plot(epochs, history['val_accs'], 
                    label=model_name, color=colors[i % len(colors)], 
                    linewidth=2, marker='s', markersize=4)
    
    ax4.set_title('Validation Accuracy', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training curves saved to {save_dir}/training_curves.png")

def plot_final_results(histories, save_dir="plots"):
    """Plot final test accuracy comparison"""
    
    # Extract final test accuracies (from the log file or use best validation accuracies)
    model_names = []
    test_accuracies = []
    
    # Final results from the training log
    final_results = {
        'PerFrame2D': 24.17,
        'LateFusion2D': 80.83,
        'EarlyFusion2D': 35.00,
        'PerFrame3D': 21.67,
        'LateFusion3D': 30.00,
        'EarlyFusion3D': 25.83
    }
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    models = list(final_results.keys())
    accuracies = list(final_results.values())
    
    bars = plt.bar(models, accuracies, color=colors[:len(models)], alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Highlight the best model
    best_idx = accuracies.index(max(accuracies))
    bars[best_idx].set_color('#ff4444')
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)
    
    plt.title('Final Test Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Model Architecture', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 90)
    
    # Add performance categories
    plt.axhline(y=70, color='green', linestyle='--', alpha=0.7, label='Excellent (>70%)')
    plt.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Good (>50%)')
    plt.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Poor (<30%)')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_results.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Final results plot saved to {save_dir}/final_results.png")

def plot_learning_efficiency(histories, save_dir="plots"):
    """Plot learning efficiency (how quickly models reach certain accuracy thresholds)"""
    
    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (model_name, history) in enumerate(histories.items()):
        if history is not None:
            epochs = range(1, len(history['val_accs']) + 1)
            plt.plot(epochs, history['val_accs'], 
                    label=f"{model_name} (Final: {history['val_accs'][-1]:.1f}%)", 
                    color=colors[i % len(colors)], linewidth=3, marker='o', markersize=6)
    
    plt.title('Learning Efficiency: Validation Accuracy Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 85)
    
    # Add threshold lines
    plt.axhline(y=70, color='green', linestyle='--', alpha=0.7, label='Excellent (70%)')
    plt.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Good (50%)')
    plt.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Poor (30%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Learning efficiency plot saved to {save_dir}/learning_efficiency.png")

def create_summary_table(histories, save_dir="plots"):
    """Create a summary table of all results"""
    
    # Final results
    final_results = {
        'PerFrame2D': 24.17,
        'LateFusion2D': 80.83,
        'EarlyFusion2D': 35.00,
        'PerFrame3D': 21.67,
        'LateFusion3D': 30.00,
        'EarlyFusion3D': 25.83
    }
    
    # Create summary data
    summary_data = []
    for model_name in final_results.keys():
        history = histories.get(model_name)
        if history:
            summary_data.append({
                'Model': model_name,
                'Test Accuracy': f"{final_results[model_name]:.2f}%",
                'Best Val Accuracy': f"{history['best_val_acc']:.2f}%",
                'Best Epoch': history['best_epoch'] + 1,
                'Final Train Acc': f"{history['train_accs'][-1]:.2f}%",
                'Final Val Acc': f"{history['val_accs'][-1]:.2f}%"
            })
    
    # Save as CSV
    import pandas as pd
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(save_dir, 'results_summary.csv'), index=False)
    
    print(f"Results summary saved to {save_dir}/results_summary.csv")
    print("\nResults Summary:")
    print(df.to_string(index=False))

def main():
    """Main function to generate all plots and analysis"""
    
    print("🔬 Deep Learning Project 2 - Results Analysis")
    print("=" * 50)
    
    # Define model names
    model_names = [
        'PerFrame2D', 'LateFusion2D', 'EarlyFusion2D',
        'PerFrame3D', 'LateFusion3D', 'EarlyFusion3D'
    ]
    
    # Load all training histories
    print("📊 Loading training histories...")
    histories = {}
    for model_name in model_names:
        history = load_training_history(model_name)
        histories[model_name] = history
        if history:
            print(f"✅ Loaded {model_name}")
        else:
            print(f"❌ Failed to load {model_name}")
    
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    # Generate all plots
    print("\n📈 Generating plots...")
    
    print("1. Training curves...")
    plot_training_curves(histories)
    
    print("2. Final results comparison...")
    plot_final_results(histories)
    
    print("3. Learning efficiency...")
    plot_learning_efficiency(histories)
    
    print("4. Summary table...")
    create_summary_table(histories)
    
    print("\n🎉 All plots and analysis completed!")
    print("📁 Check the 'plots' folder for all generated files:")
    print("   - training_curves.png")
    print("   - final_results.png") 
    print("   - learning_efficiency.png")
    print("   - results_summary.csv")

if __name__ == "__main__":
    main()
