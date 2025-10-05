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

def load_training_history(model_name, logs_dir="without_leakage/logs"):
    """Load training history from JSON file"""
    history_file = os.path.join(logs_dir, f"{model_name}_training_history.json")
    
    if not os.path.exists(history_file):
        print(f"Warning: {history_file} not found")
        return None
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    return history

def load_test_results(results_dir='without_leakage'):
    """Load test results from JSON file"""
    test_results_file = f'{results_dir}/results.json'
    
    if not os.path.exists(test_results_file):
        print(f"Warning: Test results file not found: {test_results_file}")
        print("Run 'python eval.py' first to generate test results")
        return {}
    
    with open(test_results_file, 'r') as f:
        test_results = json.load(f)
    
    # Extract just the accuracy values
    final_results = {}
    for model_name, results in test_results.items():
        final_results[model_name] = results['accuracy']
    
    return final_results

def plot_training_curves(histories, save_dir="without_leakage/plots"):
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
    ax3.set_ylim(0, 100)
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
    ax4.set_ylim(0, 100)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training curves saved to {save_dir}/training_curves.png")

def plot_final_results(histories, save_dir="without_leakage/plots"):
    """Plot test accuracy comparison"""
    
    # Load test results from file
    final_results = load_test_results()
    
    if not final_results:
        print("ERROR: No test results available.")
        print("Run 'python eval.py' first to generate test results.")
        print("Cannot create test accuracy comparison plot without test data.")
        return
    
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
    
    plt.title('Test Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Model Architecture', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'testing_bars.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Final results plot saved to {save_dir}/final_results.png")

def load_test_results(results_dir='without_leakage'):
    """Load test results from JSON file"""
    test_results_file = f'{results_dir}/results.json'
    
    if not os.path.exists(test_results_file):
        print(f" Test results file not found: {test_results_file}")
        print("   Run 'python eval.py' first to generate test results")
        return {}
    
    with open(test_results_file, 'r') as f:
        test_results = json.load(f)
    
    # Extract just the accuracy values
    final_results = {}
    for model_name, results in test_results.items():
        final_results[model_name] = results['accuracy']
    
    return final_results


def main(results_dir='without_leakage'):
    """Main function to generate all plots and analysis"""
    
    print("Deep Learning Project 2 - Results Analysis")
    print("=" * 50)
    
    # Define model names
    model_names = [
        'PerFrame2D', 'LateFusion2D', 'EarlyFusion2D',
        'PerFrame3D', 'LateFusion3D', 'EarlyFusion3D'
    ]
    
    # Load all training histories
    print("Loading training histories...")
    histories = {}
    for model_name in model_names:
        history = load_training_history(model_name, f'{results_dir}/logs')
        histories[model_name] = history
        if history:
            print(f"Loaded {model_name}")
        else:
            print(f"No history found for {model_name}")
    
    # Check for test results
    test_results = load_test_results(results_dir)
    if test_results:
        print(f"Test results loaded: {len(test_results)} models")
    else:
        print(f"No test results found. Run 'python eval.py' to generate test results.")
        print("   Will use validation accuracies as fallback.")
    
    # Create plots directory
    os.makedirs(f"{results_dir}/plots", exist_ok=True)
    
    # Generate all plots
    print("\nGenerating plots...")
    
    print("1. Training curves...")
    plot_training_curves(histories, f'{results_dir}/plots')
    
    print("2. Final results comparison...")
    plot_final_results(histories, f'{results_dir}/plots')
    
    # Check if test results are available
    test_results = load_test_results(results_dir)
    if test_results:
        print(f"\nTest results loaded: {len(test_results)} models")
        print("Generating test accuracy comparison plot...")
    else:
        print(f"\nERROR: No test results found.")
        print("Run 'python eval.py' to generate test results first.")
        print("Skipping test accuracy comparison plot.")
    
    print("\nAll plots and analysis completed!")
    print(f"Check the '{results_dir}/plots' folder for all generated files:")
    print("   - training_curves.png")
    print("   - testing_bars.png")

if __name__ == "__main__":
    main()
