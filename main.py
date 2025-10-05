#!/usr/bin/env python3
"""
Main script for Deep Learning Project 2 - Video Classification Workflow

This script orchestrates the complete workflow:
1. Training models on specified dataset
2. Evaluating models on test set
3. Generating analysis plots

Automatically organizes results into appropriate directories based on dataset type.
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path

def create_directories(dataset_type, include_dual_stream=False):
    """Create necessary directories for the specified dataset type"""
    if include_dual_stream:
        base_dir = f"{dataset_type}_leakage_all"
    else:
        base_dir = f"{dataset_type}_leakage"
    
    directories = [
        f"{base_dir}/logs",
        f"{base_dir}/models", 
        f"{base_dir}/plots",
        f"{base_dir}/output"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    return base_dir

def run_training(dataset_name, base_dir, include_dual_stream=False):
    """Run training script with appropriate dataset"""
    print(f"\n{'='*60}")
    print(f"PHASE 1: TRAINING MODELS")
    print(f"Dataset: {dataset_name}")
    print(f"Results will be saved to: {base_dir}/")
    if include_dual_stream:
        print("Including dual-stream models (RGB + Optical Flow)")
    print(f"{'='*60}")
    
    # Import and run training
    try:
        from train import main as train_main
        train_main(dataset_name, base_dir, include_dual_stream)
        print("Training completed successfully!")
        return True
    except Exception as e:
        print(f"Training failed: {e}")
        return False

def run_evaluation(dataset_name, base_dir):
    """Run evaluation script"""
    print(f"\n{'='*60}")
    print(f"PHASE 2: EVALUATION")
    print(f"Dataset: {dataset_name}")
    print(f"Results will be saved to: {base_dir}/results.json")
    print(f"{'='*60}")
    
    # Import and run evaluation
    try:
        from eval import main as eval_main
        eval_main(dataset_name, base_dir)
        print("Evaluation completed successfully!")
        return True
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return False

def run_plotting(dataset_name, base_dir):
    """Run plotting script"""
    print(f"\n{'='*60}")
    print(f"PHASE 3: ANALYSIS & PLOTTING")
    print(f"Dataset: {dataset_name}")
    print(f"Plots will be saved to: {base_dir}/plots/")
    print(f"{'='*60}")
    
    # Import and run plotting
    try:
        from plotting import main as plot_main
        plot_main(base_dir)
        print("Plotting completed successfully!")
        return True
    except Exception as e:
        print(f"Plotting failed: {e}")
        return False

def check_dataset_availability(dataset_name):
    """Check if the specified dataset exists"""
    if not os.path.exists(dataset_name):
        print(f"Dataset '{dataset_name}' not found!")
        print(f"Available datasets:")
        for item in os.listdir('.'):
            if os.path.isdir(item) and not item.startswith('.'):
                print(f"  - {item}")
        return False
    return True

def print_summary(dataset_name, base_dir, success_phases):
    """Print workflow summary"""
    print(f"\n{'='*60}")
    print(f"WORKFLOW SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Results directory: {base_dir}/")
    print(f"Successful phases: {success_phases}")
    
    if len(success_phases) == 3:
        print(f"\nComplete workflow completed successfully!")
        print(f"Check results in: {base_dir}/")
        print(f"View plots in: {base_dir}/plots/")
    else:
        print(f"\nWorkflow completed with some issues")
        print(f"Partial results available in: {base_dir}/")

def main():
    """Main function to orchestrate the complete workflow"""
    parser = argparse.ArgumentParser(description='Deep Learning Project 2 - Complete Workflow')
    parser.add_argument('--dataset', '-d', 
                       choices=['ufc10', 'ucf101_noleakage'], 
                       default='ucf101_noleakage',
                       help='Dataset to use (default: ucf101_noleakage)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training phase (use existing models)')
    parser.add_argument('--skip-eval', action='store_true', 
                       help='Skip evaluation phase')
    parser.add_argument('--skip-plotting', action='store_true',
                       help='Skip plotting phase')
    parser.add_argument('--include-dual-stream', action='store_true',
                       help='Include dual-stream models (RGB + Optical Flow)')
    
    args = parser.parse_args()
    
    # Determine dataset type and base directory
    if args.dataset == 'ufc10':
        dataset_type = 'with'
        dataset_description = 'UFC-10 (with information leakage)'
    else:
        dataset_type = 'without'
        dataset_description = 'UCF-101 (without information leakage)'
    
    # Set base directory based on dual-stream flag
    if args.include_dual_stream:
        base_dir = f"{dataset_type}_leakage_all"
    else:
        base_dir = f"{dataset_type}_leakage"
    
    print(f"Deep Learning Project 2 - Complete Workflow")
    print(f"Dataset: {dataset_description}")
    print(f"Results directory: {base_dir}/")
    print(f"{'='*60}")
    
    # Check dataset availability
    if not check_dataset_availability(args.dataset):
        sys.exit(1)
    
    # Create directories
    create_directories(dataset_type, args.include_dual_stream)
    
    # Track successful phases
    success_phases = []
    
    # Phase 1: Training
    if not args.skip_training:
        if run_training(args.dataset, base_dir, args.include_dual_stream):
            success_phases.append("Training")
    else:
        print("Skipping training phase")
        success_phases.append("Training (skipped)")
    
    # Phase 2: Evaluation
    if not args.skip_eval:
        if run_evaluation(args.dataset, base_dir):
            success_phases.append("Evaluation")
    else:
        print("Skipping evaluation phase")
        success_phases.append("Evaluation (skipped)")
    
    # Phase 3: Plotting
    if not args.skip_plotting:
        if run_plotting(args.dataset, base_dir):
            success_phases.append("Plotting")
    else:
        print("Skipping plotting phase")
        success_phases.append("Plotting (skipped)")
    
    # Print summary
    print_summary(args.dataset, base_dir, success_phases)

if __name__ == "__main__":
    main()
