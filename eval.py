#!/usr/bin/env python3
"""
Test Models Script
Loads trained models and evaluates them on the test set
"""

import torch
import torch.nn as nn
import json
import os
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datasets import FrameVideoDataset, DualStreamVideoDataset
from models import (
    PerFrameAggregation2D, LateFusion2D, EarlyFusion2D,
    PerFrameAggregation3D, LateFusion3D, EarlyFusion3D,
    DualStreamPerFrame2D, DualStreamLateFusion2D, DualStreamEarlyFusion2D
)

def load_trained_model(model_name, device, models_dir="without_leakage/models"):
    """Load the best trained model for a given model name"""
    model_path = f'{models_dir}/{model_name}_best_weights.pth'
    
    if not os.path.exists(model_path):
        print(f"Model weights not found: {model_path}")
        return None
    
    # Create model instance
    if model_name == 'PerFrame2D':
        model = PerFrameAggregation2D(num_classes=10, num_frames=10)
    elif model_name == 'LateFusion2D':
        model = LateFusion2D(num_classes=10, num_frames=10)
    elif model_name == 'EarlyFusion2D':
        model = EarlyFusion2D(num_classes=10, num_frames=10)
    elif model_name == 'PerFrame3D':
        model = PerFrameAggregation3D(num_classes=10, num_frames=10)
    elif model_name == 'LateFusion3D':
        model = LateFusion3D(num_classes=10, num_frames=10)
    elif model_name == 'EarlyFusion3D':
        model = EarlyFusion3D(num_classes=10, num_frames=10)
    elif model_name == 'DualStreamPerFrame2D':
        model = DualStreamPerFrame2D(num_classes=10, num_frames=10)
    elif model_name == 'DualStreamLateFusion2D':
        model = DualStreamLateFusion2D(num_classes=10, num_frames=10)
    elif model_name == 'DualStreamEarlyFusion2D':
        model = DualStreamEarlyFusion2D(num_classes=10, num_frames=10)
    else:
        print(f"Unknown model: {model_name}")
        return None
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Loaded {model_name} from {model_path}")
    return model

def evaluate_model(model, test_loader, device, model_name):
    """Evaluate a single model on the test set"""
    model.eval()
    test_correct = 0
    test_total = 0
    test_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    print(f"Testing {model_name} on test set...")
    
    # Check if this is a dual-stream model
    is_dual_stream = model_name.startswith('DualStream')
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            if is_dual_stream:
                # Dual-stream models return (rgb_frames, flow_frames, labels)
                rgb_frames, flow_frames, labels = batch_data
                rgb_frames = rgb_frames.to(device)
                flow_frames = flow_frames.to(device)
                labels = labels.to(device)
                outputs = model(rgb_frames, flow_frames)
            else:
                # Standard single-stream models return (frames, labels)
                frames, labels = batch_data
                frames = frames.to(device)
                labels = labels.to(device)
                outputs = model(frames)
            
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            if batch_idx % 5 == 0:
                print(f'  Batch {batch_idx}/{len(test_loader)}, Loss: {loss.item():.4f}')
    
    test_accuracy = 100 * test_correct / test_total
    avg_test_loss = test_loss / len(test_loader)
    
    print(f"{model_name} Results:")
    print(f"   Test Accuracy: {test_accuracy:.2f}%")
    print(f"   Test Loss: {avg_test_loss:.4f}")
    print(f"   Correct: {test_correct}/{test_total}")
    
    return test_accuracy, avg_test_loss

def main(dataset_name='ucf101_noleakage', results_dir='without_leakage'):
    """Main function to test all trained models"""
    print("Model Testing - Test Set Evaluation")
    print("=" * 50)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU for testing")
    else:
        device = torch.device('cpu')
        print("Using CPU for testing")
    
    # Create test dataset
    print("\nLoading test dataset...")
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor()
    ])
    
    # Check if we need dual-stream dataset by looking for dual-stream models
    model_files = [f for f in os.listdir(f'{results_dir}/models') if f.endswith('_best_weights.pth')]
    has_dual_stream = any('DualStream' in f for f in model_files)
    
    # Create datasets for both standard and dual-stream models
    standard_test_dataset = FrameVideoDataset(root_dir=dataset_name, split='test', transform=transform, stack_frames=True)
    standard_test_loader = DataLoader(standard_test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    if has_dual_stream:
        print("Detected dual-stream models, creating dual-stream dataset")
        dual_stream_test_dataset = DualStreamVideoDataset(root_dir=dataset_name, split='test', transform=transform, stack_frames=True)
        dual_stream_test_loader = DataLoader(dual_stream_test_dataset, batch_size=32, shuffle=False, num_workers=4)
        print(f"Dual-stream test dataset loaded: {len(dual_stream_test_dataset)} samples")
    else:
        dual_stream_test_loader = None
    
    print(f"Standard test dataset loaded: {len(standard_test_dataset)} samples")
    
    # Model names to test - include dual-stream models if they exist
    base_model_names = ['PerFrame2D', 'LateFusion2D', 'EarlyFusion2D', 
                       'PerFrame3D', 'LateFusion3D', 'EarlyFusion3D']
    dual_stream_model_names = ['DualStreamPerFrame2D', 'DualStreamLateFusion2D', 'DualStreamEarlyFusion2D']
    
    # Check which models actually exist
    model_names = []
    for model_name in base_model_names + dual_stream_model_names:
        model_path = f'{results_dir}/models/{model_name}_best_weights.pth'
        if os.path.exists(model_path):
            model_names.append(model_name)
    
    print(f"Found {len(model_names)} models to evaluate: {model_names}")
    
    # Results storage
    test_results = {}
    
    print(f"\n Testing {len(model_names)} models on test set...")
    print("=" * 50)
    
    # Test each model
    for model_name in model_names:
        print(f"\n{'='*20} {model_name} {'='*20}")
        
        # Load model
        model = load_trained_model(model_name, device, f'{results_dir}/models')
        if model is None:
            print(f"Skipping {model_name} - model not found")
            test_results[model_name] = {'accuracy': 0.0, 'loss': 0.0}
            continue
        
        # Choose the appropriate test loader based on model type
        is_dual_stream = model_name.startswith('DualStream')
        if is_dual_stream and dual_stream_test_loader is not None:
            test_loader = dual_stream_test_loader
        else:
            test_loader = standard_test_loader
        
        # Evaluate on test set
        test_acc, test_loss = evaluate_model(model, test_loader, device, model_name)
        test_results[model_name] = {'accuracy': test_acc, 'loss': test_loss}
        
        # Clean up memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for model_name, results in test_results.items():
        print(f"{model_name:15s}: {results['accuracy']:6.2f}% (Loss: {results['loss']:.4f})")
    
    # Find best model
    best_model = max(test_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n Best Model: {best_model[0]} with {best_model[1]['accuracy']:.2f}% test accuracy")
    
    # Save results to file
    results_file = f'{results_dir}/results.json'
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nTest results saved to: {results_file}")
    print("Model testing completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--dataset', default='ucf101_noleakage', help='Dataset name')
    parser.add_argument('--results-dir', default='without_leakage', help='Results directory')
    args = parser.parse_args()
    main(dataset_name=args.dataset, results_dir=args.results_dir)
