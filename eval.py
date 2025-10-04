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
from datasets import FrameVideoDataset
from networks import FrameEncoder2D, VideoEncoder3D, Classifier
from models import (
    PerFrameAggregation2D, LateFusion2D, EarlyFusion2D,
    PerFrameAggregation3D, LateFusion3D, EarlyFusion3D
)

def load_trained_model(model_name, device):
    """Load the best trained model for a given model name"""
    model_path = f'models/{model_name}_best_weights.pth'
    
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
    
    with torch.no_grad():
        for batch_idx, (frames, labels) in enumerate(test_loader):
            frames, labels = frames.to(device), labels.to(device)
            
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

def main():
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
    test_dataset = FrameVideoDataset(root_dir='ufc10', split='test', transform=transform, stack_frames=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    print(f"Test dataset loaded: {len(test_dataset)} samples")
    
    # Model names to test
    model_names = ['PerFrame2D', 'LateFusion2D', 'EarlyFusion2D', 
                   'PerFrame3D', 'LateFusion3D', 'EarlyFusion3D']
    
    # Results storage
    test_results = {}
    
    print(f"\n Testing {len(model_names)} models on test set...")
    print("=" * 50)
    
    # Test each model
    for model_name in model_names:
        print(f"\n{'='*20} {model_name} {'='*20}")
        
        # Load model
        model = load_trained_model(model_name, device)
        if model is None:
            print(f"Skipping {model_name} - model not found")
            test_results[model_name] = {'accuracy': 0.0, 'loss': 0.0}
            continue
        
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
    results_file = 'results.json'
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nTest results saved to: {results_file}")
    print("Model testing completed!")

if __name__ == "__main__":
    main()
