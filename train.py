# Unified training script for all fusion models

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
import os
import json
from datasets import FrameImageDataset, FrameVideoDataset, DualStreamVideoDataset
from models import (
    PerFrameAggregation2D, LateFusion2D, EarlyFusion2D,
    PerFrameAggregation3D, LateFusion3D, EarlyFusion3D
)

def create_dataloaders(root_dir='ucf101_noleakage', batch_size=8, results_dir='without_leakage', include_dual_stream=False):
    """Create dataloaders for train, validation, and test sets"""
    
    # Define transforms
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor()
    ])
    
    if include_dual_stream:
        # Create dual-stream datasets for RGB + optical flow
        train_dataset = DualStreamVideoDataset(root_dir=root_dir, split='train', transform=transform, stack_frames=True)
        val_dataset = DualStreamVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames=True)
    else:
        # Create standard RGB-only datasets
        train_dataset = FrameVideoDataset(root_dir=root_dir, split='train', transform=transform, stack_frames=True)
        val_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames=True)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, num_epochs=75, learning_rate=0.001, model_name="model", results_dir='without_leakage'):
    """Train the model with saving functionality"""
    
    # Create save directories
    os.makedirs(f'{results_dir}/models', exist_ok=True)
    os.makedirs(f'{results_dir}/logs', exist_ok=True)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("The code will run on GPU.")

    ## MPS doesn't support 3D operations like MaxPool3d
    # elif torch.backends.mps.is_available():
    #     device = torch.device('mps') 
    #     print("The code will run on MPS")
        
    else:
        device = torch.device('cpu')
        print("The code will run on CPU")

    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Track training metrics
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, data in enumerate(train_loader):
            if len(data) == 3:  # Dual-stream data: (rgb_frames, flow_frames, labels)
                rgb_frames, flow_frames, labels = data
                rgb_frames, flow_frames, labels = rgb_frames.to(device), flow_frames.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(rgb_frames, flow_frames)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            else:  # Single-stream data: (frames, labels)
                frames, labels = data
                frames, labels = frames.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(frames)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data in val_loader:
                if len(data) == 3:  # Dual-stream data: (rgb_frames, flow_frames, labels)
                    rgb_frames, flow_frames, labels = data
                    rgb_frames, flow_frames, labels = rgb_frames.to(device), flow_frames.to(device), labels.to(device)
                    outputs = model(rgb_frames, flow_frames)
                else:  # Single-stream data: (frames, labels)
                    frames, labels = data
                    frames, labels = frames.to(device), labels.to(device)
                    outputs = model(frames)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # Store metrics
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model (weights only)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), f'{results_dir}/models/{model_name}_best_weights.pth')
            print(f'New best model saved! (Val Acc: {val_acc:.2f}%)')
        
        
        print('-' * 50)
    
    # Save final model (weights only)
    torch.save(model.state_dict(), f'{results_dir}/models/{model_name}_final_weights.pth')
    
    
    # Save training history as JSON
    training_history = {
        'model_name': model_name,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch
    }
    
    with open(f'{results_dir}/logs/{model_name}_training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"Model saving completed!")
    print(f"Best weights saved to: {results_dir}/models/{model_name}_best_weights.pth")
    print(f"Final weights saved to: {results_dir}/models/{model_name}_final_weights.pth")
    print(f"Training history saved to: {results_dir}/logs/{model_name}_training_history.json")
    print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch + 1})")
    
    return model, training_history


def main(dataset_name='ucf101_noleakage', results_dir='without_leakage', include_dual_stream=False):
    """Main function to run training for all models"""
    
    # Define all available models
    models = {}
    
    # Add dual-stream models first if enabled
    if include_dual_stream:
        from models import DualStreamPerFrame2D, DualStreamLateFusion2D, DualStreamEarlyFusion2D
        models.update({
            'DualStreamPerFrame2D': DualStreamPerFrame2D,
            'DualStreamLateFusion2D': DualStreamLateFusion2D,
            'DualStreamEarlyFusion2D': DualStreamEarlyFusion2D,
        })
    
    # Add standard models
    models.update({
        'PerFrame2D': PerFrameAggregation2D,
        'LateFusion2D': LateFusion2D,
        'EarlyFusion2D': EarlyFusion2D,
        'PerFrame3D': PerFrameAggregation3D,
        'LateFusion3D': LateFusion3D,
        'EarlyFusion3D': EarlyFusion3D,
    })
    
    # Create dataloaders
    print("Creating dataloaders...")
    if include_dual_stream:
        # Create separate dataloaders for original and dual-stream models
        train_loader_rgb, val_loader_rgb = create_dataloaders(root_dir=dataset_name, results_dir=results_dir, include_dual_stream=False)
        train_loader_dual, val_loader_dual = create_dataloaders(root_dir=dataset_name, results_dir=results_dir, include_dual_stream=True)
    else:
        # Single dataloader for RGB-only models
        train_loader_rgb, val_loader_rgb = create_dataloaders(root_dir=dataset_name, results_dir=results_dir, include_dual_stream=False)
        train_loader_dual, val_loader_dual = None, None
    
    # Train each model
    for model_name, model_class in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        # Create model
        model = model_class(num_classes=10, num_frames=10)
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Choose the correct dataloader for this model type
        if model_name.startswith('DualStream'):
            if train_loader_dual is None:
                print(f"Warning: {model_name} is a dual-stream model but dual-stream dataloader is not available. Skipping.")
                continue
            train_loader, val_loader = train_loader_dual, val_loader_dual
        else:
            train_loader, val_loader = train_loader_rgb, val_loader_rgb
        
        # Train model
        print("Starting training...")
        train_model(model, train_loader, val_loader, model_name=model_name, results_dir=results_dir)
        
        print(f'{model_name} training completed!')
        print(f'Best model weights saved to: {results_dir}/models/{model_name}_best_weights.pth')
    
    # Training completed
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}")
    print("All models have been trained and saved.")
    print("Run 'python eval.py' to evaluate on test set.")
    print("Run 'python plotting.py' to generate analysis plots.")

if __name__ == '__main__':
    main()
