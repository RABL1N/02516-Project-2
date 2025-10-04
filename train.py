# Unified training script for all fusion models

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
import os
import json
from datasets import FrameImageDataset, FrameVideoDataset
from models import (
    PerFrameAggregation2D, LateFusion2D, EarlyFusion2D,
    PerFrameAggregation3D, LateFusion3D, EarlyFusion3D
)

def create_dataloaders(root_dir='ufc10', batch_size=8):
    """Create dataloaders for train, validation, and test sets"""
    
    # Define transforms
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor()
    ])
    
    # Create datasets
    train_dataset = FrameVideoDataset(root_dir=root_dir, split='train', transform=transform, stack_frames=True)
    val_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames=True)
    test_dataset = FrameVideoDataset(root_dir=root_dir, split='test', transform=transform, stack_frames=True)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, model_name="model"):
    """Train the model with saving functionality"""
    
    # Create save directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
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
        
        for batch_idx, (frames, labels) in enumerate(train_loader):
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
            for frames, labels in val_loader:
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
            torch.save(model.state_dict(), f'models/{model_name}_best_weights.pth')
            print(f'New best model saved! (Val Acc: {val_acc:.2f}%)')
        
        # Save checkpoint (weights + additional info) every epoch
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader),
            'train_acc': train_acc,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'learning_rate': learning_rate,
            'model_name': model_name
        }
        torch.save(checkpoint, f'checkpoints/{model_name}_epoch_{epoch+1}.pth')
        
        print('-' * 50)
    
    # Save final model (weights only)
    torch.save(model.state_dict(), f'models/{model_name}_final_weights.pth')
    
    # Save final checkpoint
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs - 1,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'learning_rate': learning_rate,
        'model_name': model_name
    }
    torch.save(final_checkpoint, f'checkpoints/{model_name}_final_checkpoint.pth')
    
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
    
    with open(f'checkpoints/{model_name}_training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"Model saving completed!")
    print(f"Weights saved to: models/{model_name}_best_weights.pth")
    print(f"Checkpoints saved to: checkpoints/{model_name}_*.pth")
    print(f"Training history saved to: checkpoints/{model_name}_training_history.json")
    print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch + 1})")
    
    return model, training_history

def test_model(model, test_loader):
    """Test the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for frames, labels in test_loader:
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = 100 * test_correct / test_total
    return test_acc

def main():
    """Main function to run training for all models"""
    
    # Define all available models
    models = {
        'PerFrame2D': PerFrameAggregation2D,
        'LateFusion2D': LateFusion2D,
        'EarlyFusion2D': EarlyFusion2D,
        'PerFrame3D': PerFrameAggregation3D,
        'LateFusion3D': LateFusion3D,
        'EarlyFusion3D': EarlyFusion3D,
    }
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders()
    
    results = {}
    
    # Train and test each model
    for model_name, model_class in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        # Create model
        model = model_class(num_classes=10, num_frames=10)
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Train model
        print("Starting training...")
        train_model(model, train_loader, val_loader, model_name=model_name)
        
        # Test model
        print("Testing model...")
        test_acc = test_model(model, test_loader)
        print(f'{model_name} Test Accuracy: {test_acc:.2f}%')
        
        results[model_name] = test_acc
    
    # Print summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    for model_name, accuracy in results.items():
        print(f"{model_name:15}: {accuracy:6.2f}%")
    
    # Find best model
    best_model = max(results, key=results.get)
    print(f"\nBest performing model: {best_model} ({results[best_model]:.2f}%)")

if __name__ == '__main__':
    main()
