# Network architectures for video classification

import torch
import torch.nn as nn

class FrameEncoder2D(nn.Module):
    """2D CNN encoder for processing individual frames"""
    def __init__(self, input_channels=3):
        super(FrameEncoder2D, self).__init__()
        
        self.encoder = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Global average pooling
        )
        
        self.feature_dim = 512 * 4 * 4  # 512 * 4 * 4 = 8192
        
    def forward(self, x):
        # x shape: [batch, channels, height, width]
        features = self.encoder(x)  # [batch, 512, 4, 4]
        features = features.view(features.size(0), -1)  # [batch, 512*4*4]
        return features

class OpticalFlowEncoder2D(nn.Module):
    """2D CNN encoder for processing optical flow frames"""
    def __init__(self, input_channels=2):  # 2 channels for optical flow (x, y)
        super(OpticalFlowEncoder2D, self).__init__()
        
        self.encoder = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Global average pooling
        )
        
        self.feature_dim = 512 * 4 * 4  # 512 * 4 * 4 = 8192
        
    def forward(self, x):
        # x shape: [batch, channels, height, width]
        features = self.encoder(x)  # [batch, 512, 4, 4]
        features = features.view(features.size(0), -1)  # [batch, 512*4*4]
        return features

class VideoEncoder3D(nn.Module):
    """3D CNN encoder for processing video sequences"""
    def __init__(self, input_channels=3):
        super(VideoEncoder3D, self).__init__()
        
        self.conv3d_1 = nn.Conv3d(input_channels, 64, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool3d_1 = nn.MaxPool3d(kernel_size=(1, 2, 2))  # Only pool spatial dimensions
        
        self.conv3d_2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.pool3d_2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        self.conv3d_3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.pool3d_3 = nn.MaxPool3d(kernel_size=(2, 2, 2))  # Now pool temporal dimension
        
        self.conv3d_4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=1)
        self.bn4 = nn.BatchNorm3d(512)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.relu = nn.ReLU()
        self.feature_dim = 512
        
    def forward(self, x):
        # x shape: [batch, channels, frames, height, width]
        
        # First 3D conv block
        x = self.relu(self.bn1(self.conv3d_1(x)))
        x = self.pool3d_1(x)
        
        # Second 3D conv block
        x = self.relu(self.bn2(self.conv3d_2(x)))
        x = self.pool3d_2(x)
        
        # Third 3D conv block
        x = self.relu(self.bn3(self.conv3d_3(x)))
        x = self.pool3d_3(x)
        
        # Fourth 3D conv block
        x = self.relu(self.bn4(self.conv3d_4(x)))
        
        # Global average pooling
        x = self.global_pool(x)  # [batch, 512, 1, 1, 1]
        
        # Flatten
        features = x.view(x.size(0), -1)  # [batch, 512]
        return features

class Classifier(nn.Module):
    """Shared classifier for all models"""
    def __init__(self, input_dim, num_classes=10):
        super(Classifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)
