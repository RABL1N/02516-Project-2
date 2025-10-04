# Modular fusion models for video classification

import torch
import torch.nn as nn
from networks import FrameEncoder2D, VideoEncoder3D, Classifier

class PerFrameAggregation2D(nn.Module):
    """2D Per-frame aggregation model"""
    def __init__(self, num_classes=10, num_frames=10):
        super(PerFrameAggregation2D, self).__init__()
        
        self.frame_encoder = FrameEncoder2D(input_channels=3)
        self.temporal_aggregator = nn.Sequential(
            nn.Linear(self.frame_encoder.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = Classifier(512, num_classes)
        self.num_frames = num_frames
        
    def forward(self, x):
        # x shape: [batch, channels, frames, height, width]
        batch_size, channels, num_frames, height, width = x.shape
        
        # Reshape to process frames individually
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [batch, frames, channels, height, width]
        x = x.view(batch_size * num_frames, channels, height, width)
        
        # Process each frame through the 2D CNN
        frame_features = self.frame_encoder(x)  # [batch*frames, feature_dim]
        
        # Reshape back to separate frames
        frame_features = frame_features.view(batch_size, num_frames, -1)  # [batch, frames, feature_dim]
        
        # Temporal aggregation: average pooling across frames
        aggregated_features = torch.mean(frame_features, dim=1)  # [batch, feature_dim]
        
        # Process through temporal aggregator
        temporal_features = self.temporal_aggregator(aggregated_features)  # [batch, 512]
        
        # Final classification
        output = self.classifier(temporal_features)  # [batch, num_classes]
        
        return output

class LateFusion2D(nn.Module):
    """2D Late fusion model"""
    def __init__(self, num_classes=10, num_frames=10):
        super(LateFusion2D, self).__init__()
        
        self.frame_encoder = FrameEncoder2D(input_channels=3)
        self.frame_classifier = Classifier(self.frame_encoder.feature_dim, num_classes)
        self.fusion_method = 'average'  # Options: 'average', 'max', 'weighted'
        self.num_frames = num_frames
        
    def forward(self, x):
        # x shape: [batch, channels, frames, height, width]
        batch_size, channels, num_frames, height, width = x.shape
        
        # Reshape to process frames individually
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [batch, frames, channels, height, width]
        x = x.view(batch_size * num_frames, channels, height, width)
        
        # Process each frame through the 2D CNN
        frame_features = self.frame_encoder(x)  # [batch*frames, feature_dim]
        
        # Get individual frame predictions
        frame_predictions = self.frame_classifier(frame_features)  # [batch*frames, num_classes]
        
        # Reshape back to separate frames
        frame_predictions = frame_predictions.view(batch_size, num_frames, -1)  # [batch, frames, num_classes]
        
        # Late fusion: combine frame predictions
        if self.fusion_method == 'average':
            final_predictions = torch.mean(frame_predictions, dim=1)  # [batch, num_classes]
        elif self.fusion_method == 'max':
            final_predictions = torch.max(frame_predictions, dim=1)[0]  # [batch, num_classes]
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return final_predictions

class EarlyFusion2D(nn.Module):
    """2D Early fusion model"""
    def __init__(self, num_classes=10, num_frames=10):
        super(EarlyFusion2D, self).__init__()
        
        # Early fusion: process all frames together as a single input
        # Input channels = 3 * num_frames (concatenate all frames)
        self.early_fusion_encoder = FrameEncoder2D(input_channels=3 * num_frames)
        self.classifier = Classifier(self.early_fusion_encoder.feature_dim, num_classes)
        self.num_frames = num_frames
        
    def forward(self, x):
        # x shape: [batch, channels, frames, height, width]
        batch_size, channels, num_frames, height, width = x.shape
        
        # Early fusion: concatenate all frames along channel dimension
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [batch, frames, channels, height, width]
        x = x.view(batch_size, channels * num_frames, height, width)  # [batch, channels*frames, height, width]
        
        # Process the concatenated frames through 2D CNN
        features = self.early_fusion_encoder(x)  # [batch, feature_dim]
        
        # Final classification
        output = self.classifier(features)  # [batch, num_classes]
        
        return output

class PerFrameAggregation3D(nn.Module):
    """3D Per-frame aggregation model - process each frame with 3D conv, then aggregate"""
    def __init__(self, num_classes=10, num_frames=10):
        super(PerFrameAggregation3D, self).__init__()
        
        # 3D CNN for processing individual frame sequences
        # We'll process overlapping windows of frames
        self.frame_encoder = VideoEncoder3D(input_channels=3)
        self.temporal_aggregator = nn.Sequential(
            nn.Linear(self.frame_encoder.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = Classifier(512, num_classes)
        self.num_frames = num_frames
        self.window_size = 3  # Process 3-frame windows
        
    def forward(self, x):
        # x shape: [batch, channels, frames, height, width]
        batch_size, channels, num_frames, height, width = x.shape
        
        # Create overlapping windows of frames
        windows = []
        for i in range(num_frames - self.window_size + 1):
            window = x[:, :, i:i+self.window_size, :, :]  # [batch, channels, window_size, height, width]
            windows.append(window)
        
        # Process each window with 3D CNN
        window_features = []
        for window in windows:
            features = self.frame_encoder(window)  # [batch, feature_dim]
            window_features.append(features)
        
        # Stack features
        window_features = torch.stack(window_features, dim=1)  # [batch, num_windows, feature_dim]
        
        # Temporal aggregation: average features across windows
        aggregated_features = torch.mean(window_features, dim=1)  # [batch, feature_dim]
        
        # Process through temporal aggregator
        temporal_features = self.temporal_aggregator(aggregated_features)  # [batch, 512]
        
        # Final classification
        output = self.classifier(temporal_features)  # [batch, num_classes]
        
        return output

class LateFusion3D(nn.Module):
    """3D Late fusion model - split video into segments, process each with 3D CNN, then fuse"""
    def __init__(self, num_classes=10, num_frames=10):
        super(LateFusion3D, self).__init__()
        
        # 3D CNN for processing video segments
        self.segment_encoder = VideoEncoder3D(input_channels=3)
        self.segment_classifier = Classifier(self.segment_encoder.feature_dim, num_classes)
        self.fusion_method = 'average'
        self.num_frames = num_frames
        self.segment_size = 5  # Process 5-frame segments
        
    def forward(self, x):
        # x shape: [batch, channels, frames, height, width]
        batch_size, channels, num_frames, height, width = x.shape
        
        # Split video into overlapping segments
        segments = []
        for i in range(0, num_frames - self.segment_size + 1, self.segment_size // 2):
            segment = x[:, :, i:i+self.segment_size, :, :]  # [batch, channels, segment_size, height, width]
            segments.append(segment)
        
        # If we don't have enough segments, pad the last one
        if len(segments) == 0:
            segments = [x[:, :, :self.segment_size, :, :]]
        
        # Process each segment
        segment_predictions = []
        for segment in segments:
            segment_features = self.segment_encoder(segment)  # [batch, feature_dim]
            segment_pred = self.segment_classifier(segment_features)  # [batch, num_classes]
            segment_predictions.append(segment_pred)
        
        # Stack predictions
        segment_predictions = torch.stack(segment_predictions, dim=1)  # [batch, num_segments, num_classes]
        
        # Late fusion: average predictions across segments
        if self.fusion_method == 'average':
            output = torch.mean(segment_predictions, dim=1)  # [batch, num_classes]
        elif self.fusion_method == 'max':
            output = torch.max(segment_predictions, dim=1)[0]  # [batch, num_classes]
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return output

class EarlyFusion3D(nn.Module):
    """3D Early fusion model (this is just the standard 3D CNN)"""
    def __init__(self, num_classes=10, num_frames=10):
        super(EarlyFusion3D, self).__init__()
        
        self.video_encoder = VideoEncoder3D(input_channels=3)
        self.classifier = Classifier(self.video_encoder.feature_dim, num_classes)
        self.num_frames = num_frames
        
    def forward(self, x):
        # x shape: [batch, channels, frames, height, width]
        # Standard 3D CNN processing
        video_features = self.video_encoder(x)  # [batch, feature_dim]
        output = self.classifier(video_features)  # [batch, num_classes]
        
        return output
