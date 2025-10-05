from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T

root_dir = 'Project2/ufc10'

class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir=root_dir,
    split='train', 
    transform=None
):
        self.frame_paths = sorted(glob(f'{root_dir}/frames/{split}/*/*/*.jpg'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
       
    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        video_name = frame_path.split('/')[-2]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()
        
        frame = Image.open(frame_path).convert("RGB")

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        return frame, label


class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir = root_dir, 
    split = 'train', 
    transform = None,
    stack_frames = True
):

        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        
        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split('/')[-1].split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        video_frames_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'frames')
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]
        
        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)


        return frames, label
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames


class DualStreamVideoDataset(torch.utils.data.Dataset):
    """Dataset for dual-stream models that need both RGB and optical flow data"""
    def __init__(self, 
        root_dir=root_dir, 
        split='train', 
        transform=None,
        stack_frames=True
    ):
        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        
        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split('/')[-1].split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        # Load RGB frames
        rgb_frames_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'frames')
        rgb_frames = self.load_rgb_frames(rgb_frames_dir)
        
        # Load optical flow frames
        flow_frames_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'flows')
        flow_frames = self.load_flow_frames(flow_frames_dir)

        # Apply transforms to RGB frames
        if self.transform:
            rgb_frames = [self.transform(frame) for frame in rgb_frames]
        else:
            rgb_frames = [T.ToTensor()(frame) for frame in rgb_frames]
        
        # Convert flow frames from numpy arrays to tensors and resize
        import torch.nn.functional as F
        processed_flow_frames = []
        for flow in flow_frames:
            # flow shape is (2, 224, 224), convert to tensor
            flow_tensor = torch.from_numpy(flow).float()  # [2, 224, 224]
            # Resize to 64x64 to match RGB frames
            flow_tensor = F.interpolate(flow_tensor.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0)
            processed_flow_frames.append(flow_tensor)
        
        flow_frames = processed_flow_frames
        
        if self.stack_frames:
            rgb_frames = torch.stack(rgb_frames).permute(1, 0, 2, 3)  # [channels, frames, H, W]
            flow_frames = torch.stack(flow_frames).permute(1, 0, 2, 3)  # [channels, frames, H, W]

        return rgb_frames, flow_frames, label
    
    def load_rgb_frames(self, frames_dir):
        """Load RGB frames from frames directory"""
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)
        return frames
    
    def load_flow_frames(self, flow_frames_dir):
        """Load optical flow frames from flows directory"""
        import numpy as np
        flow_frames = []
        
        # Check if flows directory exists
        if not os.path.exists(flow_frames_dir):
            print(f"Flow directory does not exist: {flow_frames_dir}")
            # If flows directory doesn't exist, create zero flow placeholders
            for i in range(1, self.n_sampled_frames + 1):
                # Create zero flow placeholder
                zero_flow = np.zeros((2, 224, 224), dtype=np.float32)
                flow_frames.append(zero_flow)
        else:
            # Load actual optical flow data from .npy files
            for i in range(1, self.n_sampled_frames + 1):
                # Try the correct naming convention: flow_X_Y.npy
                if i < self.n_sampled_frames:
                    flow_file = os.path.join(flow_frames_dir, f"flow_{i}_{i+1}.npy")
                else:
                    # For the last frame, use the previous flow
                    flow_file = os.path.join(flow_frames_dir, f"flow_{i-1}_{i}.npy")
                
                if os.path.exists(flow_file):
                    # Load the .npy file
                    flow_data = np.load(flow_file)
                    flow_frames.append(flow_data)
                else:
                    # Create zero flow placeholder if flow file doesn't exist
                    zero_flow = np.zeros((2, 224, 224), dtype=np.float32)
                    flow_frames.append(zero_flow)
        
        return flow_frames


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    root_dir = root_dir

    transform = T.Compose([T.Resize((64, 64)),T.ToTensor()])
    frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
    framevideostack_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = True)
    framevideolist_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = False)


    frameimage_loader = DataLoader(frameimage_dataset,  batch_size=8, shuffle=False)
    framevideostack_loader = DataLoader(framevideostack_dataset,  batch_size=8, shuffle=False)
    framevideolist_loader = DataLoader(framevideolist_dataset,  batch_size=8, shuffle=False)


    # # Printing the shapes of the datasets
    # for frames, labels in frameimage_loader:
    #     print(frames.shape, labels.shape) # [batch, channels, height, width]

    # for video_frames, labels in framevideolist_loader:
    #     print(45*'-')
    #     for frame in video_frames: # loop through number of frames
    #         print(frame.shape, labels.shape)# [batch, channels, height, width]
    
    # Printing the shapes of the datasets with stacked frames
    for video_frames, labels in framevideostack_loader:
        print(video_frames.shape, labels.shape) # [batch, channels, number of frames, height, width] 
            
