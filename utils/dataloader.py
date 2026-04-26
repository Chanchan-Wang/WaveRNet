"""
Dataloader for retinal vessel segmentation datasets
Supports multi-domain training with domain-specific labels
"""
import os
from skimage import io
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as pytorch_transforms
import torch.nn.functional as F


class BinaryLoader(Dataset):
    """
    Dataset loader for binary segmentation (vessel/background)
    
    Args:
        data_name: Name of the dataset (not used, kept for compatibility)
        jsfiles: List of image filenames
        transforms: Albumentations transforms
        pixel_mean: Mean values for normalization
        pixel_std: Std values for normalization
        domain_list: Dict mapping domain_id to list of image names
    """
    def __init__(self, data_name, jsfiles, transforms, 
                 pixel_mean=[123.675, 116.280, 103.530], 
                 pixel_std=[58.395, 57.12, 57.375], 
                 domain_list=None):
        # Data path: adjust based on your directory structure
        # Default: assumes data/ folder at same level as code
        self.path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
        self.jsfiles = jsfiles
        self.img_tensor = pytorch_transforms.Compose([pytorch_transforms.ToTensor()])
        self.transforms = transforms
        self.img_size = 1024
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        self.domain_list = domain_list
        
    def __len__(self):
        return len(self.jsfiles)
    
    def __getitem__(self, idx):
        image_id = list(self.jsfiles[idx].split('.'))[0]
        
        # Find domain id
        self.domain_id = None
        if self.domain_list is not None:
            for domain_id in self.domain_list.keys():
                if image_id in self.domain_list[domain_id]:
                    self.domain_id = domain_id
                    break
            
            # If not found, use default
            if self.domain_id is None:
                self.domain_id = list(self.domain_list.keys())[0] if self.domain_list else 0
        else:
            self.domain_id = 0  # Default domain

        image_path = os.path.join(self.path, 'image_1024/', image_id)
        mask_path = os.path.join(self.path, 'mask_1024/', image_id)

        img = io.imread(image_path + '.png')[:, :, :3].astype('float32')
        mask = io.imread(mask_path + '.png', as_gray=True)

        mask[mask > 0] = 255

        data_group = self.transforms(image=img, mask=mask)
        img_resized = data_group['image']
        mask = data_group['mask']

        img = self.img_tensor(img)
        img = self.preprocess(img)

        return (img_resized, img, mask, image_id, self.domain_id)
    
    def preprocess(self, x):
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))

        return x


class TestLoader(Dataset):
    """
    Dataset loader for testing/evaluation
    
    Args:
        data_name: Name of the dataset (not used, kept for compatibility)
        jsfiles: List of image filenames
        transforms: Albumentations transforms
        pixel_mean: Mean values for normalization
        pixel_std: Std values for normalization
    """
    def __init__(self, data_name, jsfiles, transforms,
                 pixel_mean=[123.675, 116.280, 103.530],
                 pixel_std=[58.395, 57.12, 57.375]):
        # Data path: adjust based on your directory structure
        self.path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
        self.jsfiles = jsfiles
        self.img_tensor = pytorch_transforms.Compose([pytorch_transforms.ToTensor()])
        self.transforms = transforms
        self.img_size = 1024
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
    
    def __len__(self):
        return len(self.jsfiles)
    
    def __getitem__(self, idx):
        image_id = list(self.jsfiles[idx].split('.'))[0]

        image_path = os.path.join(self.path, 'image_1024/', image_id)
        mask_path = os.path.join(self.path, 'mask_1024/', image_id)

        img = io.imread(image_path + '.png')[:, :, :3].astype('float32')
        mask = io.imread(mask_path + '.png', as_gray=True)

        mask[mask > 0] = 255

        data_group = self.transforms(image=img, mask=mask)
        img_resized = data_group['image']
        mask = data_group['mask']

        img = self.img_tensor(img)
        img = self.preprocess(img)

        return (img_resized, img, mask, image_id)
    
    def preprocess(self, x):
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))

        return x
