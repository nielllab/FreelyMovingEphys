"""
datasets.py
"""
import glob, os, sys, re, json
import pandas as pd
import numpy as np
from PIL import Image
from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def pil_loader(path):
    """
    open the worldcam image and return it as rbg
    """
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class WorldcamDataset3D(Dataset):
    def __init__(self, image_csv, history_size, spike_npy, root_dir, transform=None):
        self.img_paths = pd.read_csv(image_csv)
        self.root_dir = root_dir
        self.spike_bins = np.load(spike_npy)
        self.transform = transform
        self.history_size = history_size

    def __len__(self):
        return(len(self.img_paths))
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = []
        for n in range(self.history_size):
            img_name = self.img_paths['N_{:02d}'.format(n)].iloc[idx]
            img_path = os.path.join(self.root_dir,self.img_paths.iloc[idx,1],img_name)
            img = pil_loader(img_path)
            if self.transform:
                img = self.transform(img)
            sample.append(img)
            
        sample = torch.cat(sample,dim=0).unsqueeze(0)
        return sample