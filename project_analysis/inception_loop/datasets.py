"""
datasets.py
"""
import glob, os, sys
import pandas as pd
import numpy as np
from PIL import Image
from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class WorldcamDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_paths = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return(len(self.data_paths))
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,self.data_paths.iloc[idx,1],self.data_paths.iloc[idx,2])
        sample = pil_loader(img_name)
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample