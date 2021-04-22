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

class WorldcamDataset(Dataset):
    def __init__(self, img_csv, spike_json, root_dir, history_size, transform=None):
        self.img_data = pd.read_csv(img_csv)
        with open(spike_json, 'r') as fp:
            self.spike_data = json.load(fp)
        self.root_dir = root_dir
        self.transform = transform
        self.history_size = history_size

    def __len__(self):
        return(len(self.img_data))
    
    def __getitem__(self,idx):
        # images with history
        for n in range(self.history_size):
            img_name = self.img_data['N_{:02d}'.format(n)].iloc[idx]
            spike_arr = self.spike_data
            img_path = os.path.join(self.root_dir,self.img_data.iloc[idx,1],img_name)
            img = pil_loader(img_path)
            if self.transform:
                img = self.transform(img)
            sample.append(img)
        img_sample = torch.cat(sample,dim=0).unsqueeze(0)

        return img_sample, spike_sample