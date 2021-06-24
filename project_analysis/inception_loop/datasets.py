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
    def __init__(self, csv, history_size, root_dir, transform=None):
        self.metadata = pd.read_csv(csv)
        self.root_dir = root_dir
        self.transform = transform
        self.history_size = history_size

    def __len__(self):
        return(len(self.metadata))
    
    def __getitem__(self,idx):
        """
        idx is a worldcam frame index
        OUTPUTS
            imgs: tensor of images, with the number of images used determined by the given history_size
            spikes: number of spikes at each window for a given unit
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imgs = []
        spikes = []
        if self.history_size == 0:
            history_win = [0]
        elif self.history_size != 0:
            history_win = list(range(-self.history_size+1,1)) # 7 to 0
        for n in history_win:
            img_name = self.metadata['F'+str(n)].iloc[idx]
            img_path = os.path.join(str(os.path.join(self.root_dir,self.metadata.loc[idx,'filename'])),img_name)
            img = pil_loader(img_path)
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
            spike_list = self.metadata['SR'+str(n)].iloc[idx] # list of spike rates for all units
            spikes.append(eval(spike_list)) # append list to list of spikes for entire history window

        imgs = torch.cat(imgs,dim=0).unsqueeze(0)
        spikes = torch.FloatTensor(spikes)

        return imgs, spikes