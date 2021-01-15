"""
clear_dlc_outputs.py

delete all existing DeepLabCut .h5 and .pickle files in a recording directory

Jan. 12, 2021
"""
# package imports
import pandas as pd
import numpy as np
import xarray as xr
from glob import glob
import os
import fnmatch
import dateutil
import cv2
from tqdm import tqdm
from datetime import datetime
import time
import argparse

# glob for subdirectories
def find(pattern, path):
    result = [] # initialize the list as empty
    for root, dirs, files in os.walk(path): # walk though the path directory, and files
        for name in files:  # walk to the file in the directory
            if fnmatch.fnmatch(name,pattern):  # if the file matches the filetype append to list
                result.append(os.path.join(root,name))
    return result # return full list of file of a given type

# get user inputs
def pars_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--recording_dir')
    args = parser.parse_args()
    
    return args

def main(args):
    experiment_path = args.recording_dir
    h5_list = find('*.h5',experiment_path)
    pickle_list = find('*.pickle',experiment_path)
    file_list = h5_list + pickle_list
    for item in file_list:
        os.remove(item)
        print('removed ' + item)

if __name__ == '__main__':
    args = pars_args()
    main(args)