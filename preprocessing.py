"""
preprocessing.py

deinterlace videos, analyze with DLC, and extract parameters

Nov. 17, 2020
"""
# package imports
import argparse, json, sys, os, subprocess, shutil
import cv2
import pandas as pd
import deeplabcut
import numpy as np
import xarray as xr
import warnings
import tkinter as tk
from tkinter import filedialog
from glob import glob
from multiprocessing import freeze_support
# module imports
from util.params import extract_params
from util.dlc import run_DLC_Analysis
from util.deinterlace import deinterlace_data

# get user inputs
def pars_args():
    parser = argparse.ArgumentParser(description='deinterlace videos and adjust timestamps to match')
    parser.add_argument('-c', '--json_config_path', 
        default='~/Desktop/preprocessing_config.json',
        help='path to video analysis config file')
    args = parser.parse_args()
    
    return args

def main(args=None, json_config_path=None):
    if (args == None) & (json_config_path != None):
        json_config_path = os.path.normpath(os.path.expanduser(json_config_path))
    else:
        json_config_path = os.path.normpath(os.path.expanduser(args.json_config_path))

    # open config file
    with open(json_config_path, 'r') as fp:
        config = json.load(fp)

    print('Config: ')    
    print(json.dumps(config, indent=1))
    data_path = os.path.expanduser(config['data_path'])
    if config.get('save_path') is None:
        config['save_path'] = data_path
    else: 
        save_path = os.path.expanduser(config['save_path'])

    steps = config['steps_to_run']

    # deinterlace data
    if steps['deinter'] is True:
        deinterlace_data(config)
    # get dlc tracking
    if steps['dlc'] is True:
        run_DLC_Analysis(config)
    # extract parameters from dlc
    if steps['params'] is True:
        extract_params(config)

if __name__ == '__main__':
    args = pars_args()
    
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    
    main(args,json_config_path=file_path)