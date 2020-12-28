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
from util.track_world import track_LED

# get user inputs
def pars_args():
    parser = argparse.ArgumentParser(description='deinterlace videos and adjust timestamps to match')
    parser.add_argument('-c', '--json_config_path',
        help='path to video analysis config file')
    args = parser.parse_args()
    
    return args

def main(args=None, json_config_path=None):

    # open config file
    with open(json_config_path, 'r') as fp:
        config = json.load(fp)

    # print('Config: ')    
    # print(json.dumps(config, indent=1))

    # update the config read in with default values if any required keys aren't there
    config = set_preprocessing_config_defaults(config)

    data_path = os.path.expanduser(config['data_path'])
    if config.get('save_path') is None:
        config['save_path'] = data_path
    else: 
        save_path = os.path.expanduser(config['save_path'])

    steps = config['steps_to_run']

    # deinterlace data
    if steps['deinter'] is True:
        deinterlace_data(config)
    if steps['get_calibration_params'] is True:
        get_calibration_params(config)
    if steps['calibrate_recording'] is True:
        calibrate_new_world_vids(config)
        calibrate_new_top_vids(config)
    # get dlc tracking
    if steps['dlc'] is True:
        run_DLC_Analysis(config)
    # extract parameters from dlc
    if steps['params'] is True:
        extract_params(config)
    if steps['addtl_params']:
        track_LED(config)

if __name__ == '__main__':
    # args = pars_args()
    
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    
    main(args=None,json_config_path=file_path)