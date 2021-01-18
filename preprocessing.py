"""
preprocessing.py

deinterlace videos, analyze with DLC, and extract parameters

Jan. 14, 2021
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
from util.config import set_preprocessing_config_defaults
from util.calibration import get_calibration_params, calibrate_new_world_vids, calibrate_new_top_vids

def main(json_config_path):
    # open config file
    with open(json_config_path, 'r') as fp:
        config = json.load(fp)
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
    if steps['get_cam_calibration_params'] is True:
        get_calibration_params(config)
    if steps['undistort_recording'] is True:
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
    
    try:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
    except:
        print('cannot open dialog box')
        file_path = input('enter path to json config file: ')
    
    main(json_config_path=file_path)