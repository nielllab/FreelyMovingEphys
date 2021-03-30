"""
manual_preprocessing.py

deinterlace videos, analyze with DLC, and extract parameters
"""
import argparse, json, sys, os, subprocess, shutil
import cv2
import pandas as pd
os.environ["DLClight"] = "True"
import deeplabcut
import numpy as np
import xarray as xr
import warnings
import tkinter as tk
from tkinter import filedialog
from glob import glob
from multiprocessing import freeze_support
import timeit

from util.params import extract_params
from util.dlc import run_DLC_Analysis
from util.deinterlace import deinterlace_data
from util.track_world import track_LED
from util.config import set_preprocessing_config_defaults
from util.calibration import get_calibration_params, calibrate_new_world_vids, calibrate_new_top_vids
from util.paths import find

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--data_path', type=str, default=None)
    args = parser.parse_args()
    return args

def main(json_config_path):
    # open config file
    with open(json_config_path, 'r') as fp:
        config = json.load(fp)

    if args.data_path != None:
        config['data_path']=args.data_path
    # update the config read in with default values if any required keys aren't there
    config = set_preprocessing_config_defaults(config)

    data_path = os.path.expanduser(config['data_path'])
    if config.get('save_path') is None:
        config['save_path'] = data_path
    else: 
        save_path = os.path.expanduser(config['save_path'])

    steps = config['steps_to_run']
    start = timeit.default_timer()
    # deinterlace data
    if steps['deinter'] is True:
        deinterlace_data(config)
    end_deinter = timeit.default_timer()
    if steps['get_cam_calibration_params'] is True:
        get_calibration_params(config)
    end_calib = timeit.default_timer()
    if steps['undistort_recording'] is True:
        calibrate_new_world_vids(config)
        # calibrate_new_top_vids(config)
    end_undistort = timeit.default_timer()
    # get dlc tracking
    if steps['dlc'] is True:
        # delete existing DLC .h5 files so that there will be only one in the directory
        # needed in case a different DLC network is being used
        h5_list = find('*DLC_resnet50*.h5',data_path)
        pickle_list = find('*DLC_resnet50*.pickle',data_path)
        file_list = h5_list + pickle_list
        for item in file_list:
            os.remove(item)
            print('Deleted:',item)

        run_DLC_Analysis(config)
    end_dlc = timeit.default_timer()
    # extract parameters from dlc
    if steps['params'] is True:
        extract_params(config)
    end_params = timeit.default_timer()
    if steps['addtl_params']:
        track_LED(config)
    end = timeit.default_timer()

    print('PREPROCESSING TIMES (min):')
    print('deinterlacing: '+str((end_deinter-start)/60))
    print('calibration: '+str((end_calib-end_deinter)/60))
    print('undistortion: '+str((end_undistort-end_calib)/60))
    print('pose estimation: '+str((end_dlc-end_undistort)/60))
    print('parameters: '+str((end_params-end_dlc)/60))
    print('additional parameters: '+str((end-end_params)/60))

if __name__ == '__main__':
    args = get_args()
    config_path = args.config_path
    main(config_path)