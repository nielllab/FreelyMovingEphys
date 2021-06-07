"""
session_analysis.py
"""
import argparse, json, sys, os, subprocess, shutil, yaml
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
from util.config import set_preprocessing_config_defaults, str_to_bool, open_config
from util.calibration import get_calibration_params, calibrate_new_world_vids, calibrate_new_top_vids
from util.img_processing import auto_contrast
from project_analysis.ephys.ephys_by_session import session_ephys_analysis
from util.paths import find

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--clear_dlc', type=str_to_bool, nargs='?', const=True, default=False)
    args = parser.parse_args()
    return args

def main(config_path, clear_dlc=False, force_probe_name=None):
    if type(config_path) == dict:
        # if config options were provided instead of the expected path to a file
        config = config_path
    else:
        with open(config_path, 'r') as infile:
            config = yaml.load(infile, Loader=yaml.FullLoader)

    print('analyzing session with path',config['animal_dir'])

    if force_probe_name is not None:
        config['ephys_analysis']['probe_type'] = force_probe_name
    
    if config['deinterlace']['run_deinter']:
        deinterlace_data(config)
    if config['img_correction']['run_img_correction']:
        auto_contrast(config)
    if config['calibration']['run_cam_calibration']:
        get_calibration_params(config)
    if config['calibration']['undistort_recordings']:
        calibrate_new_world_vids(config)
    if config['pose_estimation']['run_dlc']:
        if clear_dlc:
            h5_list = find('*DLC*.h5',config['animal_dir'])
            pickle_list = find('*DLC*.pickle',config['animal_dir'])
            file_list = h5_list + pickle_list
            for item in file_list:
                os.remove(item)
                print('removed ' + item)
        run_DLC_Analysis(config)
    if config['parameters']['run_params']:
        extract_params(config)
    if config['ir_spot_in_space']['run_is_spot_in_space']:
        track_LED(config)
    if config['ephys_analysis']['run_ephys_analysis']:
        session_ephys_analysis(config)

if __name__ == '__main__':
    args = get_args()
    main(args.config, args.clear_dlc)