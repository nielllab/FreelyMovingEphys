"""
session_analysis.py
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
import yaml

from util.params import extract_params
from util.dlc import run_DLC_Analysis
from util.deinterlace import deinterlace_data
from util.track_world import track_LED
from util.config import set_preprocessing_config_defaults
from util.calibration import get_calibration_params, calibrate_new_world_vids, calibrate_new_top_vids
from util.img_processing import auto_contrast

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    return args

def main(args):
    # check the file extension of config
    # json and yaml configs need to be handled differently
    if os.path.splitext(args.config)[1] == '.json':
        print('config read in as JSON')
        config_is_yaml = False
    elif os.path.splitext(args.config)[1] == '.yaml':
        print('config read in as YAML')
        config_is_yaml = True
    # open config
    with open(args.config, 'r') as infile:
        config = yaml.load(infile, Loader=yaml.FullLoader)
    # label the config internally so that it's clear which format it uses
    config['config_is_yaml'] = config_is_yaml
    # update the config read in with default values if any required keys aren't there
    

    # run through steps of preprocessing and subsequent analysis
    if config['deinterlace']['run_deinter']:
        deinterlace_data(config)

    if config['img_correction']['run_img_correction']:
        auto_contrast(config)

    if config['calibration']['run_cam_calibration']:
        get_calibration_params(config)

    if config['calibration']['undistort_recordings']:
        calibrate_new_world_vids(config)

    if config['pose_estimation']['run_dlc']:
        run_DLC_Analysis(config)

    if config['parameters']['run_params']:
        extract_params(config)

    if config['ir_spot_in_space']['run_is_spot_in_space']:
        track_LED(config)

    if config['ephys_analysis']['run_ephys_analysis']:
        session_ephys_analysis(config)

if __name__ == '__main__':
    args = get_args()
    main(args)