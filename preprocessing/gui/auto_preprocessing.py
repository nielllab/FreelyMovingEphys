"""
auto_preprocessing.py

automaticlly build a config file from inputs to the GUI

Jan. 15, 2021
"""
# package imports
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
# module imports
from util.params import extract_params
from util.dlc import run_DLC_Analysis
from util.deinterlace import deinterlace_data
from util.track_world import track_LED
from util.config import set_preprocessing_config_defaults
from util.calibration import get_calibration_params, calibrate_new_world_vids, calibrate_new_top_vids

def run_auto_preprocessing(config_path):

    with open(config_path, 'r') as fp:
        config = json.load(fp)

    steps = config['steps_to_run']

    # a few functions look for both a data path and save path
    # this is a quick way to avoid updating those functions for the curent json fields
    data_path = os.path.expanduser(config['data_path'])
    if config.get('save_path') is None:
        config['save_path'] = data_path
    else: 
        save_path = os.path.expanduser(config['save_path'])

    # deinterlace data
    if steps['deinter'] is True:
        deinterlace_data(config)
    if steps['get_cam_calibration_params'] is True:
        get_calibration_params(config)
    if steps['undistort_recording'] is True:
        calibrate_new_world_vids(config)
        # calibrate_new_top_vids(config)
    # get dlc tracking
    if steps['dlc'] is True:
        run_DLC_Analysis(config)
    # extract parameters from dlc
    if steps['params'] is True:
        extract_params(config)
    if steps['addtl_params']:
        track_LED(config)