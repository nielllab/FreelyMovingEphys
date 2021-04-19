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
from util.config import set_preprocessing_config_defaults
from util.calibration import get_calibration_params, calibrate_new_world_vids, calibrate_new_top_vids
from util.img_processing import auto_contrast

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    return args

def main(args):
    config = open_config(args.config)

    steps = config['steps_to_run']
    
    if steps['deinter']:
        deinterlace_data(config)
    if steps['img_correction']:
        auto_contrast(config)
    if steps['get_cam_calibration_params']:
        get_calibration_params(config)
    if steps['undistort_recording']:
        calibrate_new_world_vids(config)
    if steps['dlc']:
        run_DLC_Analysis(config)
    if steps['params']:
        extract_params(config)
    if steps['addtl_params']:
        track_LED(config)
    if steps['ephys']:
        session_ephys_analysis(config)

if __name__ == '__main__':
    args = get_args()
    main(args)