"""
preprocessing.py

deinterlace videos, analyze with DLC, and extract parameters

Oct. 16, 2020
"""

import argparse, json, sys, os, subprocess, shutil
import cv2
import pandas as pd
import deeplabcut
import numpy as np
import xarray as xr
import warnings
from glob import glob
from multiprocessing import freeze_support

from util.read_data import pars_args
from util.params import extract_params
from util.dlc import run_DLC_Analysis
from util.deinterlace import deinterlace_data

def main(args):
    json_config_path = os.path.expanduser(args.json_config_path)

    # open config file
    with open(json_config_path, 'r') as fp:
        config = json.load(fp)

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
    main(args)