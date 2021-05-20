"""
batch_analysis.py

takes in a csv file path, yaml config file, and directory into which log should be saved
might work with json config, but ephys analysis won't be possible, so yaml is best
runs preprocessing and ephys analysis for each of the trials marked to be analyzed in csv file
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
import traceback

from util.params import extract_params
from util.dlc import run_DLC_Analysis
from util.deinterlace import deinterlace_data
from util.track_world import track_LED
from util.config import set_preprocessing_config_defaults, str_to_bool, open_config
from util.calibration import get_calibration_params, calibrate_new_world_vids, calibrate_new_top_vids
from project_analysis.ephys.analyze_ephys import find_files, run_ephys_analysis
from util.log import log
from util.paths import find
from session_analysis import main as analyze_session

# get user arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_filepath', type=str, help='read path for metadata .csv')
    parser.add_argument('--config', type=str, help='yaml config file')
    parser.add_argument('--log_dir', type=str, help='save path for logger .csv')
    parser.add_argument('--clear_dlc', type=str_to_bool, nargs='?', const=True, default=False, help='delete existing DLC .h5 files?')
    args = parser.parse_args()

    return args

def main(csv_filepath, config_path, log_dir, clear_dlc):
    # initialize logger
    logf = log(os.path.join(log_dir,'batch_log.csv'),name=['recording'])

    # read in the csv batch file
    print('opening csv file')
    csv = pd.read_csv(csv_filepath)

    # filter out rows of the csv that are marked to be analyzed with preprocessing and ephys analysis (these should be seperate columns in the df)
    run_preproc = csv.loc[csv['run_preprocessing'] == 'TRUE']
    run_ephys = csv.loc[csv['run_ephys_analysis'] == 'TRUE']

    # delete existing DLC .h5 files so that there will be only one in the directory
    # needed in case a different DLC network is being used
    if clear_dlc is True:
        for ind, row in run_preproc.iterrows():
            del_path = row['animal_dirpath']
            h5_list = find('*DLC_resnet50*.h5',del_path)
            pickle_list = find('*DLC_resnet50*.pickle',del_path)
            file_list = h5_list + pickle_list
            for item in file_list:
                os.remove(item)

    for ind, row in csv.iterrows():
        # get the provided data path
        data_path = row['animal_dirpath']
        # read in the generic config for this batch analysis
        config = open_config(config_path)
        # update generic config path for the current index of batch file
        config['data_path'] = data_path
        # if step was switched off for this index in the batch file, overwrite what is in the config file
        # if the csv file has a step switched on, this will leave the config file as it is
        if row['run_preprocessing'] != 'TRUE':
            config['steps_to_run']['deinter'] = False
            config['steps_to_run']['img_correction'] = False
            config['steps_to_run']['get_cam_calibration_params'] = False
            config['steps_to_run']['undistort_recording'] = False
            config['steps_to_run']['dlc'] = False
            config['steps_to_run']['params'] = False
            config['steps_to_run']['addtl_params'] = False
        if row['run_ephys_analysis'] != 'TRUE':
            config['steps_to_run']['ephys'] = False
        # run session analysis using the yaml config file
        try:
            analyze_session(config, clear_dlc=clear_dlc, force_probe_name=row['probe_name'])
        except Exception as e:
            logf.log([row['experiment_date']+'_'+row['animal_name'], traceback.format_exc()],PRINT=False)

if __name__ == '__main__':
    args = get_args()
    main(args.csv_filepath, args.config, args.log_dir, args.clear_dlc)