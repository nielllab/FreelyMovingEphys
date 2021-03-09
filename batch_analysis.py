"""
batch_analysis.py

takes in a csv file path
runs preprocessing and ephys analysis for each of the trials marked to be analyzed in csv file
"""
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

from util.params import extract_params
from util.dlc import run_DLC_Analysis
from util.deinterlace import deinterlace_data
from util.track_world import track_LED
from util.config import set_preprocessing_config_defaults, str_to_bool
from util.calibration import get_calibration_params, calibrate_new_world_vids, calibrate_new_top_vids
from project_analysis.ephys.analyze_ephys import find_files, run_ephys_analysis
from util.log import log

# get user arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='read path for metadata .csv')
    parser.add_argument('--log_path', type=str, help='save path for logger .csv')
    parser.add_argument('--clear_dlc', type=str_to_bool, nargs='?', const=True, default=False, help='delete existing DLC .h5 files?')
    args = parser.parse_args()

    return args

def main(csv_path, log_path, clear_dlc):
    # initialize logger
    logf = log(os.path.join(args.log_path+'_log.csv'),name=['recording'])

    # read in the csv batch file
    csv = pd.read_csv(csv_path)

    # filter out rows of the csv that are marked to be analyzed with preprocessing and ephys analysis (these should be seperate columns in the df)
    run_preproc = csv.loc[csv['run_preproc'] == True]
    run_ephys = csv.loc[csv['run_ephys'] == True]

    # delete existing DLC .h5 files so that there will be only one in the directory
    # needed in case a different DLC network is being used
    for ind, row in run_preproc.iterrows():
        del_path = row['Data location (i.e. V2/Kraken, drive)']
        h5_list = find('*DLC_resnet50*.h5',del_path)
        pickle_list = find('*DLC_resnet50*.pickle',del_path)
        file_list = h5_list + pickle_list
        for item in file_list:
            os.remove(item)

    # itereate through the preprocessing list
    for ind, row in run_preproc.iterrows():
        try:
            # get the provided data path
            data_path = row['Data location (i.e. V2/Kraken, drive)']
            # create a config file w/ data path from csv
            config = {'data_path': data_path}
            # set all of the default config options in the github default json
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
                # calibrate_new_top_vids(config)
            # get dlc tracking
            if steps['dlc'] is True:
                run_DLC_Analysis(config)
            # extract parameters from dlc
            if steps['params'] is True:
                extract_params(config)
            if steps['addtl_params']:
                track_LED(config)
        except Exception as e:
            logf.log([ind, e],PRINT=False)

    # iterate through ephys analysis list
    for ind, row in run_ephys.iterrows():
        data_path = row['Data location (i.e. V2/Kraken, drive)']
        recording_names = row['rec_types'].split(',')
        for recording_name in recording_names:
            try:
                if 'fm' in recording_name:
                    fm = True
                elif 'fm' not in recording_name:
                    fm = False
                this_unit = int(row['unit2highlight'])
                if fm == True:
                    stim_type = 'None'
                elif 'wn' in recording_name:
                    stim_type = 'white_noise'
                elif 'grat' in recording_name:
                    stim_type = 'gratings'
                elif 'noise' in recording_name:
                    stim_type = 'sparse_noise'
                elif 'revchecker' in recording_name:
                    stim_type = 'rev_checker'
                if 'Rig' in recording:
                    rec_label = recording.split('_')[4:]
                else: # for older trials before rig was labeled
                    rec_label = recording.split('_')[3:]
                recording_path = os.path.join(data_path, rec_label)
                mp4 = True
                find_files(recording_path, recording_name, fm, this_unit, stim_type, mp4)
                run_ephys_analysis(file_dict)
            except Exception as e:
                logf.log([ind, e],PRINT=False)

if __name__ == '__main__':
    args = get_args()
    main(args.csv_path, args.log_path, args.clear_dlc)