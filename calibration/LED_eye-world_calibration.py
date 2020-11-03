"""
LED_eye-world_calibration.py

format DeepLabCut points of an IR LED light source seperately in the world
camera and in the reflection of the pupil in the eye camera

Sept. 25, 2020
"""

# package imports
import argparse
import numpy as np
import xarray as xarray
import pandas as pd
import os

# module imports
from util.read_data import h5_to_xr

# get user inputs
parser = argparse.ArgumentParser(description='convert .h5 files to ')
parser.add_argument('-d', '--data_path', help='path to eye and world .h5 files in folder together')
parser.add_argument('-v', '--vid_path', help='path to eye and world .avi files in folder together')
parser.add_argument('-s', '--save_path', help='save path for two xarrays of point positions')
args = parser.parse_args()

camera_names = ['EYE', 'WORLD']

# find all the names of datasets and append them to a list
calib_set_paths = []
for h5 in find('*.h5', args.data_path):
    split_name = h5.split('_')[:-1]
    trial = '_'.join(split_name)
    if trial not in calib_set_paths:
        calib_set_paths.append(trial)

# loop through each unique set
for calib_set in calib_set_paths:
    trial_cam_h5 = [(args.data_path +'_{}.h5').format(name) for name in camera_names]

    # format the ephys data
    if 'EYE' in camera_names:
        print('formatting EYE data for ' + t_name)
        eye_h5 = os.path.join(args.data_path,)
        eye_csv = os.path.join(args.data_path,'ephys','spike_clusters.npy')
        eye_avi = os.path.join(args.data_path,'ephys','cluster_group.tsv')
        # read in the data and structure them in one xarray for all spikes during this trial
        topdlc, topnames = h5_to_xr(top_h5, top_csv, top_view)
        # save out the data
        ephys_xr.to_netcdf(os.path.join(config['save_path'], str(t_name+'ephys.nc')), engine='netcdf4')
