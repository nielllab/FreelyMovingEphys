#####################################################################################
"""
load_all_csv.py of FreelyMovingEphys

Loads in top-down camera and right or left eye from DLC outputs and data are aligned.

Requires alignment_from_DLC.py
Adapted from /niell-lab-analysis/freely moving/loadAllCsv.m

last modified: May 17, 2020''
"""
#####################################################################################
from glob import glob
import pandas as pd
import os.path
import h5py
import numpy as np
import xarray as xr
import h5netcdf

from utilities.find_function import find
from alignment_from_DLC import align_head_from_DLC

def read_data(topdown_input=None, acc_input=None, time_input=None, lefteye_input=None, righteye_input=None):
    if topdown_input != None:
        print('top-down camera data read in: ' + topdown_input)
        data = xr.open_dataset(topdown_input, engine='h5netcdf')
    elif topdown_input == None:
        print('no top-down data given')

    if lefteye_input != None:
        try:
            with xr.open_dataset(lefteye_input, engine='h5netcdf') as le:
                data = xr.concat([data, le], 'cam_input')
        except NameError:
            print('cannot add left eye because no top-down camera data were given')
    elif lefteye_input == None:
        print('no left eye data given')

    # points = align_head_from_DLC(topdown_data)

    return data

####################################
# find list of all data
main_path = '/Users/dylanmartins/data/Niell/PreyCapture/Cohort?/*/*/Approach/'

topdown_file_list = glob(main_path + '*top*DeepCut*.h5')
acc_file_list = glob(main_path + '*acc*.dat')
time_file_list = glob(main_path + '*topTS*.h5')
righteye_file_list = glob(main_path + '*eye1r*DeepCut*.h5')
lefteye_file_list = glob(main_path + '*eye2l*DeepCut*.h5')

loop_count = 0
limit_of_loops = 1 # for testing purposes, limit to first file
for file in topdown_file_list:
    if loop_count < limit_of_loops:
        split_path = os.path.split(file)
        file_name = split_path[1]
        mouse_key = file_name[0:5]
        trial_key = file_name[17:27]
        acc_file = ', '.join([i for i in acc_file_list if mouse_key and trial_key in i])
        time_file = ', '.join([i for i in time_file_list if mouse_key and trial_key in i])
        righteye_file = ', '.join([i for i in righteye_file_list if mouse_key and trial_key in i])
        lefteye_file = ', '.join([i for i in lefteye_file_list if mouse_key and trial_key in i])
        points = read_data(file, acc_file, time_file, righteye_file, lefteye_file)
    loop_count = loop_count + 1
