#####################################################################################
"""
load_all_csv.py of FreelyMovingEphys

Loads in top-down camera and right or left eye from DLC outputs and data are aligned.

Requires alignment_from_DLC.py
Adapted from /niell-lab-analysis/freely moving/loadAllCsv.m

last modified: May 18, 2020
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

####################################
def read_dlc(dlcfile):
    pts = pd.read_hdf(dlcfile)
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
    xarray = pts.to_xarray()
    return xarray

####################################
def read_in_eye(total_data, data_input, side):
    # create list of eye points that matches data variables in data xarray
    eye_pts = []
    for eye_pt in range(1, 9):
        eye_pts.append('p' + str(eye_pt) + ' x')
        eye_pts.append('p' + str(eye_pt) + ' y')
        eye_pts.append('p' + str(eye_pt) + ' likelihood')

    # create list of eye points labeled with which eye they come from
    new_eye_pts = []
    for old_eye_pt in eye_pts:
        new_eye_pts.append(str(side) + ' eye ' + str(old_eye_pt))

    # turn old and new lables into dictionary so that eye points can be renamed
    eye_dict = {eye_pts[i]: new_eye_pts[i] for i in range(len(new_eye_pts))}

    if data_input != None:
        try:
            eye_read_in = read_dlc(data_input)
            eye_data = eye_read_in.rename(eye_dict)
            print(eye_data)
            total_data = xr.merge([total_data, eye_data])
        except NameError:
            print('cannot add ' + str(side) + ' eye because no top-down camera data were given')
    elif data_input == None:
        print('no ' + str(side) + ' eye data given')

    return total_data

####################################
def read_data(topdown_input=None, acc_input=None, time_input=None, lefteye_input=None, righteye_input=None):

    # read top-down camera data into xarray
    if topdown_input != None:
        data = read_dlc(topdown_input)
    elif topdown_input == None:
        print('no top-down data given')

    # read in left eye
    data = read_in_eye(data, lefteye_input, 'left')
    data = read_in_eye(data, righteye_input, 'right')

    # aligned = align_head_from_DLC(data)

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
        data = read_data(file, acc_file, time_file, righteye_file, lefteye_file)
    loop_count = loop_count + 1
