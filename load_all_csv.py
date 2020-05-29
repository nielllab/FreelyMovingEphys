#####################################################################################
"""
load_all_csv.py of FreelyMovingEphys

Loads in top-down camera and right or left eye from DLC outputs and data are aligned.

Requires alignment_from_DLC.py
Adapted from /niell-lab-analysis/freely moving/loadAllCsv.m

last modified: May 29, 2020
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
    # read in .h5 file
    pts = pd.read_hdf(dlcfile)
    # organize columns of pts
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
    return pts

####################################
def read_in_eye(data_input, side):
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

    # if eye data input exists, read it in and rename the data variables using the eye_dict of side-specific names
    if data_input != None:
        try:
            # read in .h5 file
            eye_data = read_dlc(data_input)
            # turn old and new labels into dictionary so that eye points can be renamed
            eye_data.rename(columns={eye_pts[i]: new_eye_pts[i] for i in range(len(new_eye_pts))})
        # if the trial's main data file wasn't provided, raise error
        except NameError:
            print('cannot add ' + str(side) + ' eye because no top-down camera data were given')
    # if eye data wasn't given, provide message (should still move forward with top-down or just one eye)
    elif data_input == None:
        print('no ' + str(side) + ' eye data given')
        eye_data = None

    return eye_data

####################################
def read_data(topdown_input=None, lefteye_input=None, righteye_input=None):

    # read top-down camera data into xarray
    if topdown_input != None:
        topdown_pts = read_dlc(topdown_input)
    elif topdown_input == None:
        print('no top-down data given')

    # read in left and right eye (okay if not provided)
    lefteye_pts = read_in_eye(lefteye_input, 'left')
    righteye_pts = read_in_eye(righteye_input, 'right')

    return topdown_pts, lefteye_pts, righteye_pts

####################################
# find list of all data
main_path = '/Users/dylanmartins/data/Niell/PreyCapture/Cohort?/*/*/Approach/'

topdown_file_list = glob(main_path + '*top*DeepCut*.h5')
acc_file_list = glob(main_path + '*acc*.dat')
time_file_list = glob(main_path + '*topTS*.h5')
righteye_file_list = glob(main_path + '*eye1r*DeepCut*.h5')
lefteye_file_list = glob(main_path + '*eye2l*DeepCut*.h5')

# loop through each topdown file and find the associated files
# then, read the data in for each set, and build from it an xarray DataArray
loop_count = 0
limit_of_loops = 1 # for testing purposes, limit number of topdown files read in
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

        topdown_pts, lefteye_pts, righteye_pts = read_data(file, righteye_file, lefteye_file)

        trial_id = 'mouse_' + str(mouse_key) + '_trial_' + str(trial_key)
        loop_label = 'trial_' + str(loop_count)

        # PROBLEM
        # intention here is to build one DataArray that stacks up all trials
        # currently, it seems to be appending the frames, which is not what we want
        if loop_count == 0:
            topdown = xr.DataArray(topdown_pts)
            topdown = xr.DataArray.rename(topdown, new_name_or_name_dict={'dim_0': 'frame', 'dim_1': 'point_loc'})
        elif loop_count > 0:
            topdown_trial = xr.DataArray(topdown_pts)
            topdown_trial = xr.DataArray.rename(topdown_trial, new_name_or_name_dict={'dim_0': 'frame', 'dim_1': 'point_loc'})
            topdown = xr.concat([topdown, topdown_trial], dim='frame', fill_value='NaN')

        # TO DO: align by time files instead of by frames

        # align topdown head points
        good_aligned_topdown = align_head_from_DLC(topdown, figures=True)

        loop_count = loop_count + 1
