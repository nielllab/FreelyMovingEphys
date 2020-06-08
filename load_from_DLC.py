#####################################################################################
"""
load_from_DLC.py of FreelyMovingEphys
(formerly: load_all_csv.py)

Loads in top-down camera and right or left eye from DeepLabCut .h5 file outputs.
Contains three helper functions: (1) read_dlc() opens .h5 files and organizes
the column names of input data as a pandas DataFrame. (2) read_in_eye() reads in the
left and right eye data passed to it from read_data(). Eye tag positions are renamed
so that the side of the mouse's eye that the data comes from is in that column label.
(3) read_data() is passed data files for as many cameras as user wants, and
returns a pandas structure for all of them.
The user provides to this script a file path for all of the data, collected by a
glob function, and an xarray DataArray is built from each of these for each trial.
The trials are then passed through functions to preen the data.

Requires the functions in topdown_preening.py
Adapted from GitHub repository /niell-lab-analysis/freely moving/loadAllCsv.m

TO DO:
- align data by time instead of by video frame
- interpret the right and left eye points using code in eye_tracking.py
- transform worldcam with eye direction
- start using find() from fine_function.py

last modified: June 3, 2020 by Dylan Martins (dmartins@uoregon.edu)
"""
#####################################################################################
from glob import glob
import pandas as pd
import os.path
import h5py
import numpy as np
import xarray as xr
import h5netcdf

# from utilities.find_function import find
from topdown_preening import preen_topdown_data
from eye_tracking import eye_angles

####################################################
def read_dlc(dlcfile):
    # read in .h5 file
    pts = pd.read_hdf(dlcfile)
    # organize columns of pts
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
    pt_loc_names = pts.columns.values
    return pts, pt_loc_names

####################################################
def read_in_eye(data_input, side, num_points=8):
    # create list of eye points that matches data variables in data xarray
    eye_pts = []
    num_points_for_range = num_points + 1
    for eye_pt in range(1, num_points_for_range):
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
            eye_data, eye_names = read_dlc(data_input)
            # turn old and new labels into dictionary so that eye points can be renamed
            col_corrections = {new_eye_pts[i]: eye_pts[i] for i in range(0, len(new_eye_pts))}
            eye_data = pd.DataFrame.rename(eye_data, columns=col_corrections)
        except NameError:
            # if the trial's main data file wasn't provided, raise error
            print('cannot add ' + str(side) + ' eye because no topdown camera data were given')
    # if eye data wasn't given, provide message (should still move forward with top-down or just one eye)
    elif data_input == None:
        print('no ' + str(side) + ' eye data given')
        eye_data = None
        eye_names = None

    return eye_data, eye_names

####################################################
def read_data(topdown_input=None, lefteye_input=None, righteye_input=None):

    # read top-down camera data into xarray
    if topdown_input != None:
        topdown_pts, topdown_names = read_dlc(topdown_input)
    elif topdown_input == None:
        print('no top-down data given')

    # read in left and right eye (okay if not provided)
    lefteye_pts, lefteye_names = read_in_eye(lefteye_input, 'left')
    righteye_pts, righteye_names = read_in_eye(righteye_input, 'right')

    return topdown_pts, topdown_names, lefteye_pts, lefteye_names, righteye_pts, righteye_names

####################################
##          USER INPUTS           ##
####################################
# find list of all data
main_path = '/Users/dylanmartins/data/Niell/PreyCapture/Cohort?/*/*/Approach/'

# find the files wanted from the given main_path
# first the DeepLabCut point locations
topdown_file_list = glob(main_path + '*top*DeepCut*.h5')
acc_file_list = glob(main_path + '*acc*.dat')
time_file_list = glob(main_path + '*topTS*.h5')
righteye_file_list = glob(main_path + '*eye1r*DeepCut*.h5')
lefteye_file_list = glob(main_path + '*eye2l*DeepCut*.h5')
# then video files that those points come from
righteye_vid_list = glob(main_path + '*eye1r*.avi')
lefteye_vid_list = glob(main_path + '*eye2l*.avi')
topdown_vid_list = glob(main_path + '*top*.avi')

# loop through each topdown file and find the associated files
# then, read the data in for each set, and build from it an xarray DataArray
loop_count = 0

# for testing purposes, limit number of topdown files read in. error will be raised if limit_of_loops is not 2 or
# greater; align_head_from_DLC wants to index through trials and it can only do that if there is more than one
limit_of_loops = 2


####################################
trial_id_list = []

for file in topdown_file_list:
    if loop_count < limit_of_loops:
        split_path = os.path.split(file)
        file_name = split_path[1]
        mouse_key = file_name[0:5]
        trial_key = file_name[17:27]

        # get a the other recorded files that are associated with the topdown file currently being read
        acc_file = ', '.join([i for i in acc_file_list if mouse_key and trial_key in i])
        time_file = ', '.join([i for i in time_file_list if mouse_key and trial_key in i])
        righteye_file = ', '.join([i for i in righteye_file_list if mouse_key and trial_key in i])
        lefteye_file = ', '.join([i for i in lefteye_file_list if mouse_key and trial_key in i])
        righteye_vid = ', '.join([i for i in righteye_vid_list if mouse_key and trial_key in i])
        lefteye_vid = ', '.join([i for i in lefteye_vid_list if mouse_key and trial_key in i])
        topdown_vid_list = ', '.join([i for i in topdown_vid_list if mouse_key and trial_key in i])

        # read in the data from file locations
        topdown_pts, topdown_names, lefteye_pts, lefteye_names, righteye_pts, righteye_names = read_data(file, lefteye_file, righteye_file)

        # make a unique name for the mouse and the recording trial
        trial_id = 'mouse_' + str(mouse_key) + '_trial_' + str(trial_key)
        trial_id_list.append(trial_id)

        # build one DataArray that stacks up all topdown trials in separate dimensions
        if loop_count == 0:
            topdown = xr.DataArray(topdown_pts)
            topdown = xr.DataArray.rename(topdown, new_name_or_name_dict={'dim_0': 'frame', 'dim_1': 'point_loc'})
            topdown['trial'] = trial_id
        elif loop_count > 0:
            topdown_trial = xr.DataArray(topdown_pts)
            topdown_trial = xr.DataArray.rename(topdown_trial, new_name_or_name_dict={'dim_0': 'frame', 'dim_1': 'point_loc'})
            topdown_trial['trial'] = trial_id
            topdown = xr.concat([topdown, topdown_trial], dim='trial', fill_value=np.nan)

        # build one DataArray that stacks up all trials in separate dimensions for each of two possible eyes
        if lefteye_pts is not None:
            if loop_count == 0:
                lefteye = xr.DataArray(lefteye_pts)
                lefteye = xr.DataArray.rename(lefteye, new_name_or_name_dict={'dim_0': 'frame', 'dim_1': 'point_loc'})
                lefteye['trial'] = trial_id
            elif loop_count > 0:
                lefteye_trial = xr.DataArray(lefteye_pts)
                lefteye_trial = xr.DataArray.rename(lefteye_trial, new_name_or_name_dict={'dim_0': 'frame', 'dim_1': 'point_loc'})
                lefteye_trial['trial'] = trial_id
                lefteye = xr.concat([lefteye, lefteye_trial], dim='trial', fill_value=np.nan)
        elif lefteye_pts is None:
            print('trial ' + trial_id + ' has no left eye camera data')
            lefteye = None

        if righteye_pts is not None:
            if loop_count == 0:
                righteye = xr.DataArray(righteye_pts)
                righteye = xr.DataArray.rename(righteye, new_name_or_name_dict={'dim_0': 'frame', 'dim_1': 'point_loc'})
                righteye['trial'] = trial_id
            elif loop_count > 0:
                righteye_trial = xr.DataArray(righteye_pts)
                righteye_trial = xr.DataArray.rename(righteye_trial, new_name_or_name_dict={'dim_0': 'frame', 'dim_1': 'point_loc'})
                righteye_trial['trial'] = trial_id
                righteye = xr.concat([righteye, righteye_trial], dim='trial', fill_value=np.nan)
        elif righteye_pts is None:
            print('trial ' + trial_id + ' has no right eye camera data')
            righteye = None

        loop_count = loop_count + 1

# run through each topdown trial, correct y-coordinates, and threshold point liklihoods
preened_topdown = preen_topdown_data(topdown, trial_id_list, topdown_names, figures=False)

eye_angles(lefteye, lefteye_names, trial_id_list, figures=True)
# left_theta, left_phi, left_longaxis_all, left_shortaxis_all, left_CamCent = eye_angles(lefteye, lefteye_names, trial_id_list, figures=True)
# right_theta, right_phi, right_longaxis_all, right_shortaxis_all, right_CamCent = eye_angles(righteye, righteye_names, trial_id_list, figures=True)
