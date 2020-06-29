#####################################################################################
"""
data_reading.py

Functions to read in DeepLabCut outputs

last modified: June 27, 2020
"""
#####################################################################################

# import packages
import pandas as pd

####################################################

def read_dlc(dlcfile):
    '''
    Read in and manage column names of topdown data passed in in the form of .h5 files.
    '''
    try:
        # read in .h5 file
        pts = pd.read_hdf(dlcfile)
    except ValueError:
        # read in .h5 file when there is a key set in corral_files.py
        pts = pd.read_hdf(dlcfile, key='data')
    # organize columns of pts
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
    pts = pts.rename(columns={pts.columns[n]: pts.columns[n].replace(' ', '_') for n in range(len(pts.columns))})
    pt_loc_names = pts.columns.values
    return pts, pt_loc_names

####################################################

def read_in_eye(data_input, side, num_points=8):
    '''
    Read in and manage column names of eye data passed in in the form of .h5 files.
    '''

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
    '''
    Read in topdown, left eye, and/or right eye .h5 files by calling above functions.
    '''

    # read top-down camera data into xarray
    if topdown_input != None:
        topdown_pts, topdown_names = read_dlc(topdown_input)
    elif topdown_input == None:
        print('no top-down data given')

    # read in left and right eye (okay if not provided)
    lefteye_pts, lefteye_names = read_in_eye(lefteye_input, 'left')
    righteye_pts, righteye_names = read_in_eye(righteye_input, 'right')

    return topdown_pts, topdown_names, lefteye_pts, lefteye_names, righteye_pts, righteye_names

####################################################

def test_trial_presence(data, trial_name):
    '''
    test to make sure the trial exists before using it. This function is used in topdown_preening.py and check_tracking.py.
    '''
    try:
        data.sel(trial=trial_name)
        exists = True
    except ValueError:
        exists = False
    return exists
