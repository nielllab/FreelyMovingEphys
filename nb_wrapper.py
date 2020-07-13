"""
FreelyMovingEphys wrapper functions to access analysis functions
nb_wrapper.py

Last modified July 12, 2020
"""

# package imports
import os
import xarray as xr

# module imports
from util.read_data import open_h5, open_time
from util.track_eye import eye_tracking # , check_eye_calibration
from util.track_topdown import topdown_tracking # , head_angle
from util.plot_video import check_tracking

# topdown view function access
def topdown_intake(data_path, file_name, save_path, lik_thresh, coord_cor, topdown_pt_num, cricket, bonsaitime):
    dir = os.path.join(save_path, file_name)
    if not os.path.exists(dir):
        os.makedirs(dir)

    # get the complete path to the topdown trial named in the jupyter notebook
    try:
        h5_path = os.path.join(data_path, file_name) + '.h5'

        # read in .h5 DLC data
        pts, names = open_h5(h5_path)

        # interpolate, threshold, and plot safety-checks
        clean_pts = topdown_tracking(pts, names, save_path, file_name, lik_thresh, coord_cor, topdown_pt_num, cricket)

        # get head angle, plot safety-checks
        # theta = head_angle(clean_pts, names, lik_thresh)

        topout = xr.merge([pts, clean_pts])
    except FileNotFoundError:
        print('missing DLC file... output DLC xarray object is type None')
        h5_path = None
        topout = None

    try:
        avi_path = os.path.join(data_path, file_name) + '.avi'

        if h5_path is not None:
            # plot head points and head angle on video
            check_tracking(file_name, 't', avi_path, save_path, dlc_data=clean_pts)
        elif h5_path is None:
            # plot video without DLC data
            check_tracking(file_name, 't', avi_path, save_path)
    except FileNotFoundError:
        print('missing video file... no output video object is being saved')
        avi_path = None

    try:
        if bonsaitime is True:
            csv_path = os.path.join(data_path, file_name) + '_BonsaiTS.csv'
        elif bonsaitime is False:
            csv_path = os.path.join(data_path, file_name) + '_FlirTS.csv'

        # read in .csv timestamps
        if h5_path is not None:
            time = open_time(csv_path, len(pts))
            xtime = xr.DataArray(time)
        elif h5_path is None:
            time = open_time(csv_path)
            xtime = xr.DataArray(time)
    except FileNotFoundError:
        print('missing time file... output time object is type None')
        csv_path = None
        xtime = None

    return topout, xtime

# eye cam function access
def eye_intake(data_path, file_name, save_path, lik_thresh, pxl_thresh, ell_thresh, eye_pt_num, tear, bonsaitime):
    dir = os.path.join(save_path, file_name)
    if not os.path.exists(dir):
        os.makedirs(dir)

    # get the complete path to the eye trial named in the jupyter notebook
    try:
        h5_path = os.path.join(data_path, file_name) + '.h5'

        # read in .h5 DLC data
        pts, names = open_h5(h5_path)

        # calculate ellipse and get eye angles
        params = eye_tracking(pts, names, save_path, file_name, lik_thresh, pxl_thresh, eye_pt_num, tear)

        # get head angle, plot safety-checks
        # theta = head_angle(clean_pts, names, lik_thresh)

        eyeout = xr.merge([pts, params])
    except FileNotFoundError:
        print('missing DLC file... output DLC xarray object is type None')
        h5_path = None
        eyeout = None

    try:
        avi_path = os.path.join(data_path, file_name) + '.avi'

        if h5_path is not None:
            # plot eye points and ellipses on video
            check_tracking(file_name, 'e', avi_path, save_path, dlc_data=pts, ell_data=params)
        elif h5_path is None:
            # plot video without DLC data
            check_tracking(file_name, 'e', avi_path, save_path)
    except FileNotFoundError:
        print('missing video file... no output video object is being saved')
        avi_path = None

    try:
        if bonsaitime is True:
            csv_path = os.path.join(data_path, file_name) + '_BonsaiTS.csv'
        elif bonsaitime is False:
            csv_path = os.path.join(data_path, file_name) + '_FlirTS.csv'

        # read in .csv timestamps
        if h5_path is not None:
            time = open_time(csv_path, len(pts))
            xtime = xr.DataArray(time)
        elif h5_path is None:
            time = open_time(csv_path)
            xtime = xr.DataArray(time)
    except FileNotFoundError:
        print('missing time file... output time object is type None')
        csv_path = None
        xtime = None

    return eyeout, xtime
