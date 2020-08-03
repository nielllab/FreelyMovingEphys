"""
FreelyMovingEphys wrapper functions to access analysis functions from jupyter
nb_wrapper.py

Last modified July 26, 2020
"""

# package imports
import os
import xarray as xr

# module imports
from util.read_data import open_h5, open_time, read_paths, read1path
from util.track_eye import eye_tracking # , check_eye_calibration
from util.track_topdown import topdown_tracking, head_angle
from util.plot_video import check_eye_tracking, check_topdown_tracking

# topdown view function access
def topdown_intake(data_path, file_name, viewext, save_path, lik_thresh, coord_cor, topdown_pt_num, cricket, bonsaitime):
    dir = os.path.join(save_path, file_name)
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('created save directory at ' + str(dir))
    elif os.path.exists(dir):
        print('using existing save directory ' + str(dir))

    # get view extension
    viewext = file_name.split('_')[-1]

    # get the complete path to the topdown trial named in the jupyter notebook
    try:
        print('attempting dlc and time data reads...')
        h5_path = os.path.join(data_path, file_name) + '.h5'

        if bonsaitime is True:
            csv_path = os.path.join(data_path, file_name) + '_BonsaiTS.csv'
        elif bonsaitime is False:
            csv_path = os.path.join(data_path, file_name) + '_FlirTS.csv'

        # build xarray out of paths
        pts, times, names = read_paths(h5_path, csv_path)

        print('attempting dlc data processing...')
        # interpolate, threshold, and plot safety-checks
        pts = xr.Dataset.to_array(pts)
        pts = xr.DataArray.sel(pts, variable='v1')

        clean_pts = topdown_tracking(pts, names, save_path, file_name, lik_thresh, coord_cor, topdown_pt_num, cricket)

        # get head angle, plot safety-checks
        thetas = head_angle(clean_pts, names, lik_thresh, save_path, cricket, file_name)

        pts.name = 'raw_pt_values'
        clean_pts.name = 'output_pt_values'
        thetas.name = 'head_angle_values'
        topout = xr.merge([pts, clean_pts, thetas])
        print('dlc operations complete')
    except FileNotFoundError:
        print('missing either DLC or time file... output DLC xarray object is type None')
        h5_path = None
        topout = None

    try:
        avi_path = os.path.join(data_path, file_name) + '.avi'

        if h5_path is not None:
            print('plotting points on video')
            # plot head points and head angle on video
            check_topdown_tracking(file_name, avi_path, save_path, dlc_data=clean_pts, vext=viewext)
        elif h5_path is None:
            print('saving video without points')
            # plot video without DLC data
            check_topdown_tracking(file_name, avi_path, save_path, vext=viewext)
    except FileNotFoundError:
        print('missing video file... no output video object is being saved')
        avi_path = None

    return topout

# eye cam function access
def eye_intake(data_path, file_name, viewext, save_path, lik_thresh, pxl_thresh, ell_thresh, eye_pt_num, tear, bonsaitime):
    dir = os.path.join(save_path, file_name)
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('created save directory at ' + str(dir))
    elif os.path.exists(dir):
        print('using existing save directory ' + str(dir))

    # get the complete path to the eye trial named in the jupyter notebook
    try:
        print('attempting dlc and time data reads...')
        h5_path = os.path.join(data_path, file_name) + '.h5'

        if bonsaitime is True:
            csv_path = os.path.join(data_path, file_name) + '_BonsaiTS.csv'
        elif bonsaitime is False:
            csv_path = os.path.join(data_path, file_name) + '_FlirTS.csv'

        # read in .h5 DLC data
        pts, times, names = read1path(h5_path, csv_path)

        print('attempting ellipse calculations...')
        # calculate ellipse and get eye angles
        params = eye_tracking(pts, names, save_path, file_name, lik_thresh, pxl_thresh, eye_pt_num, tear)

        # get head angle, plot safety-checks
        # theta = head_angle(clean_pts, names, lik_thresh)

        pts.name = 'raw_pt_values'
        params.name = 'ellipse_param_values'
        eyeout = xr.merge([pts, params])
        print('ellipse calculations complete')
    except FileNotFoundError:
        print('missing DLC file... output DLC xarray object is type None')
        h5_path = None
        eyeout = None

    try:
        avi_path = os.path.join(data_path, file_name) + '.avi'

        if h5_path is not None:
            print('plotting points on video')
            # plot eye points and ellipses on video
            check_eye_tracking(file_name, avi_path, save_path, dlc_data=pts, ell_data=params, vext=viewext)
        elif h5_path is None:
            # plot video without DLC data
            print('saving video without points')
            check_eye_tracking(file_name, avi_path, save_path, vext=viewext)
    except FileNotFoundError:
        print('missing video file... no output video object is being saved')
        avi_path = None

    return eyeout
