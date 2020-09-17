"""
nb_wrapper.py

wrapper functions to access analysis functions from jupyter

Last modified September 07, 2020
"""

# package imports
import os
import xarray as xr

# module imports
from util.read_data import open_h5, open_time, h5_to_xr
from util.track_eye import eye_tracking, check_eye_tracking
from util.track_topdown import topdown_tracking, head_angle, check_topdown_tracking
from util.track_cricket import get_cricket_props
from util.track_world import find_pupil_rotation, pupil_rotation_wrapper

# topdown view function access
def topdown_intake(config, trial_name):
config['cricket'], config['use_BonsaiTS']):
    if not os.path.exists(config['save_path']):
        os.makedirs(config['save_path'])
        print('created save directory at ' + str(config['save_path']))
    elif os.path.exists(config['save_path']):
        print('using existing save directory ' + str(config['save_path']))

    # get the complete path to the topdown trial named in the jupyter notebook
    try:
        print('attempting dlc and time data reads...')
        h5_path = os.path.join(config['data_path'], trial_name) + '.h5'

        if config['use_BonsaiTS'] is True:
            csv_path = os.path.join(config['data_path'], trial_name) + '_BonsaiTS.csv'
        elif config['use_BonsaiTS'] is False:
            csv_path = os.path.join(config['data_path'], trial_name) + '_FlirTS.csv'

        # build xarray out of paths
        pts, names = h5_to_xr(h5_path, csv_path, 'TOP')

        print('attempting dlc data processing...')
        # interpolate, threshold, and plot safety-checks
        pts = xr.Dataset.to_array(pts)
        pts = xr.DataArray.sel(pts, variable='v1')

        clean_pts, nose_x, nose_y = topdown_tracking(pts, names, config['save_path'], trial_name, config['lik_thresh'], config['coord_correction'], topdown_pt_num, config['cricket'])

        # get head angle, plot safety-checks
        thetas = head_angle(clean_pts, names, config['lik_thresh'], config['save_path'], config['cricket'], trial_name, nose_x, nose_y)

        if config['cricket'] is True:
            # get out config['cricket'] properties
            cricket_props = get_cricket_props(clean_pts, thetas, config['save_path'], trial_name)

            clean_pts.name = 'output_pt_values'
            thetas.name = 'head_angle_values'
            cricket_props.name = 'cricket_properties'
            topout = xr.merge([clean_pts, thetas, cricket_props])
            print('dlc operations complete')

        if config['cricket'] is False:
            pts.name = 'raw_pt_values'
            clean_pts.name = 'output_pt_values'
            thetas.name = 'head_angle_values'
            topout = xr.merge([clean_pts, thetas])
            print('dlc operations complete')
    except FileNotFoundError:
        print('missing either DLC or time file; output DLC xarray object is type None')
        h5_path = None
        topout = None

    try:
        avi_path = os.path.join(config['data_path'], trial_name) + '.avi'

        if h5_path is not None:
            print('plotting points on video...')
            # plot head points and head angle on video
            check_topdown_tracking(trial_name, avi_path, config['save_path'], dlc_data=clean_pts, head_ang=thetas, vext=viewext)
        elif h5_path is None:
            print('saving video without points...')
            # plot video without DLC data
            check_topdown_tracking(trial_name, avi_path, config['save_path'], vext=viewext)
    except FileNotFoundError:
        print('missing video file; no output video object is being saved')
        avi_path = None

    return topout

# eye cam function access
def eye_intake(data_path, trial_name, viewext, config['save_path'], config['lik_thresh'], pxl_thresh, ell_thresh, eye_pt_num, tear, config['use_BonsaiTS']):
    fig_dir = os.path.join(config['save_path'], trial_name)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        print('created save directory at ' + str(fig_dir))
    elif os.path.exists(fig_dir):
        print('using existing save directory ' + str(fig_dir))

    trial_name_pieces = trial_name.split('_')[:-1]
    trial_name = '_'.join(trial_name_pieces)
    side_letter = viewext[0]
    print(trial_name)

    # get the complete path to the eye trial named in the jupyter notebook
    try:
        print('attempting dlc and time data reads...')
        h5_path = os.path.join(data_path, trial_name) + '.h5'

        if config['use_BonsaiTS'] is True:
            csv_path = os.path.join(data_path, trial_name) + '_BonsaiTS.csv'
        elif config['use_BonsaiTS'] is False:
            csv_path = os.path.join(data_path, trial_name) + '_FlirTS.csv'

        # read in .h5 DLC data
        pts, names = h5_to_xr(h5_path, csv_path, viewext)

        print('attempting ellipse calculations...')
        # calculate ellipse and get eye angles
        params = eye_tracking(pts, names, config['save_path'], trial_name, config['lik_thresh'], pxl_thresh, eye_pt_num, tear)

        rfit, shift = pupil_rotation_wrapper(data_path, trial_name, side_letter, params, fig_dir)

        pts.name = 'raw_pt_values'
        params.name = 'ellipse_param_values'
        eyeout = xr.merge([pts, params])
        print('ellipse calculations complete')
    except FileNotFoundError:
        print('missing DLC file; output DLC xarray object is type None')
        h5_path = None
        eyeout = None

    try:
        avi_path = os.path.join(data_path, trial_name) + '.avi'

        if h5_path is not None:
            print('plotting points on video...')
            # plot eye points and ellipses on video
            check_eye_tracking(trial_name, avi_path, config['save_path'], dlc_data=pts, ell_data=params, vext=viewext)
        elif h5_path is None:
            # plot video without DLC data
            print('saving video without points...')
            check_eye_tracking(trial_name, avi_path, config['save_path'], vext=viewext)
    except FileNotFoundError:
        print('missing video file; no output video object is being saved')
        avi_path = None

    return eyeout, rfit, shift
