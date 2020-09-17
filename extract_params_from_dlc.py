"""
extract_params_from_dlc.py

Extract mouse and prey parameters from videos and DeepLabCut/Anipose outputs.

last modified: September 14, 2020
"""

# package imports
from glob import glob
import os.path
import numpy as np
import xarray as xr
import pandas as pd
import argparse
import warnings
import json
from multiprocessing import freeze_support

# module imports
from util.read_data import h5_to_xr, find
from util.track_topdown import topdown_tracking, head_angle, plot_top_vid, get_top_props
from util.track_eye import plot_eye_vid, eye_tracking
from util.save_data import savecomplete
from util.track_world import adjust_world, find_pupil_rotation, pupil_rotation_wrapper
from util.analyze_jump import jump_gaze_trace

# get user inputs
parser = argparse.ArgumentParser(description='extract mouse and prey parameters from DeepLabCut data and corresponding videos')
parser.add_argument('-c', '--json_config_path', help='path to .json config file')
args = parser.parse_args()

def main():

    with open(args.json_config_path, 'r') as fp:
        config = json.load(fp)

    # get trial name out of each h5 file and make a list of the unique entries
    trial_paths = []
    for h5 in find('*.h5', config['data_path']):
        split_name = h5.split('_')[:-1]
        trial = '_'.join(split_name)
        if trial not in trial_paths:
            trial_paths.append(trial)

    # go into each trial and get out the camera types according to what's listed in json file
    for trial_path in trial_paths:
        trial_cam_h5 = [(trial_path +'_{}.h5').format(name) for name in config['camera_names']]
        if config['use_BonsaiTS'] is True:
            trial_cam_csv = [(trial_path+'_{}_BonsaiTS.csv').format(name) for name in config['camera_names']]
        elif config['use_BonsaiTS'] is False:
            trial_cam_csv = [(trial_path+'_{}_FlirTS.csv').format(name) for name in config['camera_names']]
        trial_cam_avi = [(trial_path+'_{}.avi').format(name) for name in config['camera_names']]
        t_name = os.path.split(trial_path)[1]

        # make the save path if it doesn't exist
        if not os.path.exists(config['save_path']):
            os.makedirs(config['save_path'])

        # analyze the top view
        if 'TOP' in config['camera_names']:
            print('tracking TOP for ' + t_name)
            # filter the list of files for the current trial to get the topdown view
            top_h5 = list(filter(lambda a: 'TOP' in a, trial_cam_h5))[0]
            top_csv = list(filter(lambda a: 'TOP' in a, trial_cam_csv))[0]
            top_avi = list(filter(lambda a: 'TOP' in a, trial_cam_avi))[0]
            # make an xarray of dlc point values out of the found .h5 files
            # also assign timestamps as coordinates of the xarray
            topdlc, topnames = h5_to_xr(top_h5, top_csv, 'TOP')
            # clean DLC points up
            pts, noseX, noseY = topdown_tracking(topdlc, config, t_name) #key_save_path, key, args.lik_thresh, args.coord_cor, args.topdown_pt_num, args.cricket
            # calculate head angle
            print('finding head angle and mouse/cricket properties')
            head_theta = head_angle(pts, noseX, noseY, config, t_name) # args.lik_thresh, key_save_path, args.cricket, key, nose_x, nose_y
            # get mouse properties (and cricket properties if there is one)
            top_props = get_top_props(pts, head_theta, config, t_name)
            # make videos (only saved if config says so)
            if config['save_vids'] is True:
                print('plotting points on top video')
                plot_top_vid(top_avi, pts, head_theta, config, t_name)
            # name and organize data
            pts.name = 'top_dlc_pts'; head_theta.name = 'top_head_theta'; top_props = 'top_properties'
            gathered_top = xr.merge([pts, head_theta, top_props])
            gathered_top['trial'] = t_name
            if trial_path == trial_paths[0]:
                topout = gathered_top
            elif trial_path != trial_paths[0]:
                topout = xr.concat([topout, gathered_top], dim='trial', fill_value=np.nan)

        # analyze eye eye views
        eye_sides = []
        if 'REYE' in config['camera_names']:
            eye_sides.append('R')
        if 'LEYE' in config['camera_names']:
            eye_sides.append('L')
        for i in range(0,len(eye_sides)):
            eye_side = eye_sides[i]
            print('tracking ' + eye_side + 'EYE for ' + t_name)
            # filter the list of files for the current trial to get the topdown view
            eye_h5 = list(filter(lambda a: (eye_side+'EYE') in a, trial_cam_h5))[0]
            eye_csv = list(filter(lambda a: (eye_side+'EYE') in a, trial_cam_csv))[0]
            eye_avi = list(filter(lambda a: (eye_side+'EYE') in a, trial_cam_avi))[0]
            # make an xarray of dlc point values out of the found .h5 files
            # also assign timestamps as coordinates of the xarray
            eyedlc, eyenames = h5_to_xr(eye_h5, eye_csv, (eye_side+'EYE'))
            # get ellipse parameters from dlc points
            print('fitting ellipse to pupil')
            eyeparams = eye_tracking(eyedlc, config, t_name, eye_side)
            # get pupil rotation and plot video -- slow step
            rfit, shift = pupil_rotation_wrapper(eyeparams, config, t_name, eye_side)
            # make videos (only if config says so)
            if config['save_vids'] is True:
                print('plotting parameters on video')
                plot_eye_vid(eye_avi, eyedlc, eyeparams, config, t_name, eye_side)
            # name and organize data
            eyedlc.name = 'eye_dlc_pts'; eyeparams.name = 'eye_ellipse_params'
            try:
                rfit.name = 'eye_radius_fit'; shift.name = 'eye_pupil_rotation'
                gathered_eye = xr.merge([eyedlc, eyeparams, rfit, shift])
            except NameError:
                gathered_eye = xr.merge([eyedlc, eyeparams])
            gathered_eye['trial'] = t_name; gathered_eye['side'] = eye_side
            if trial_path == trial_paths[0]:
                eyeout = gathered_eye
            elif trial_path != trial_paths[0]:
                eyeout = xr.concat([eyeout, gathered_eye], dim='trial', fill_value=np.nan)

        # read in trials that aren't a top, side, eye, or world cameara, and save their points out in the same xarray foramat that others are in
        # otherh5s = sorted([cam for cam in trial_cam_h5 if not any(i in cam for i in ['TOP', 'SIDE', 'EYE'])])
        # otheravis = sorted([cam for cam in trial_cam_h5 if not any(i in cam for i in ['TOP', 'SIDE', 'EYE'])])
        # othercsvs = sorted([cam for cam in trial_cam_h5 if not any(i in cam for i in ['TOP', 'SIDE', 'EYE'])])
        # for num in range(0,len(otherh5s)):
        #     current_otherh5 = otherh5s[num]
        #     current_othercsv = othercsvs[num]
        #     other_name = current_otherh5.split('_')[-1].split('.')[0]
        #     otherdlc, othernames = h5_to_xr(current_otherh5, current_othercsv, other_name)
        #     otherdlc['trial'] = t_name; otherdlc['cam_source'] = other_name
        #     if trial_path == trial_paths[0]:
        #         otherout = otherdlc
        #     elif trial_path != trial_paths[0]:
        #         otherout = xr.concat([eyeout, gathered_eye], dim='trial', fill_value=np.nan)

        # extract parameters from side camera and save out plots, etc.
        # if sidedlc is not None:
        #     print('tracking side view of ' + key)
        #     LEye_params = eyeout.sel(variable='ellipse_param_values', trial=key, view='LEye')
        #     REye_params = eyeout.sel(variable='ellipse_param_values', trial=key, view='REye')
        #     top_params = topout.sel(variable='head_angle_values', trial=key)
        #     # manage the points
        #     side_pts = side_tracking(sidedlc, sidenames, args.lik_thresh)
        #     # get side head angle out
        #     print('calculating head angle in side view')
        #     side_theta = side_angle(side_xr, side_pt_names, front_side_theta_pt, back_side_theta_pt)
        #     sidedlc.name = 'raw_pt_values'
        #     side_pts.name = 'preened_pt_values'
        #     side_theta.name = 'side_angle_values'
        #     gatheredside = xr.merge([sidedlc, side_pts, side_theta])
        #     print('finding side view trace of gaze during jump for ' + viewext)
        #     jump_gaze_trace(args.global_data_path, key_save_path, key, REye_params, LEye_params, top_params, gatheredside, eyeRvidpath, eyeLvidpath, sidevidpath, topvidpath)
        #     jump_cc(args.global_data_path, key_save_path, key, REye_params, LEye_params, top_params, gatheredside)
        # if key_item == key_source[0]:
        #     gatheredside['trial'] = key
        #     sideout = gatheredside
        # elif key_item != key_source[0]:
        #     gatheredside['trial'] = key
        #     sideout = xr.concat([sideout, gatheredside], dim='trial', fill_value=np.nan)

        print('processing of trial ' + t_name + ' is complete, but .nc files wont be saved until the end of all queued trials')

    # save out data at top of parent directory, tell user if nothing was saved
    try:
        savecomplete(topout, config['save_path'], 'tops')
        print('saved topdown data at ' + config['save_path'])
    except NameError:
        print('no topdown data saved')
    try:
        savecomplete(eyeout, config['save_path'], 'eyes')
        print('saved eye data at ' + config['save_path'])
    except NameError:
        print('no eye data saved')
    # try:
    #     savecomplete(sideout, args.global_save_path, 'side')
    #     print('saved side data at ' + args.global_save_path)
    # except NameError:
    #     print('no side data saved')

if __name__ == '__main__':
    main()
