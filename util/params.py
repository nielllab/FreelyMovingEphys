"""
params.py
"""
import argparse, json, sys, os, subprocess, shutil
import cv2
import pandas as pd
# os.environ["DLClight"] = "True"
# import deeplabcut
import numpy as np
import xarray as xr
import warnings
from glob import glob
from multiprocessing import freeze_support

from util.format_data import h5_to_xr, format_frames, safe_xr_merge
from util.paths import find, check_path, list_subdirs
from util.time import open_time, merge_xr_by_timestamps
from util.track_topdown import topdown_tracking, head_angle1, plot_top_vid, body_props, body_angle
from util.track_eye import plot_eye_vid, eye_tracking, find_pupil_rotation
from util.track_world import track_LED
from util.ephys import format_spikes
from util.track_ball import ball_tracking
from util.track_side import side_angle, side_tracking
from util.track_imu import read_8ch_imu

def extract_params(config):
    """
    get parameters out of video, optical mouse, IMU, etc.
    INPUTS
        config: options dict
    OUTPUTS
        None
    """
    # get the path to each recording directory
    recording_names = [i for i in list_subdirs(config['animal_dir']) if 'hf' in i or 'fm' in i]
    recording_paths = [os.path.join(config['animal_dir'], recording_name) for recording_name in recording_names]
    recordings_dict = dict(zip(recording_names, recording_paths))
    
    # sort dictionary of {name: path} so fm recordings are always handled first
    sorted_keys = sorted(recordings_dict, key=lambda x:('fm' not in x, x))
    recordings_dict = dict(zip(sorted_keys, [recordings_dict[k] for k in sorted_keys]))

    # go into each trial and get out the camera/ephys types according to what's listed in json file
    for dir_name in recordings_dict:
        config['recording_path'] = recordings_dict[dir_name]
        recording_name = '_'.join(os.path.splitext(os.path.split([i for i in find('*.avi', config['recording_path']) if all(bad not in i for bad in ['plot','IR','rep11','betafpv','side_gaze'])][0])[1])[0].split('_')[:-1])
        
        trial_cam_h5 = find(('*.h5'), config['recording_path'])
        trial_cam_csv = find(('*BonsaiTS*.csv'), config['recording_path'])
        trial_cam_avi = find(('*.avi'), config['recording_path'])
        trial_ball_csv = find(('*BALLMOUSE*.csv'), config['recording_path'])
        trial_imu_bin = find(('*IMU.bin'), config['recording_path'])

        trial_cam_h5 = [x for x in trial_cam_h5 if x != []]
        trial_cam_csv = [x for x in trial_cam_csv if x != []]
        trial_cam_avi = [x for x in trial_cam_avi if x != []]
        trial_ball_csv = [x for x in trial_ball_csv if x != []]
        trial_imu_bin = [x for x in trial_imu_bin if x != []]

        try:
            if config['pose_estimation']['projects'] != [] and config['parameters']['ignore_avis'] is False:
                top_views = [k for k in config['pose_estimation']['projects'].keys() if 'top' in k.lower()]
            else:
                top_views = []
            # iterate through found top views
            for top_view in top_views:
                print('tracking '+top_view+ ' for ' + recording_name)
                # filter the list of files for the current trial to get the topdown view
                try:
                    top_h5 = [i for i in trial_cam_h5 if top_view in i][0]
                except IndexError:
                    top_h5 = None
                if config['parameters']['follow_strict_naming'] is True:
                    top_csv = [i for i in trial_cam_csv if top_view in i][0]
                    top_avi = [i for i in trial_cam_avi if top_view in i and 'plot' not in i and 'calib' not in i][0]
                elif config['parameters']['follow_strict_naming'] is False:
                    top_csv = [i for i in trial_cam_csv if top_view in i][0]
                    top_avi = [i for i in trial_cam_avi if top_view in i and 'plot' not in i][0]
                if top_h5 is not None:
                    # make an xarray of dlc point values out of the found .h5 files
                    # also assign timestamps as coordinates of the xarray
                    topdlc = h5_to_xr(top_h5, top_csv, top_view, config)
                    # clean DLC points up
                    pts = topdown_tracking(topdlc, config, recording_name, top_view) #key_save_path, key, args.lik_thresh, args.coord_cor, args.topdown_pt_num, args.cricket
                    # calculate head angle, body angle, and get properties of the mouse (and cricket if config says one was there)
                    if config['parameters']['topdown']['get_top_thetas'] is True:
                        head_theta = head_angle1(pts, config, recording_name, top_view)
                        body_theta = body_angle(pts, config, recording_name, top_view)
                        head_theta.name = top_view+'_head_angle'; body_theta.name = top_view+'_body_angle'
                    # top_props = body_props(pts, head_theta, config, recording_name, top_view)
                    # make videos (only saved if config says so)
                    if config['parameters']['outputs_and_visualization']['save_avi_vids'] is True:
                        print('plotting points on top video')
                        if config['parameters']['topdown']['get_top_thetas'] is True:
                            plot_top_vid(top_avi, pts, head_ang=head_theta, config=config, trial_name=recording_name, top_view=top_view)
                        elif config['parameters']['topdown']['get_top_thetas'] is False:
                            plot_top_vid(top_avi, pts, head_ang=None, config=config, trial_name=recording_name, top_view=top_view)
                    # make xarray of video frames
                    xr_top_frames = format_frames(top_avi, config); xr_top_frames.name = top_view+'_video'
                    # name and organize data
                    print('saving...')
                    pts.name = top_view+'_pts'
                    if config['parameters']['outputs_and_visualization']['save_nc_vids'] is True:
                        if config['parameters']['topdown']['get_top_thetas'] is True:
                            trial_top_data = safe_xr_merge([pts, head_theta, body_theta, xr_top_frames])
                            trial_top_data.to_netcdf(os.path.join(config['recording_path'], str(recording_name+'_'+top_view+'.nc')), engine='netcdf4', encoding={top_view+'_video':{"zlib": True, "complevel": 4}})
                        elif config['parameters']['topdown']['get_top_thetas'] is False:
                            trial_top_data = safe_xr_merge([pts, xr_top_frames])
                            trial_top_data.to_netcdf(os.path.join(config['recording_path'], str(recording_name+'_'+top_view+'.nc')), engine='netcdf4', encoding={top_view+'_video':{"zlib": True, "complevel": 4}})
                    elif config['parameters']['outputs_and_visualization']['save_nc_vids'] is False:
                        if config['parameters']['topdown']['get_top_thetas'] is True:
                            trial_top_data = safe_xr_merge([pts, head_theta, body_theta])
                            trial_top_data.to_netcdf(os.path.join(config['recording_path'], str(recording_name+'_'+top_view+'.nc')))
                        elif config['parameters']['topdown']['get_top_thetas'] is False:
                            pts.to_netcdf(os.path.join(config['recording_path'], str(recording_name+'_'+top_view+'.nc')))
                elif top_h5 is None:
                    # make an xarray of timestamps without dlc points, since there aren't any for a world camera
                    topdlc = h5_to_xr(pt_path=None, time_path=top_csv, view=top_view, config=config)
                    topdlc.name = top_view+'_pts'
                    # make xarray of video frames
                    if config['parameters']['outputs_and_visualization']['save_nc_vids'] is True:
                        xr_top_frames = format_frames(top_avi, config); xr_top_frames.name = top_view+'_video'
                    if config['parameters']['outputs_and_visualization']['save_nc_vids'] is True:
                        trial_top_data = safe_xr_merge([topdlc, xr_top_frames])
                    elif config['parameters']['outputs_and_visualization']['save_nc_vids'] is False:
                        trial_top_data = topdlc
                    print('saving...')
                    if config['parameters']['outputs_and_visualization']['save_nc_vids'] is True:
                        trial_top_data.to_netcdf(os.path.join(config['recording_path'], str(recording_name+'_'+top_view+'.nc')), engine='netcdf4', encoding={top_view+'_video':{"zlib": True, "complevel": 4}})
                    elif config['parameters']['outputs_and_visualization']['save_nc_vids'] is False:
                        trial_top_data.to_netcdf(os.path.join(config['recording_path'], str(recording_name+'_'+top_view+'.nc')))
        except IndexError:
            print('no TOP trials found for ' + recording_name)

        # analyze eye views
        try:
            if config['pose_estimation']['projects'] != [] and config['parameters']['ignore_avis'] is False:
                eye_sides = [k for k in config['pose_estimation']['projects'].keys() if 'eye' in k.lower()]
            else:
                eye_sides = []
            for eye_side in eye_sides:
                print('tracking ' + eye_side + 'for ' + recording_name)
                # filter the list of files for the current trial to get the eye of this side
                if config['parameters']['follow_strict_naming'] is True:
                    eye_h5 = [i for i in trial_cam_h5 if eye_side in i and 'deinter' in i][0]
                    if len([i for i in trial_cam_csv if eye_side in i and 'formatted' in i]) > 0:
                        eye_csv = [i for i in trial_cam_csv if eye_side in i and 'formatted' in i][0]
                    else:
                        eye_csv = None
                    eye_avi = [i for i in trial_cam_avi if eye_side in i and 'deinter' in i and 'unflipped' not in i][0]
                elif config['parameters']['follow_strict_naming'] is False:
                    eye_h5 = [i for i in trial_cam_h5 if eye_side in i][0]
                    eye_csv = [i for i in trial_cam_csv if eye_side in i][0]
                    eye_avi = [i for i in trial_cam_avi if eye_side in i and 'unflipped' not in i][0]
                # make an xarray of dlc point values out of the found .h5 files
                # also assign timestamps as coordinates of the xarray
                eyedlc = h5_to_xr(eye_h5, eye_csv, eye_side, config=config)
                # get ellipse parameters from dlc points
                print('fitting ellipse to pupil')
                eyeparams = eye_tracking(eyedlc, config, recording_name, eye_side)
                # get pupil rotation and plot video -- slow step
                if config['parameters']['eyes']['get_eye_omega'] is True:
                    rfit, rfit_conv, shift = find_pupil_rotation(eyeparams, config, recording_name)
                    rfit.name = eye_side+'_pupil_radius'; rfit_conv.name = eye_side+'_pupil_radius_conv'
                    shift.name = eye_side+'_omega'
                # make videos (only if config says so)
                if config['parameters']['outputs_and_visualization']['save_avi_vids'] is True:
                    print('plotting parameters on video')
                    plot_eye_vid(eye_avi, eyedlc, eyeparams, config, recording_name, eye_side)
                # make xarray of video frames
                if config['parameters']['outputs_and_visualization']['save_nc_vids'] is True:
                    xr_eye_frames = format_frames(eye_avi, config); xr_eye_frames.name = eye_side+'_video'
                # name and organize data
                eyedlc.name = eye_side+'_pts'; eyeparams.name = eye_side+'_ellipse_params'
                if config['parameters']['outputs_and_visualization']['save_nc_vids'] is True and config['parameters']['eyes']['get_eye_omega'] is True:
                    print('saving...')
                    trial_eye_data = safe_xr_merge([eyedlc, eyeparams, xr_eye_frames, rfit, rfit_conv, shift])
                    trial_eye_data.to_netcdf(os.path.join(config['recording_path'], str(recording_name+'_'+eye_side+'.nc')), engine='netcdf4', encoding={eye_side+'_video':{"zlib": True, "complevel": 4}})
                elif config['parameters']['outputs_and_visualization']['save_nc_vids'] is False and config['parameters']['eyes']['get_eye_omega'] is True:
                    print('saving...')
                    trial_eye_data = safe_xr_merge([eyedlc, eyeparams, rfit, rfit_conv, shift])
                    trial_eye_data.to_netcdf(os.path.join(config['recording_path'], str(recording_name+eye_side+'.nc')))
                elif config['parameters']['outputs_and_visualization']['save_nc_vids'] is True and config['parameters']['eyes']['get_eye_omega'] is False:
                    print('saving...')
                    trial_eye_data = safe_xr_merge([eyedlc, eyeparams, xr_eye_frames])
                    trial_eye_data.to_netcdf(os.path.join(config['recording_path'], str(recording_name+'_'+eye_side+'.nc')), engine='netcdf4', encoding={eye_side+'_video':{"zlib": True, "complevel": 4}})
                elif config['parameters']['outputs_and_visualization']['save_nc_vids'] is False and config['parameters']['eyes']['get_eye_omega'] is False:
                    print('saving...')
                    trial_eye_data = safe_xr_merge([eyedlc, eyeparams])
                    trial_eye_data.to_netcdf(os.path.join(config['recording_path'], str(recording_name+eye_side+'.nc')))
        except IndexError as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)
            print('no EYE trials found for ' + recording_name)

        try:
            if config['parameters']['ignore_avis'] is False:
                # analyze world views
                print('tracking WORLD for ' + recording_name)
                # filter the list of files for the current trial to get the world view of this side
                if config['parameters']['follow_strict_naming'] is True:
                    world_csv = [i for i in trial_cam_csv if 'WORLD' in i and 'formatted' in i][0]
                    world_avi = [i for i in trial_cam_avi if 'WORLD' in i and 'calib' in i][0]
                elif config['parameters']['follow_strict_naming'] is False:
                    world_csv = [i for i in trial_cam_csv if 'WORLD' in i][0]
                    world_avi = [i for i in trial_cam_avi if 'WORLD' in i][0]
                # make an xarray of timestamps without dlc points, since there aren't any for world camera
                worlddlc = h5_to_xr(pt_path=None, time_path=world_csv, view=('WORLD'), config=config)
                worlddlc.name = 'WORLD_times'
                # make xarray of video frames
                if config['parameters']['outputs_and_visualization']['save_nc_vids'] is True:
                    xr_world_frames = format_frames(world_avi, config); xr_world_frames.name = 'WORLD_video'
                # merge but make sure they're not off in lenght by one value, which happens occasionally
                print('saving...')
                if config['parameters']['outputs_and_visualization']['save_nc_vids'] is True:
                    trial_world_data = safe_xr_merge([worlddlc, xr_world_frames])
                    trial_world_data.to_netcdf(os.path.join(config['recording_path'], str(recording_name+'_world.nc')), engine='netcdf4', encoding={'WORLD_video':{"zlib": True, "complevel": 4}})
                elif config['parameters']['outputs_and_visualization']['save_nc_vids'] is False:
                    worlddlc.to_netcdf(os.path.join(config['recording_path'], str(recording_name+'_world.nc')))
        except IndexError:
            print('no WORLD trials found for ' + recording_name)

        # analyze side views
        if config['pose_estimation']['projects'] != [] and config['parameters']['ignore_avis'] is False:
            side_sides = [k for k in config['pose_estimation']['projects'].keys() if 'side' in k.lower()]   
        else:
            side_sides = []
        for side_side in side_sides:
            print('tracking '+ side_side +' for ' + recording_name)
            # filter the list of files for the current trial to get the world view of this side
            side_h5 = [i for i in trial_cam_h5 if side_side in i][0]
            if config['parameters']['follow_strict_naming'] is True:
                side_csv = [i for i in trial_cam_csv if side_side in i and 'formatted' in i][0]
                side_avi = [i for i in trial_cam_avi if side_side in i][0]
            elif config['parameters']['follow_strict_naming'] is False:
                side_csv = [i for i in trial_cam_csv if side_side in i][0]
                side_avi = [i for i in trial_cam_avi if side_side in i][0]
            # make an xarray of timestamps without dlc points, since there aren't any for a world camera
            sidedlc = h5_to_xr(pt_path=side_h5, time_path=side_csv, view='SIDE', config=config)
            # threshold and preprocess dlc pts
            side_pts = side_tracking(sidedlc, config); side_pts.name = 'SIDE_pts'
            # get side parameters
            side_theta = side_angle(side_pts); side_theta.name = 'SIDE_theta'
            # format frames
            if config['parameters']['outputs_and_visualization']['save_nc_vids'] is True:
                xr_side_frames = format_frames(side_avi, config); xr_side_frames.name = 'SIDE_video'
                # save data
                print('saving...')
                trial_side_data = safe_xr_merge([side_pts, side_theta, xr_side_frames])
                trial_side_data.to_netcdf(os.path.join(config['recording_path'], str(recording_name+'_side.nc')), engine='netcdf4', encoding={'SIDE_video':{"zlib": True, "complevel": 4}})
            if config['parameters']['outputs_and_visualization']['save_nc_vids'] is False:
                trial_side_data = safe_xr_merge([side_pts, side_theta])
                trial_side_data.to_netcdf(os.path.join(config['recording_path'], str(recording_name+'_side.nc')))

        # analyze ball movements
        if trial_ball_csv != []:
            print('tracking ball movement for ' + recording_name)
            speed_data = ball_tracking(trial_ball_csv[0], config); speed_data.name = 'BALL_data'
            speed_data.to_netcdf(os.path.join(config['recording_path'], str(recording_name+'_speed.nc')))

        if trial_imu_bin != [] and config['parameters']['imu']['orientation'] is True:
            print('reading imu data for ' + recording_name)
            trial_imu_csv = os.path.join(config['recording_path'],recording_name+'_Ephys_BonsaiBoardTS.csv') # use ephys timestamps
            imu_data = read_8ch_imu(trial_imu_bin[0], trial_imu_csv, config); imu_data.name = 'IMU_data'
            imu_data.to_netcdf(os.path.join(config['recording_path'], str(recording_name+'_imu.nc')))

    print('done with ' + str(len(recordings_dict)) + ' queued recordings')
