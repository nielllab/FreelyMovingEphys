"""
params.py

get parameters from DLC points and generate .nc or .json files

Oct. 26, 2020
"""

import argparse, json, sys, os, cv2, subprocess, shutil
import pandas as pd
import deeplabcut
import numpy as np
import xarray as xr
import warnings
from glob import glob
from multiprocessing import freeze_support

from util.read_data import h5_to_xr, find, format_frames, merge_xr_by_timestamps, open_time, check_path, pars_args
from util.track_topdown import topdown_tracking, head_angle1, plot_top_vid, body_props, body_angle
from util.track_eye import plot_eye_vid, eye_tracking
from util.track_world import adjust_world, find_pupil_rotation, pupil_rotation_wrapper
from util.analyze_jump import jump_gaze_trace
from util.ephys import format_spikes

def extract_params(config):
    # get trial name out of each avi file and make a list of the unique entries
    trial_units = []; name_check = []; path_check = []
    for avi in find('*.avi', config['data_path']):
        if 'plot' not in avi:
            split_name = avi.split('_')[:-1]
            trial = '_'.join(split_name)
            path_to_trial = os.path.join(os.path.split(trial)[0])
            trial_name = os.path.split(trial)[1]
            if trial_name not in name_check:
                trial_units.append([path_to_trial, trial_name])
                path_check.append(path_to_trial); name_check.append(trial_name)

    # go into each trial and get out the camera/ephys types according to what's listed in json file
    for trial_unit in trial_units:
        config['trial_path'] = trial_unit[0]
        t_name = trial_unit[1]
        trial_cam_h5 = find(('*.h5'), config['trial_path'])
        trial_cam_csv = find(('*BonsaiTS*.csv'), config['trial_path'])
        trial_cam_avi = find(('*.avi'), config['trial_path'])

        trial_cam_h5 = [x for x in trial_cam_h5 if x != []]
        trial_cam_csv = [x for x in trial_cam_csv if x != []]
        trial_cam_avi = [x for x in trial_cam_avi if x != []]

        # make the save path if it doesn't exist
        if not os.path.exists(config['save_path']):
            os.makedirs(config['save_path'])

        # format the ephys data
        if config['has_ephys'] is True:
            print('formatting electrophysiology recordings for ' + t_name)
            # filter the list of files for the current tiral to get to the ephys data
            try:
                trial_spike_times = os.path.join(config['trial_path'],'spike_times.npy')
                trial_spike_clusters = os.path.join(config['trial_path'],'spike_clusters.npy')
                trial_cluster_group = os.path.join(config['trial_path'],'cluster_group.tsv')
                trial_templates = os.path.join(config['trial_path'],'templates.npy')
                trial_ephys_time = os.path.join(config['trial_path'],t_name+'_Ephys_BonsaiTS.csv')
                trial_cluster_info = os.path.join(config['trial_path'],'cluster_info.tsv')
                # read in the data for all spikes during this trial
                ephys = format_spikes(trial_spike_times, trial_spike_clusters, trial_cluster_group, trial_ephys_time, trial_templates, trial_cluster_info, config)
                # save out the data as a json
                print('saving...')
                ephys.to_json(os.path.join(config['trial_path'], str(t_name+'_ephys.json')))
            except FileNotFoundError as e:
                print('missing one or more ephys files -- assuming no ephys analysis for this trial')

        try:
            # analyze top views
            top_views = []
            if 'TOP' in config['cams']:
                top_views.append('TOP')
            if 'Top' in config['cams']:
                top_views.append('Top')
            if 'TOP1' in config['cams']:
                top_views.append('TOP1')
            if 'TOP2' in config['cams']:
                top_views.append('TOP2')
            if 'TOP3' in config['cams']:
                top_views.append('TOP3')
            for i in range(0,len(top_views)):
                top_view = top_views[i]
                print('tracking '+top_view+ ' for ' + t_name)
                # filter the list of files for the current trial to get the topdown view
                try:
                    top_h5 = [i for i in trial_cam_h5 if top_view in i][0]
                except IndexError:
                    top_h5 = None
                if config['run_with_form_time_TOP'] is True:
                    top_csv = [i for i in trial_cam_csv if top_view in i and 'formatted' in i][0]
                elif config['run_with_form_time_TOP'] is False:
                    top_csv = [i for i in trial_cam_csv if top_view in i][0]
                top_avi = [i for i in trial_cam_avi if top_view in i][0]
                if top_h5 is not None:
                    # make an xarray of dlc point values out of the found .h5 files
                    # also assign timestamps as coordinates of the xarray
                    topdlc = h5_to_xr(top_h5, top_csv, top_view, config)
                    # clean DLC points up
                    pts = topdown_tracking(topdlc, config, t_name, top_view) #key_save_path, key, args.lik_thresh, args.coord_cor, args.topdown_pt_num, args.cricket
                    # calculate head angle, body angle, and get properties of the mouse (and cricket if config says one was there)
                    if config['run_top_angles'] is True:
                        head_theta = head_angle1(pts, config, t_name, top_view)
                        body_theta = body_angle(pts, config, t_name, top_view)
                        head_theta.name = top_view+'_head_angle'; body_theta.name = top_view+'_body_angle'
                    # top_props = body_props(pts, head_theta, config, t_name, top_view)
                    # make videos (only saved if config says so)
                    if config['save_vids'] is True:
                        print('plotting points on top video')
                        if config['run_top_angles'] is True:
                            plot_top_vid(top_avi, pts, head_ang=head_theta, config=config, trial_name=t_name, top_view=top_view)
                        elif config['run_top_angles'] is False:
                            plot_top_vid(top_avi, pts, head_ang=None, config=config, trial_name=t_name, top_view=top_view)
                    # make xarray of video frames
                    xr_top_frames = format_frames(top_avi, config); xr_top_frames.name = top_view+'_video'
                    # name and organize data
                    print('saving...')
                    pts.name = top_view+'_pts'
                    if config['run_top_angles'] is True:
                        trial_top_data = xr.merge([pts, head_theta, body_theta, xr_top_frames])
                        trial_top_data.to_netcdf(os.path.join(config['trial_path'], str(t_name+'_'+top_view+'.nc')), engine='netcdf4', encoding={top_view+'_video':{"zlib": True, "complevel": 9}})
                    elif config['run_top_angles'] is False:
                        trial_top_data = xr.merge([pts, xr_top_frames])
                        trial_top_data.to_netcdf(os.path.join(config['trial_path'], str(t_name+'_'+top_view+'.nc')), engine='netcdf4', encoding={top_view+'_video':{"zlib": True, "complevel": 9}})
                elif top_h5 is None:
                    # make an xarray of timestamps without dlc points, since there aren't any for a world camera
                    topdlc = h5_to_xr(pt_path=None, time_path=top_csv, view=top_view, config=config)
                    topdlc.name = top_view+'_pts'
                    # make xarray of video frames
                    xr_top_frames = format_frames(top_avi, config); xr_top_frames.name = top_view+'_video'
                    try:
                        trial_top_data = xr.merge([topdlc, xr_top_frames])
                    except ValueError:
                        if len(topdlc) > len(xr_top_frames):
                            trial_top_data = xr.merge([topdlc[:-1], xr_top_frames])
                        elif len(topdlc) < len(xr_top_frames):
                            trial_top_data = xr.merge([topdlc, xr_top_frames[:-1]])
                    print('saving...')
                    trial_top_data.to_netcdf(os.path.join(config['trial_path'], str(t_name+'_'+top_view+'.nc')), engine='netcdf4', encoding={top_view+'_video':{"zlib": True, "complevel": 9}})
        except IndexError:
            print('no TOP trials found for ' + t_name)

        # analyze eye views
        eye_sides = []
        if 'REYE' in config['cams']:
            eye_sides.append('R')
        if 'LEYE' in config['cams']:
            eye_sides.append('L')
        for i in range(0,len(eye_sides)):
            eye_side = eye_sides[i]
            print('tracking ' + eye_side + 'EYE for ' + t_name)
            # filter the list of files for the current trial to get the eye of this side
            eye_h5 = [i for i in trial_cam_h5 if (eye_side+'EYE') in i and 'deinter' in i][0]
            if config['run_with_form_time'] is True:
                eye_csv = [i for i in trial_cam_csv if (eye_side+'EYE') in i and 'formatted' in i][0]
            elif config['run_with_form_time'] is False:
                eye_csv = [i for i in trial_cam_csv if (eye_side+'EYE') in i][0]
            eye_avi = [i for i in trial_cam_avi if (eye_side+'EYE') in i and 'deinter' in i][0]
            # make an xarray of dlc point values out of the found .h5 files
            # also assign timestamps as coordinates of the xarray
            eyedlc = h5_to_xr(eye_h5, eye_csv, (eye_side+'EYE'), config=config)
            # get ellipse parameters from dlc points
            print('fitting ellipse to pupil')
            eyeparams = eye_tracking(eyedlc, config, t_name, eye_side)
            # get pupil rotation and plot video -- slow step
            if config['run_pupil_rotation'] is True:
                rfit, shift = pupil_rotation_wrapper(eyeparams, config, t_name, eye_side)
            # make videos (only if config says so)
            if config['save_vids'] is True:
                print('plotting parameters on video')
                plot_eye_vid(eye_avi, eyedlc, eyeparams, config, t_name, eye_side)
            # make xarray of video frames
            xr_eye_frames = format_frames(eye_avi, config)
            # name and organize data
            eyedlc.name = eye_side+'EYE_pts'; eyeparams.name = eye_side+'EYE_ellipse_params'
            if config['run_pupil_rotation'] is True:
                rfit.name = eye_side+'eye_radius_fit'; shift.name = eye_side+'eye_pupil_rotation'
            xr_eye_frames.name = eye_side+'EYE_video'
            if config['run_pupil_rotation'] is False:
                print('saving...')
                trial_eye_data = xr.merge([eyedlc, eyeparams, xr_eye_frames])
                trial_eye_data.to_netcdf(os.path.join(config['trial_path'], str(t_name+eye_side+'eye.nc')), engine='netcdf4', encoding={eye_side+'EYE_video':{"zlib": True, "complevel": 9}})
            if config['run_pupil_rotation'] is True:
                print('saving...')
                trial_eye_data = xr.merge([eyedlc, eyeparams, xr_eye_frames, rfit, shift])
                trial_eye_data.to_netcdf(os.path.join(config['trial_path'], str(t_name+eye_side+'eye.nc')), engine='netcdf4', encoding={eye_side+'EYE_video':{"zlib": True, "complevel": 9}})

        # analyze world views
        world_sides = []
        if 'WORLD' in config['cams']:
            world_sides.append('WORLD')
        if 'World' in config['cams']:
            world_sides.append('World')    
        for i in range(0,len(world_sides)):
            world_side = world_sides[i]
            print('tracking '+ world_side +' for ' + t_name)
            # filter the list of files for the current trial to get the world view of this side
            if config['run_with_form_time'] is True:
                world_csv = [i for i in trial_cam_csv if world_side in i and 'formatted' in i][0]
            elif config['run_with_form_time'] is False:
                world_csv = [i for i in trial_cam_csv if world_side in i][0]
            world_avi = [i for i in trial_cam_avi if world_side in i and 'deinter' in i][0]
            # make an xarray of timestamps without dlc points, since there aren't any for a world camera
            worlddlc = h5_to_xr(pt_path=None, time_path=world_csv, view=('WORLD'), config=config)
            worlddlc.name = 'WORLD_times'
            # make xarray of video frames
            xr_world_frames = format_frames(world_avi, config); xr_world_frames.name = 'WORLD_video'
            # merge but make sure they're not off in lenght by one value, which happens occasionally
            print('saving...')
            try:
                trial_world_data = xr.merge([worlddlc, xr_world_frames])
            except ValueError:
                if len(worlddlc) > len(xr_world_frames):
                    trial_world_data = xr.merge([worlddlc[:-1], xr_world_frames])
                elif len(worlddlc) < len(xr_world_frames):
                    trial_world_data = xr.merge([worlddlc, xr_world_frames[:-1]])
            trial_world_data.to_netcdf(os.path.join(config['trial_path'], str(t_name+'world.nc')), engine='netcdf4', encoding={'WORLD_video':{"zlib": True, "complevel": 9}})

        # analyze side views
        side_sides = []
        if 'SIDE' in config['cams']:
            side_sides.append('SIDE')
        if 'Side' in config['cams']:
            side_sides.append('Side')    
        for i in range(0,len(side_sides)):
            side_side = side_sides[i]
            print('tracking '+ side_side +' for ' + t_name)
            # filter the list of files for the current trial to get the world view of this side
            side_h5 = [i for i in trial_cam_h5 if side_side in i][0]
            if config['run_with_form_time'] is True:
                side_csv = [i for i in trial_cam_csv if side_side in i and 'formatted' in i][0]
            elif config['run_with_form_time'] is False:
                side_csv = [i for i in trial_cam_csv if side_side in i][0]
            side_avi = [i for i in trial_cam_avi if side_side in i][0]
            # make an xarray of timestamps without dlc points, since there aren't any for a world camera
            sideddlc = h5_to_xr(pt_path=world_h5, time_path=side_csv, view='SIDE', config=config)
            sidedlc.name = 'SIDE_pts'
            # get side parameters

            # format frames
            xr_side_frames = format_frames(side_avi, config); xr_side_frames.name = 'SIDE_video'
            # save data
            trial_side_data = xr.merge([sidedlc, xr_side_frames])
            trial_side_data.to_netcdf(os.path.join(config['trial_path'], str(t_name+'side.nc')), engine='netcdf4', encoding={'SIDE_video':{"zlib": True, "complevel": 9}})

    print('done with ' + str(len(trial_units)) + ' queued trials')

if __name__ == '__main__':
    args = pars_args()
    
    json_config_path = os.path.expanduser(args.json_config_path)
    # open config file
    with open(json_config_path, 'r') as fp:
        config = json.load(fp)

    extract_params(config)
