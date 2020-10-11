"""
extract_params_from_dlc.py

extract parameters of mouse (and prey) in a freely moving environment
takes in videos, timestamps, DeepLabCut outputs, and ephys data
outputs xarrays of original data and parameters
each camera will have its own xarray Dataset, and ephys will have one too
combining these views accross time happens when analysis is happening by hand downstream from this

Oct. 08, 2020
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
from util.read_data import h5_to_xr, find, format_frames, merge_xr_by_timestamps
from util.track_topdown import topdown_tracking, head_angle, plot_top_vid, get_top_props
from util.track_eye import plot_eye_vid, eye_tracking
from util.track_world import adjust_world, find_pupil_rotation, pupil_rotation_wrapper
from util.analyze_jump import jump_gaze_trace
from util.ephys import format_spikes

# get user inputs
parser = argparse.ArgumentParser(description='extract mouse and prey parameters from DeepLabCut data and corresponding videos')
parser.add_argument('-c', '--json_config_path', help='path to .json config file')
args = parser.parse_args()

def main():
    print('config file: ' + args.json_config_path)

    # open config file
    with open(args.json_config_path, 'r') as fp:
        config = json.load(fp)

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
        trial_path = trial_unit[0]
        t_name = trial_unit[1]
        trial_cam_h5 = find((t_name+'*.h5'), trial_path)
        trial_cam_csv = find((t_name+'*BonsaiTSformatted.csv'), trial_path)
        trial_cam_avi = find((t_name+'*.avi'), trial_path)

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
                trial_spike_times = os.path.join(trial_path, t_name,'spike_times.npy')
                trial_spike_clusters = os.path.join(trial_path, t_name,'spike_clusters.npy')
                trial_cluster_group = os.path.join(trial_path, t_name,'cluster_group.tsv')
                trial_templates = os.path.join(trial_path, t_name,'templates.npy')
                trial_ephys_time = os.path.join(trial_path,t_name+'_Ephys_BonsaiTSformatted.csv')
                trial_cluster_info = os.path.join(trial_path, t_name,'cluster_info.tsv')
                # read in the data for all spikes during this trial
                ephys = format_spikes(trial_spike_times, trial_spike_clusters, trial_cluster_group, trial_ephys_time, trial_templates, trial_cluster_info, config)
                # save out the data as a json
                ephys.to_json(os.path.join(config['save_path'], str(t_name+'_ephys.json')))
            except FileNotFoundError:
                print('missing one or more ephys files -- assuming no ephys analysis for this trial')

        # analyze top views
        top_views = []
        if 'TOP1' in config['camera_names']:
            top_views.append('TOP1')
        if 'TOP2' in config['camera_names']:
            top_views.append('TOP2')
        if 'TOP3' in config['camera_names']:
            top_views.append('TOP3')
        for i in range(0,len(top_views)):
            top_view = top_views[i]
            print('tracking TOP for ' + t_name)
            # filter the list of files for the current trial to get the topdown view
            try:
                top_h5 = [i for i in trial_cam_h5 if top_view in i][0]
            except IndexError:
                top_h5 = None
            top_csv = [i for i in trial_cam_csv if top_view in i and 'formatted' in i][0]
            top_avi = [i for i in trial_cam_avi if top_view in i][0]
            if top_h5 is not None:
                # make an xarray of dlc point values out of the found .h5 files
                # also assign timestamps as coordinates of the xarray
                topdlc = h5_to_xr(top_h5, top_csv, top_view, config)
                # clean DLC points up
                pts = topdown_tracking(topdlc, config, t_name, top_view) #key_save_path, key, args.lik_thresh, args.coord_cor, args.topdown_pt_num, args.cricket
                # calculate head angle
                # print('finding head angle and mouse/cricket properties')
                # head_theta = head_angle(pts, noseX, noseY, config, t_name, top_view) # args.lik_thresh, key_save_path, args.cricket, key, nose_x, nose_y
                # get mouse properties (and cricket properties if there is one)
                # top_props = get_top_props(pts, head_theta, config, t_name, top_view)
                # make videos (only saved if config says so)
                if config['save_vids'] is True:
                    print('plotting points on top video')
                    plot_top_vid(top_avi, pts, head_ang=None, config=config, trial_name=t_name, top_view=top_view)
                # make xarray of video frames
                xr_top_frames = format_frames(top_avi, config); xr_top_frames.name = top_view+'_video'
                # name and organize data
                pts.name = top_view+'_pts'# ; head_theta.name = top_view+'_head_angle'; top_props = top_view+'_props'
                trial_top_data = xr.merge([pts, xr_top_frames])#, head_theta, top_props, xr_top_frames])
                trial_top_data.to_netcdf(os.path.join(config['save_path'], str(t_name+'_'+top_view+'.nc')), engine='netcdf4', encoding={top_view+'_video':{"zlib": True, "complevel": 9}})
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

                trial_top_data.to_netcdf(os.path.join(config['save_path'], str(t_name+'_'+top_view+'.nc')), engine='netcdf4', encoding={top_view+'_video':{"zlib": True, "complevel": 9}})

        # analyze eye views
        eye_sides = []
        if 'REYE' in config['camera_names']:
            eye_sides.append('R')
        if 'LEYE' in config['camera_names']:
            eye_sides.append('L')
        for i in range(0,len(eye_sides)):
            eye_side = eye_sides[i]
            print('tracking ' + eye_side + 'EYE for ' + t_name)
            # filter the list of files for the current trial to get the eye of this side
            eye_h5 = [i for i in trial_cam_h5 if (eye_side+'EYE') in i and 'deinter' in i][0]
            eye_csv = [i for i in trial_cam_csv if (eye_side+'EYE') in i and 'formatted' in i][0]
            eye_avi = [i for i in trial_cam_avi if (eye_side+'EYE') in i and 'deinter' in i][0]
            # make an xarray of dlc point values out of the found .h5 files
            # also assign timestamps as coordinates of the xarray
            eyedlc = h5_to_xr(eye_h5, eye_csv, (eye_side+'EYE'), config=config)
            # get ellipse parameters from dlc points
            print('fitting ellipse to pupil')
            eyeparams = eye_tracking(eyedlc, config)
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
                trial_eye_data = xr.merge([eyedlc, eyeparams, xr_eye_frames])
                trial_eye_data.to_netcdf(os.path.join(config['save_path'], str(t_name+eye_side+'eye.nc')), engine='netcdf4', encoding={eye_side+'EYE_video':{"zlib": True, "complevel": 9}})
            if config['run_pupil_rotation'] is True:
                trial_eye_data = xr.merge([eyedlc, eyeparams, xr_eye_frames, rfit, shift])
                trial_eye_data.to_netcdf(os.path.join(config['save_path'], str(t_name+eye_side+'eye.nc')), engine='netcdf4', encoding={eye_side+'EYE_video':{"zlib": True, "complevel": 9}})

        # analyze world views
        if 'WORLD' in config['camera_names']:
            print('tracking WORLD for ' + t_name)
            # filter the list of files for the current trial to get the world view of this side
            world_csv = [i for i in trial_cam_csv if ('WORLD') in i and 'formatted' in i][0]
            world_avi = [i for i in trial_cam_avi if ('WORLD') in i and 'deinter' in i][0]
            # make an xarray of timestamps without dlc points, since there aren't any for a world camera
            worlddlc = h5_to_xr(pt_path=None, time_path=world_csv, view=('WORLD'), config=config)
            worlddlc.name = 'WORLD_times'
            # make xarray of video frames
            xr_world_frames = format_frames(world_avi, config); xr_world_frames.name = 'WORLD_video'
            # merge but make sure they're not off in lenght by one value, which happens occasionally
            try:
                trial_world_data = xr.merge([worlddlc, xr_world_frames])
            except ValueError:
                if len(worlddlc) > len(xr_world_frames):
                    trial_world_data = xr.merge([worlddlc[:-1], xr_world_frames])
                elif len(worlddlc) < len(xr_world_frames):
                    trial_world_data = xr.merge([worlddlc, xr_world_frames[:-1]])
            trial_world_data.to_netcdf(os.path.join(config['save_path'], str(t_name+'world.nc')), engine='netcdf4', encoding={'WORLD_video':{"zlib": True, "complevel": 9}})

    print('done with ' + str(len(trial_units)) + ' queued trials')

if __name__ == '__main__':
    main()
