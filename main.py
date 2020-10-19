import argparse, json, sys, os, cv2, subprocess, shutil
import pandas as pd
import deeplabcut
import numpy as np
import xarray as xr
import warnings
from glob import glob
from multiprocessing import freeze_support

# module imports
from util.read_data import h5_to_xr, find, format_frames, merge_xr_by_timestamps, open_time
from util.track_topdown import topdown_tracking, head_angle, plot_top_vid, get_top_props
from util.track_eye import plot_eye_vid, eye_tracking
from util.track_world import adjust_world, find_pupil_rotation, pupil_rotation_wrapper
from util.analyze_jump import jump_gaze_trace
from util.ephys import format_spikes
from util.checkpath import CheckPath


def pars_args():
    # get user inputs
    parser = argparse.ArgumentParser(description='deinterlace videos and adjust timestamps to match')
    parser.add_argument('-c', '--json_config_path', 
        default='~/Desktop/puprot_test_config.json',
        help='path to video analysis config file')
    args = parser.parse_args()
    return args

def deinterlace_data(data_path, save_path):
    avi_list = find('*.avi', data_path)
    csv_list = find('*.csv', data_path)
    h5_list = find('*.h5', data_path)

    if save_path==None:
        save_path = data_path

    for this_avi in avi_list:
        # make a save path that keeps the subdirectories
        current_path = os.path.split(this_avi)[0]
        main_path = current_path.replace(data_path, save_path)
        # get out an key from the name of the video that will be shared with all other data of this trial
        vid_name = os.path.split(this_avi)[1]
        key_pieces = vid_name.split('.')[:-1]
        key = '.'.join(key_pieces)
        print('running on ' + key)
        # then, find those other pieces of the trial using the key
        try:
            this_csv = [i for i in csv_list if key in i][0]
            csv_present = True
        except IndexError:
            csv_present = False
        try:
            this_h5 = [i for i in h5_list if key in i][0]
            h5_present = True
        except IndexError:
            h5_present = False
        # get some info about the video
        cap = cv2.VideoCapture(this_avi)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not os.path.exists(main_path):
            os.makedirs(main_path)
        elif fps == 30:
            print('starting to deinterlace and interpolate on ' + key)
            # deinterlace video with ffmpeg -- will only be done on 30fps videos
            avi_out_path = os.path.join(main_path, (key + 'deinter.avi'))
            subprocess.call(['ffmpeg', '-i', this_avi, '-vf', 'yadif=1:-1:0', '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a', '256k', avi_out_path])
            frame_count_deinter = frame_count * 2
            if csv_present is True:
                # write out the timestamps that have been opened and interpolated over
                csv_out_path = os.path.join(main_path, (key + '_BonsaiTSformatted.csv'))
                csv_out = pd.DataFrame(open_time(this_csv, int(frame_count_deinter)))
                csv_out.to_csv(csv_out_path, index=False)
        else:
            print('frame rate not 30 or 60 for ' + key)

    print('done with ' + str(len(avi_list) + len(csv_list) + len(h5_list)) + ' items')
    print('data saved at ' + save_path)

# run DeepLabCut on a list of video files that share a DLC config file
def runDLCbatch(vid_list, config_path, config):
    for vid in vid_list:
        print('analyzing ' + vid)
        if config['crop_for_dlc'] is True:
            deeplabcut.cropimagesandlabels(config_path, size=(400, 400), userfeedback=False)
        deeplabcut.analyze_videos(config_path, [vid])

def run_DLC_Analysis(config):

    # get each camera type's entry in a list of lists that the json file has in it'
    for cam in config['cams']:
        # there's an entry for the name of the camera to be used
        cam_key = cam
        # and an entry for the config file for that camear type (this will be used by DLC)
        cam_config = config['cams'][cam_key]
        # if it's one of the cameras that needs to needs to be deinterlaced first, make sure and read in the deinterlaced 
        if any(cam_key in s for s in ['REYE','LEYE','WORLD']):
            # find all the videos in the data directory that are from the current camera and are deinterlaced
            vids_this_cam = find('*'+cam_key+'*deinter.avi', config['data_path'])
            print('found ' + str(len(vids_this_cam)) + ' deinterlaced videos from cam_key ' + cam_key)
            # warn the user if there's nothing found
            if len(vids_this_cam) == 0:
                print('no ' + cam_key + ' videos found -- maybe the videos are not deinterlaced yet?')
        else:
            # find all the videos for camera types that don't neeed to be deinterlaced
            vids_this_cam = find('*'+cam_key+'*.avi', config['data_path'])
            print('found ' + str(len(vids_this_cam)) + ' videos from cam_key ' + cam_key)
        # analyze the videos with DeepLabCut
        # this gives the function a list of files that it will iterate over with the same DLC config file
        runDLCbatch(vids_this_cam, cam_config, config)
        print('done analyzing ' + str(len(vids_this_cam)) + ' ' + cam_key + ' videos')


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
        trial_path = trial_unit[0]
        t_name = trial_unit[1]
        trial_cam_h5 = find(('*.h5'), trial_path)
        trial_cam_csv = find(('*BonsaiTS*.csv'), trial_path)
        trial_cam_avi = find(('*.avi'), trial_path)

        trial_cam_h5 = [x for x in trial_cam_h5 if x != []]
        trial_cam_csv = [x for x in trial_cam_csv if x != []]
        trial_cam_avi = [x for x in trial_cam_avi if x != []]

        # make the save path if it doesn't exist
        if config.get('save_path') is not None:
            if not os.path.exists(config['save_path']):
                os.makedirs(config['save_path'])

        # format the ephys data
        if config['has_ephys'] is True:
            print('formatting electrophysiology recordings for ' + t_name)
            # filter the list of files for the current tiral to get to the ephys data
            try:
                trial_spike_times = os.path.join(trial_path,'spike_times.npy')
                trial_spike_clusters = os.path.join(trial_path,'spike_clusters.npy')
                trial_cluster_group = os.path.join(trial_path,'cluster_group.tsv')
                trial_templates = os.path.join(trial_path,'templates.npy')
                trial_ephys_time = os.path.join(trial_path,t_name+'_Ephys_BonsaiTS.csv')
                trial_cluster_info = os.path.join(trial_path,'cluster_info.tsv')
                # read in the data for all spikes during this trial
                ephys = format_spikes(trial_spike_times, trial_spike_clusters, trial_cluster_group, trial_ephys_time, trial_templates, trial_cluster_info, config)
                # save out the data as a json
                ephys.to_json(os.path.join(trial_path, str(t_name+'_ephys.json')))
            except FileNotFoundError as e:
                print(e)
                print('missing one or more ephys files -- assuming no ephys analysis for this trial')

        # analyze top views
        top_views = []
        if 'TOP1' in config['cams']:
            top_views.append('TOP1')
        if 'TOP2' in config['cams']:
            top_views.append('TOP2')
        if 'TOP3' in config['cams']:
            top_views.append('TOP3')
        try:
            for i in range(0,len(top_views)):
                top_view = top_views[i]
                print('tracking TOP for ' + t_name)
                # filter the list of files for the current trial to get the topdown view
                try:
                    top_h5 = [i for i in trial_cam_h5 if top_view in i][0]
                except IndexError:
                    top_h5 = None
                top_csv = [i for i in trial_cam_csv if top_view in i][0]
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
                    trial_top_data.to_netcdf(os.path.join(trial_path, str(t_name+'_'+top_view+'.nc')), engine='netcdf4', encoding={top_view+'_video':{"zlib": True, "complevel": 9}})
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

                    trial_top_data.to_netcdf(os.path.join(trial_path, str(t_name+'_'+top_view+'.nc')), engine='netcdf4', encoding={top_view+'_video':{"zlib": True, "complevel": 9}})
        except IndexError:
            pass

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
            eye_csv = [i for i in trial_cam_csv if (eye_side+'EYE') in i and 'formatted' in i][0]
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
                trial_eye_data = xr.merge([eyedlc, eyeparams, xr_eye_frames])
                trial_eye_data.to_netcdf(os.path.join(trial_path, str(t_name+eye_side+'eye.nc')), engine='netcdf4', encoding={eye_side+'EYE_video':{"zlib": True, "complevel": 9}})
            if config['run_pupil_rotation'] is True:
                trial_eye_data = xr.merge([eyedlc, eyeparams, xr_eye_frames, rfit, shift])
                trial_eye_data.to_netcdf(os.path.join(trial_path, str(t_name+eye_side+'eye.nc')), engine='netcdf4', encoding={eye_side+'EYE_video':{"zlib": True, "complevel": 9}})

        # analyze world views
        if 'WORLD' in config['cams']:
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
            trial_world_data.to_netcdf(os.path.join(trial_path, str(t_name+'world.nc')), engine='netcdf4', encoding={'WORLD_video':{"zlib": True, "complevel": 9}})

    print('done with ' + str(len(trial_units)) + ' queued trials')


def main(args):

    json_config_path = os.path.expanduser(args.json_config_path)

    # open config file
    with open(json_config_path, 'r') as fp:
        config = json.load(fp)


    data_path = os.path.expanduser(config['data_path'])
    if config.get('save_path') is None:
        config['save_path'] = data_path
    else: 
        save_path = os.path.expanduser(config['save_path'])

    ###### deinterlace data
    # deinterlace_data(data_path, save_path=None)
    ###### Get DLC Tracking
    # run_DLC_Analysis(config)
    ###### Extract Parameters from DLC
    extract_params(config)


if __name__ == '__main__':
    args = pars_args()
    main(args)