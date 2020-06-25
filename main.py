import argparse
import sys
import pandas as pd
import os
import cv2
import subprocess
import shutil
import deeplabcut
import json
# package imports
from glob import glob
import numpy as np
import xarray as xr
import pandas as pd
import warnings
from multiprocessing import freeze_support

# module imports
from util.read_data import h5_to_xr, find, format_frames, merge_xr_by_timestamps, open_time
from util.track_topdown import topdown_tracking, head_angle, plot_top_vid, get_top_props
from util.track_eye import plot_eye_vid, eye_tracking
from util.track_world import adjust_world, find_pupil_rotation, pupil_rotation_wrapper
from util.analyze_jump import jump_gaze_trace
from util.ephys import format_spikes
import extract_params_from_dlc2

# Checks if path exists, if not then creates directory #######################################################
def CheckPath(basepath, path):
    if path in basepath:
        return basepath
    elif not os.path.exists(os.path.join(basepath, path)):
        os.makedirs(os.path.join(basepath, path))
        print('Added Directory:'+ os.path.join(basepath, path))
        return os.path.join(basepath, path)
    else:
        return os.path.join(basepath, path)

def pars_args():
    # get user inputs
    parser = argparse.ArgumentParser(description='deinterlace videos and adjust timestamps to match')
    parser.add_argument('-d', '--data_path', 
        default='~/Desktop/',
        help='parent directory of all data including timestamps, videos, and any text files of metadata')
    parser.add_argument('-s', '--save_path', type=str,
        default='~/Desktop/',
        help='where to save the data (if not given, data will be saved in the data path with changed names')
    parser.add_argument('-c', '--DLC_json_config_path', 
        default='~/Desktop/',
        help='path to video analysis config file')
    parser.add_argument('-c', '--params_json_config_path', 
        default='~/Desktop/',
        help='path to video analysis config file')
    
    args = parser.parse_args()
    return args

def deinterlace_data(data_path, save_path):
    avi_list = find('*.avi', data_path)
    csv_list = find('*.csv', data_path)
    h5_list = find('*.h5', data_path)

    for this_avi in avi_list:
        # make a save path that keeps the subdirectiries
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
def runDLCbatch(vid_list, config_path):
    for vid in vid_list:
        print('analyzing ' + vid)
        deeplabcut.analyze_videos(config_path, [vid])

def run_DLC_Analysis(DLC_json_config_path):
    # open config file
    with open(DLC_json_config_path, 'r') as fp:
        config = json.load(fp)

    # get each camera type's entry in a list of lists that the json file has in it'
    for cam in config['cams']:
        # there's an entry for the name of the camera to be used
        cam_key = cam[0]
        # and an entry for the config file for that camear type (this will be used by DLC)
        cam_config = cam[1]
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
        runDLCbatch(vids_this_cam, cam_config)
        print('done analyzing ' + str(len(vids_this_cam)) + ' ' + cam_key + ' videos')


def main():
    args = pars_args()

    data_path = os.path.expanduser(args.data_path)
    if not args.save_path:
        save_path = args.data_path
    else: 
        save_path = os.path.expanduser(args.save_path)

    data_path = CheckPath(data_path,'FreelyMovingEphys')
    save_path = CheckPath(save_path,'FreelyMovingEphys')

    # deinterlace data
    deinterlace_data(data_path,save_path)

    DLC_json_config_path = os.path.expanduser(args.DLC_json_config_path)
    DLC_json_config_path = CheckPath(DLC_json_config_path,'FreelyMovingEphys')

    run_DLC_Analysis(DLC_json_config_path)

    params_json_config_path = os.path.expanduser(args.params_json_config_path)
    params_json_config_path = CheckPath(params_json_config_path,'FreelyMovingEphys')

    extract_params_from_dlc2.main(params_json_config_path)