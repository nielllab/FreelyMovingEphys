"""
dlc.py

analyze new videos with DeepLabCut given already trained networks

Oct. 16, 2020
"""

import argparse, json, sys, os, subprocess, shutil
import cv2
import pandas as pd
import deeplabcut
import numpy as np
import xarray as xr
import warnings
from glob import glob
from multiprocessing import freeze_support

from util.read_data import h5_to_xr, find, format_frames, merge_xr_by_timestamps, open_time, check_path
from util.track_topdown import topdown_tracking, head_angle1, plot_top_vid, body_props, body_angle
from util.track_eye import plot_eye_vid, eye_tracking
from util.track_world import adjust_world, find_pupil_rotation, pupil_rotation_wrapper
from util.analyze_jump import jump_gaze_trace
from util.ephys import format_spikes

# given a list of videos, run them all on the same DLC config file
def runDLCbatch(vid_list, config_path, config):
    for vid in vid_list:
        print('analyzing ' + vid)
        if config['crop_for_dlc'] is True:
            deeplabcut.cropimagesandlabels(config_path, size=(400, 400), userfeedback=False)
        deeplabcut.analyze_videos(config_path, [vid])

# find files and organize them by which DLC config file they are associated with
def run_DLC_Analysis(config):
    # get each camera type's entry
    for cam in config['cams']:
        # there's an entry for the name of the camera to be used
        cam_key = cam
        # and an entry for the config file for that camear type (this will be used by DLC)
        cam_config = config['cams'][cam_key]
        if cam_config != '':
            # if it's one of the cameras that needs to needs to be deinterlaced first, make sure and read in the deinterlaced 
            if any(cam_key in s for s in ['REYE','LEYE']):
                # find all the videos in the data directory that are from the current camera and are deinterlaced
                if config['run_with_form_time'] is True:
                    vids_this_cam = find('*'+cam_key+'*deinter.avi', config['data_path'])
                elif config['run_with_form_time'] is False:
                    vids_this_cam = find('*'+cam_key+'*.avi', config['data_path'])
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
            vids2run = [vid for vid in vids_this_cam if 'plot' not in vid]
            runDLCbatch(vids2run, cam_config, config)
            print('done analyzing ' + str(len(vids_this_cam)) + ' ' + cam_key + ' videos')

if __name__ == '__main__':
    args = pars_args()
    
    json_config_path = os.path.expanduser(args.json_config_path)
    # open config file
    with open(json_config_path, 'r') as fp:
        config = json.load(fp)

    run_DLC_Analysis(config)