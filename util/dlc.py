"""
dlc.py

analyze new videos with DeepLabCut given already trained networks
"""

import argparse, json, sys, os, subprocess, shutil
os.environ["DLClight"] = "True"
import cv2
import pandas as pd
import deeplabcut
import numpy as np
import xarray as xr
import warnings
from glob import glob
from multiprocessing import freeze_support

from util.format_data import h5_to_xr, format_frames
from util.paths import find, check_path
from util.time import open_time, merge_xr_by_timestamps
from util.track_topdown import topdown_tracking, head_angle1, plot_top_vid, body_props, body_angle
from util.track_eye import plot_eye_vid, eye_tracking, find_pupil_rotation

# given a list of videos, run them all on the same DLC config file
def runDLCbatch(vid_list, config_path, config):
    if isinstance(vid_list, list):
        for vid in vid_list:
            print('analyzing ' + vid)
            if (glob(os.path.join(os.path.dirname(vid),'*'+os.path.basename(vid)[-8:-4]+'*.h5'))==[])|(config['pose_estimation']['rerun']==True):
                if config['pose_estimation']['crop_for_dlc'] is True:
                    deeplabcut.cropimagesandlabels(config_path, size=(400, 400), userfeedback=False)
                deeplabcut.analyze_videos(config_path, [vid])
                if config['pose_estimation']['filter_dlc_predictions'] is True:
                    deeplabcut.filterpredictions(config_path, vid)
    else:
        print('analyzing ' + vid_list)
        if config['pose_estimation']['crop_for_dlc'] is True:
            deeplabcut.cropimagesandlabels(config_path, size=(400, 400), userfeedback=False)
        deeplabcut.analyze_videos(config_path, [vid_list])
        if config['pose_estimation']['filter_dlc_predictions'] is True:
                deeplabcut.filterpredictions(config_path, vid_list)

# find files and organize them by which DLC config file they are associated with
def run_DLC_Analysis(config):
    # get each camera type's entry
    for cam in config['pose_estimation']['projects']:
        # there's an entry for the name of the camera to be used
        cam_key = cam
        # and an entry for the config file for that camear type (this will be used by DLC)
        cam_config = config['pose_estimation']['projects'][cam_key]
        if cam_config != '' and cam_config != 'None' and cam_config != None:
            # if it's one of the cameras that needs to needs to be deinterlaced first, make sure and read in the deinterlaced 
            if any(cam_key in s for s in ['REYE','LEYE']):
                # find all the videos in the data directory that are from the current camera and are deinterlaced
                if config['parameters']['follow_strict_naming'] is True:
                    vids_this_cam = find('*'+cam_key+'*deinter.avi', config['animal_dir'])
                elif config['parameters']['follow_strict_naming'] is False:
                    vids_this_cam = find('*'+cam_key+'*.avi', config['animal_dir'])
                # remove unflipped videos generated during jumping analysis
                bad_vids = find('*'+cam_key+'*unflipped*.avi', config['animal_dir'])
                for x in bad_vids:
                    if x in vids_this_cam:
                        vids_this_cam.remove(x)
                ir_vids = find('*IR*.avi', config['animal_dir'])
                for x in ir_vids:
                    if x in vids_this_cam:
                        vids_this_cam.remove(x)
                print('found ' + str(len(vids_this_cam)) + ' deinterlaced videos from cam_key ' + cam_key)
                # warning for user if no videos found
                if len(vids_this_cam) == 0:
                    print('no ' + cam_key + ' videos found -- maybe the videos are not deinterlaced yet?')
            else:
                # find all the videos for camera types that don't neeed to be deinterlaced
                if config['parameters']['follow_strict_naming'] is True:
                    vids_this_cam = find('*'+cam_key+'*.avi', config['animal_dir'])
                elif config['parameters']['follow_strict_naming'] is False:
                    vids_this_cam = find('*'+cam_key+'*.avi', config['animal_dir'])
                print('found ' + str(len(vids_this_cam)) + ' videos from cam_key ' + cam_key)
            # analyze the videos with DeepLabCut
            # this gives the function a list of files that it will iterate over with the same DLC config file
            vids2run = [vid for vid in vids_this_cam if 'plot' not in vid]
            runDLCbatch(vids2run, cam_config, config)
            print('done analyzing ' + str(len(vids_this_cam)) + ' ' + cam_key + ' videos')

def run_DLC_on_LED(dlc_config,vids2run):
    runDLCbatch(vids2run, dlc_config, {'pose_estimation':{'crop_for_dlc':False, 'filter_dlc_predictions':False}})