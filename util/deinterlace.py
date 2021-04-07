"""
deinterlace.py

deinterlace eye and world videos
"""
import argparse, json, sys, os, subprocess, shutil
import cv2
import pandas as pd
os.environ["DLClight"] = "True"
import deeplabcut
import numpy as np
import xarray as xr
import warnings
from glob import glob
from multiprocessing import freeze_support

from util.format_data import h5_to_xr, format_frames
from util.paths import find, check_path
from util.time import merge_xr_by_timestamps, open_time
from util.track_topdown import topdown_tracking, head_angle1, plot_top_vid, body_props, body_angle
from util.track_eye import plot_eye_vid, eye_tracking, find_pupil_rotation
from util.track_world import adjust_world, track_LED

def deinterlace_data(config, vid_list=None, time_list=None):
    """
    deinterlace videos and shift times to suit the new video frame count
    will deinterlace data either searching subdirectories automaticlly or using a list of files that are of specific interest
    INPUTS:
        config -- options dict
        vid_list -- list of .avi file paths (optional)
        time_list -- list of .csv file paths (optional)
    OUTPUTS: None
    """
    # get paths out of the config dictionary
    data_path = config['data_path']
    save_path = config['save_path']
    # find all the files assuming no specific files are listed
    if vid_list is None:
        data_path = config['data_path']
        save_path = config['save_path']
        avi_list = find('*.avi', data_path)
        csv_list = find('*.csv', data_path)
    # if a specific list of videos is provided, ignore the config file's data path
    elif vid_list is not None:
        avi_list = vid_list.copy()
        csv_list = time_list.copy()
    # iterate through each video
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
        # open the video
        cap = cv2.VideoCapture(this_avi)
        # get some info about the video
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) # number of total frames
        fps = cap.get(cv2.CAP_PROP_FPS) # frame rate
        # make sure the save directory exists
        if not os.path.exists(main_path):
            os.makedirs(main_path)
        # files that will need to be deinterlaced will be read in with a frame rate of 30 frames/sec
        elif fps == 30:
            print('starting to deinterlace and interpolate on ' + key)
            # create save path
            avi_out_path = os.path.join(main_path, (key + 'deinter.avi'))
            # flip the eye video horizonally and vertically and deinterlace, if this is specified in the config
            if config['flip_eye_during_deinter'] is True and 'EYE' in this_avi:
                subprocess.call(['ffmpeg', '-i', this_avi, '-vf', 'yadif=1:-1:0, vflip, hflip', '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a', '256k', '-y', avi_out_path])
            # flip the world video horizontally and vertically and deinterlace, if this is specificed in the config
            elif config['flip_world_during_deinter'] is True and 'WORLD' in this_avi:
                subprocess.call(['ffmpeg', '-i', this_avi, '-vf', 'yadif=1:-1:0, vflip, hflip', '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a', '256k', '-y', avi_out_path])
            # or, deinterlace without flipping
            else:
                subprocess.call(['ffmpeg', '-i', this_avi, '-vf', 'yadif=1:-1:0', '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a', '256k', '-y', avi_out_path])
            # correct the frame count of the video
            # now that it's deinterlaced, the video has 2x the number of frames as before
            # this will be used to correct the timestamps associated with this video
            frame_count_deinter = frame_count * 2
            if csv_present is True:
                # get the save path for new timestamps
                csv_out_path = os.path.join(main_path, (key + '_BonsaiTSformatted.csv'))
                # read in the exiting timestamps, interpolate to match the new number of steps, and format as dataframe
                csv_out = pd.DataFrame(open_time(this_csv, int(frame_count_deinter)))
                # save new timestamps
                csv_out.to_csv(csv_out_path, index=False)
        else:
            print('frame rate not 30 or 60 for ' + key)

    print('done with ' + str(len(avi_list) + len(csv_list)) + ' items')
    print('deinterlaced videos saved at ' + save_path)

if __name__ == '__main__':
    args = pars_args()
    
    json_config_path = os.path.expanduser(args.json_config_path)
    # open config file
    with open(json_config_path, 'r') as fp:
        config = json.load(fp)

    deinterlace_data(config)