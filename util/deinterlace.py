"""
deinterlace.py

deinterlace videos and shift times to suit the new video frame count

Dec. 02, 2020
"""
# package imports
import argparse, json, sys, os, subprocess, shutil
import cv2
import pandas as pd
import deeplabcut
import numpy as np
import xarray as xr
import warnings
from glob import glob
from multiprocessing import freeze_support
# module imports
from util.format_data import h5_to_xr, format_frames
from util.paths import find, check_path
from util.time import merge_xr_by_timestamps, open_time
from util.track_topdown import topdown_tracking, head_angle1, plot_top_vid, body_props, body_angle
from util.track_eye import plot_eye_vid, eye_tracking
from util.track_world import adjust_world, find_pupil_rotation, pupil_rotation_wrapper
from util.analyze_jump import jump_gaze_trace
from util.ephys import format_spikes

def deinterlace_data(config):
    data_path = config['data_path']
    save_path = config['save_path']

    # find all the files
    avi_list = find('*.avi', data_path)
    csv_list = find('*.csv', data_path)
    h5_list = find('*.h5', data_path)

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
            # optional flip to eye videos
            if config['flip_eye_during_deinter'] is True and 'EYE' in this_avi:
                subprocess.call(['ffmpeg', '-y', '-i', this_avi, '-vf', 'yadif=1:-1:0, vflip', '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a', '256k', avi_out_path])
            # optional flip to world videos
            elif config['flip_world_during_deinter'] is True and 'WORLD' in this_avi:
                subprocess.call(['ffmpeg', '-y', '-i', this_avi, '-vf', 'yadif=1:-1:0, vflip', '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a', '256k', avi_out_path])
            # or, don't flip eye/world videos
            else:
                subprocess.call(['ffmpeg', '-y', '-i', this_avi, '-vf', 'yadif=1:-1:0', '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a', '256k', avi_out_path])
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

if __name__ == '__main__':
    args = pars_args()
    
    json_config_path = os.path.expanduser(args.json_config_path)
    # open config file
    with open(json_config_path, 'r') as fp:
        config = json.load(fp)

    deinterlace_data(config)