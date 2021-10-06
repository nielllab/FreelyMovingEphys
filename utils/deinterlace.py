"""
deinterlace.py
"""
import os, subprocess
import cv2
import pandas as pd
import numpy as np

from utils.paths import find
from utils.time import open_time

def deinterlace_data(config, vid_list=None, time_list=None):
    """ Deinterlace videos, shift times to match the new video frame count.
    Searches subdirectories if vid_list and time_list are both None.
    If lists of files are provided, it will not search subdirectories and instead analyze items in those lists.

    Parameters:
    config (dict): options dict
    vid_list (list): .avi file paths for videos to deinterlace (optional)
    time_list (list): .csv file paths of timestamps matching videos to deinterlace (optional)
    """
    # get paths out of the config dictionary
    data_path = config['animal_dir']
    # find all the files assuming no specific files are listed
    if vid_list is None:
        avi_list = find('*.avi', data_path)
        csv_list = find('*.csv', data_path)
    # if a specific list of videos is provided, ignore the config file's data path
    elif vid_list is not None:
        avi_list = vid_list.copy()
        csv_list = time_list.copy()
    # iterate through each video
    for this_avi in avi_list:
        current_path = os.path.split(this_avi)[0]
        # make a save path that keeps the subdirectories
        # get out an key from the name of the video that will be shared with all other data of this trial
        vid_name = os.path.split(this_avi)[1]
        key_pieces = vid_name.split('.')[:-1]
        key = '.'.join(key_pieces)
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
        if not os.path.exists(current_path):
            os.makedirs(current_path)
        # files that will need to be deinterlaced will be read in with a frame rate of 30 frames/sec
        elif fps == 30:
            print('starting to deinterlace and interpolate on ' + key)
            # create save path
            avi_out_path = os.path.join(current_path, (key + 'deinter.avi'))
            # flip the eye video horizonally and vertically and deinterlace, if this is specified in the config
            if config['deinterlace']['flip_eye_during_deinter'] is True and ('EYE' in this_avi or 'WORLD' in this_avi):
                subprocess.call(['ffmpeg', '-i', this_avi, '-vf', 'yadif=1:-1:0, vflip, hflip, scale=640:480', '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a', '256k', '-y', avi_out_path])
            # or, deinterlace without flipping
            elif config['deinterlace']['flip_eye_during_deinter'] is False and ('EYE' in this_avi or 'WORLD' in this_avi):
                subprocess.call(['ffmpeg', '-i', this_avi, '-vf', 'yadif=1:-1:0, scale=640:480', '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a', '256k', '-y', avi_out_path])
            # correct the frame count of the video
            # now that it's deinterlaced, the video has 2x the number of frames as before
            # this will be used to correct the timestamps associated with this video
            frame_count_deinter = frame_count * 2
            if csv_present is True:
                # get the save path for new timestamps
                csv_out_path = os.path.join(current_path, (key + '_BonsaiTSformatted.csv'))
                # read in the exiting timestamps, interpolate to match the new number of steps, and format as dataframe
                csv_out = pd.DataFrame(open_time(this_csv, int(frame_count_deinter)))
                # save new timestamps
                csv_out.to_csv(csv_out_path, index=False)
        else:
            print('frame rate not 30 or 60 for ' + key)

def flip_video(config):
    vid_list = find('*EYE.avi', config['animal_dir'])
    for this_avi in vid_list:
        vid_name = os.path.split(this_avi)[1]
        key_pieces = vid_name.split('.')[:-1]
        key = '.'.join(key_pieces)
        print(key)
        avi_out_path = os.path.join(os.path.split(this_avi)[0], (key + 'deinter.avi'))
        if config['rotate_only']['flip_x'] and not config['rotate_only']['flip_y']:
            subprocess.call(['ffmpeg', '-i', this_avi, '-vf', 'hflip', '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a', '256k', '-y', avi_out_path])
        elif not config['rotate_only']['flip_x'] and config['rotate_only']['flip_y']:
            subprocess.call(['ffmpeg', '-i', this_avi, '-vf', 'vflip', '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a', '256k', '-y', avi_out_path])
        elif config['rotate_only']['flip_x'] and config['rotate_only']['flip_y']:
            subprocess.call(['ffmpeg', '-i', this_avi, '-vf', 'vflip, hflip', '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a', '256k', '-y', avi_out_path])