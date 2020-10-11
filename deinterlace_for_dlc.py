"""
deinterlace_for_dlc.py

deinterlace videos and shift times to suit the new video frame count
creates a copy of the data directory, including files which aren't going to be deinterlaced
can handle .avi, .csv, .h5 and .txt files

Sept. 28, 2020
"""

# package imports
import argparse
import sys
import pandas as pd
import os
import cv2
import subprocess
import shutil

# module imports
from util.read_data import open_time, find

# get user inputs
parser = argparse.ArgumentParser(description='deinterlace videos and adjust timestamps to match')
parser.add_argument('-d', '--data_path', help='parent directory of all data including timestamps, videos, and any text files of metadata')
parser.add_argument('-s', '--save_path', help='where to save the data (if not given, data will be saved in the data path with changed names')
args = parser.parse_args()

if not args.save_path:
    args.save_path = args.data_path

avi_list = find('*.avi', args.data_path)
csv_list = find('*.csv', args.data_path)
h5_list = find('*.h5', args.data_path)

for this_avi in avi_list:
    # make a save path that keeps the subdirectiries
    current_path = os.path.split(this_avi)[0]
    main_path = current_path.replace(args.data_path, args.save_path)
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
print('data saved at ' + args.save_path)

