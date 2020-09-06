"""
FreelyMovingEphys deinterlace 30fps videos and interpolate timestamps to match
deinterlace_for_dlc.py

Last modified August 25, 2020
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
parser = argparse.ArgumentParser(description='Deinterlace videos to go from 30fps to 60fps. Then, interpolate timestamps in subdirectories to correct for video deinterlacing.')
parser.add_argument('-d', '--data_path', help='parent directory of all data including timestamps, videos, and any text files of metadata')
parser.add_argument('-s', '--save_path', help='where to rebuld the directories and save deinterlaced/interpolated data along with data copied over.')
args = parser.parse_args()

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
    if fps == 60:
        print('video for ' + key + ' already has 60fps... copying now')
        avi_out_path = os.path.join(main_path, (key + '.avi'))
        shutil.copyfile(this_avi, avi_out_path)
        if csv_present is True:
            csv_out_path = os.path.join(main_path, (key + '_BonsaiTS.csv'))
            csv_out = pd.DataFrame(open_time(this_csv, int(frame_count)))
            csv_out.to_csv(csv_out_path, index=False)
        if h5_present is True:
            h5_out_path = os.path.join(main_path, (key + '.h5'))
            shutil.copyfile(this_h5, h5_out_path)
    elif fps == 30:
        print('starting to deinterlace and interpolate on ' + key)
        # deinterlace video with ffmpeg -- will only be done on 30fps videos
        avi_out_path = os.path.join(main_path, (key + '.avi'))
        subprocess.call(['ffmpeg', '-i', this_avi, '-vf', 'yadif=1:-1:0', '-c:v',
        'libx264', '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a', '256k', avi_out_path])
        frame_count_deinter = frame_count * 2
        if csv_present is True:
            # write out the timestamps that have been opened and interpolated over
            csv_out_path = os.path.join(main_path, (key + '_BonsaiTS.csv'))
            csv_out = pd.DataFrame(open_time(this_csv, int(frame_count_deinter)))
            csv_out.to_csv(csv_out_path, index=False)
        if h5_present is True:
            # then, move the h5 files over so they're with the other items in that trial
            h5_out_path = os.path.join(main_path, (key + '.h5'))
            shutil.copyfile(this_h5, h5_out_path)
    else:
        print('frame rate not 30 or 60 for ' + key)

print('done with ' + str(len(avi_list) + len(csv_list) + len(h5_list)) + ' items')
print('data saved at ' + args.save_path)

print('starting on text files')
txt_list = find('*.txt', args.data_path)
for this_txt in txt_list:
    txt_name = os.path.split(this_txt)[1]
    key_pieces = txt_name.split('.')[:-1]
    key = '.'.join(key_pieces)
    main_path = os.path.split(this_txt)[0]
    main_path = main_path.replace(args.data_path, args.save_path)
    txt_out_path = os.path.join(main_path, (key + '.txt'))
    shutil.copyfile(this_txt, txt_out_path)
print('done')
