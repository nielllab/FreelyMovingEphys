"""
FreelyMovingEphys interpolate timestamps to correct for deinterlacing
fix_timing.py

Last modified August 18, 2020
"""
# package imports
import argparse
import sys
import pandas as pd
import os
import cv2
import subprocess
import shutil
import ffmpeg

# module imports (done differently because parallel directories -- is there a better way?)
from util.read_data import open_time, find

# get user inputs
parser = argparse.ArgumentParser(description='Deinterlace videos to go from 30fps to 60fps. Then, interpolate timestamps in subdirectories to correct for video deinterlacing.')
parser.add_argument('data_path', help='path to timestamps')
parser.add_argument('deinter_path', help='path to deinterlaced videos')
parser.add_argument('save_path', help='path to save interpolated timestaps into')
args = parser.parse_args()

avi_list = find('*.avi', args.data_path)
csv_list = find('*.csv', args.data_path)
h5_list = find('*.h5', args.data_path)

for this_avi in avi_list:
    # make a save path that keeps the subdirectiries
    main_path = os.path.split(this_avi)[0]
    main_path = main_path.replace(args.data_path, args.save_path)
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
        avi_out_path = os.path.join(main_path, (key + '.avi'))
        shutil.copyfile(this_avi, avi_out_path)
        if csv_present is True:
            csv_out_path = os.path.join(main_path, (key + '_Bonsai.csv'))
            shutil.copyfile(this_csv, csv_out_path)
        if h5_present is True:
            h5_out_path = os.path.join(main_path, (key + '.h5'))
            shutil.copyfile(this_h5, h5_out_path)
    elif fps == 30:
        # find deinterlaced files instead
        this_avi_deinter = os.path.join(args.deinter_path, (key + '.avi'))
        avi_out_path = os.path.join(main_path, (key + '.avi'))
        shutil.copyfile(this_avi_deinter, avi_out_path)
        if csv_present is True:
            # write out the timestamps that have been opened and interpolated over
            csv_out_path = os.path.join(main_path, (key + '_BonsaiTS.csv'))
            csv_out = pd.DataFrame(open_time(this_csv, int(frame_count)))
            csv_out.to_csv(csv_out_path, index=False)
        if h5_present is True:
            # then, move the h5 files over so they're with the other items in that trial
            h5_out_path = os.path.join(main_path, (key + '.h5'))
            shutil.copyfile(this_h5, h5_out_path)
    else:
        print('frame rate not 30 or 60 for ' + key)

print('done with ' + str(len(avi_list) + len(csv_list) + len(h5_list)) + ' items')
print('data saved at ' + args.save_path)
