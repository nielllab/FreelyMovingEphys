"""
FreelyMovingEphys interpolate timestamps to correct for deinterlacing
fix_timing.py

Last modified August 17, 2020
"""
# package imports
import argparse
import sys
import pandas as pd
import os
import cv2

# module imports (done differently because parallel directories -- is there a better way?)
from util.read_data import open_time, find

# get user inputs
parser = argparse.ArgumentParser(description='Interpolate timestamps in subdirectories to correct for video deinterlacing.')
parser.add_argument('data_path', help='path to timestamps')
parser.add_argument('save_path', help='path to save interpolated timestaps into')
parser.add_argument('camtype', help='camera view to look for (i.e. LEye, REye, LWorld, etc.)')
args = parser.parse_args()

# get lists of all the files that meet input criteria
time_pattern = '*' + args.camtype + '_BonsaiTS.csv'
vid_pattern = '*' + args.camtype + '.avi'
all_time = find(time_pattern, args.data_path)
all_vids = find(vid_pattern, args.data_path)

# make sure the data were found
if len(all_time) == 0:
    print('no timestamp files found of provided camera type; confirm correct path and that data exist')
if len(all_vids) == 0:
    print('no video files found of provided camera type; confirm correct path and that data exist')

for this_time in all_time:
    time_full_name = os.path.split(this_time)[1]
    key_pieces = time_full_name.split('_')[:-1]
    key = '_'.join(key_pieces)
    print(key)

    this_vid = [i for i in all_vids if key in i]

    if len(this_vid)==0:
        print('could not find video associated with current timestamp file, ' + this_time)
        break

    cap = cv2.VideoCapture(this_vid[0]) # indeing here because it's a list of one item
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # open the time file and automaticlly interpolate if needed (must be given number of video frames)
    new_time  = open_time(this_time, num_frames)

    # then, write out the timestamps with a new path
    out_path = os.path.join(args.save_path, (key + '_BonsaiTS.csv'))
    out = pd.DataFrame(new_time)
    out.to_csv(out_path, index=False)
