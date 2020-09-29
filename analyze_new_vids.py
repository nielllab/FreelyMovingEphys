"""
analyze_new_vids.py

analyze new videos from multiple camera types together, given the path
to a previously trained network
run this in the DLC-GPU conda environment with a few added requirements
that can be loaded in from a text file in /FreelyMovingEphys/env/requirements.txt

Sept. 29, 2020
"""

# package imports
import deeplabcut
import argparse
import json
import os

# module imports
from util.read_data import find

# run DeepLabCut on a list of video files that share a DLC config file
def runDLCbatch(vid_list, config_path):
    for vid in vid_list:
        print('analyzing ' + vid)
        deeplabcut.analyze_videos(config_path, [vid])

parser = argparse.ArgumentParser(description='analyze all new videos using DeepLabCut with an already-trained network')
parser.add_argument('-c', '--json_config_path', help='path to video analysis config file')
args = parser.parse_args()

# open config file
with open(args.json_config_path, 'r') as fp:
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
        print('found ' + str(len(vids_this_cam)) + 'videos from cam_key ' + cam_key)
    # analyze the videos with DeepLabCut
    # this gives the function a list of files that it will iterate over with the same DLC config file
    runDLCbatch(vids_this_cam, cam_config)
    print('done analyzing ' + str(len(vids_this_cam)) + ' ' + cam_key + ' videos')
    