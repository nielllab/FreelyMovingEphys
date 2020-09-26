"""
analyze_new_vids.py

analyze new videos from any camera type given the path to a previously trained network
run this in the DLC-GPU conda environment, but one which has had pandas, numpy, xarray, opencv, and tqdm installed

Sept. 25, 2020
"""

# package imports
import deeplabcut
import argparse
import json
import os

# module imports
from util.read_data import find

def analyze_2d(vid_list, data_path, save_path, config_path):
    for vid in vid_list:
        current_path = os.path.split(vid)[0]
        vid_save_path = current_path.replace(data_path, save_path)
        print('analyzing ' + vid)
        deeplabcut.analyze_videos(config_path, [vid], destfolder=vid_save_path)

parser = argparse.ArgumentParser(description='analyze new videos using DeepLabCut and Anipose using an already-trained network')
parser.add_argument('-c', '--json_config_path', help='') # this is a json config file to pass in a dictionary of the pathsf
args = parser.parse_args()

# open config file
with open(args.json_config_path, 'r') as fp:
    config = json.load(fp)

avi_with_key = find('*'+config['cam_key']+'*.avi', config['data_path'])
analyze_2d(avi_with_key, config['data_path'], config['save_path'], config['config_path'])
print('done analyzing ' + str(len(avi_with_key)) + ' ' + config['cam_key'] + ' videos')

    
    
    