"""
analyze_new_vids.py

analyze new videos from any camera type given the path to a previously trained network

Sept. 25, 2020
"""

# package imports
import deeplabcut
import argparse

# module imports
from util.read_data import find

def analyze_2d(vid_list, data_path, save_path, config_path):
    for vid in vid_list:
        current_path = os.path.split(vid)[0]
        vid_save_path = current_path.replace(data_path, save_path)
        print('analyzing ' + vid)
        deeplabcut.analyze_videos(config_path, vid, destfolder=vid_save_path)

parser = argparse.ArgumentParser(description='analyze new videos using DeepLabCut and Anipose using an already-trained network')
parser.add_argument('-d', '--data_directory', help='data parent directory')
parser.add_argument('-s', '--save_directory', help='destination folder')
parser.add_argument('-c', '--deeplabcut_config_path', help='path to the DLC network config file for this camera type')
parser.add_argument('-k', '--cam_key', help='key specifying which camera type this is (e.g. EYE, TOP)')
args = parser.parse_args()

avi_with_key = find('*'+args.cam_key+'*.avi', args.data_directory)
analyze_2d(avi_with_key, args.data_directory, args.save_directory, args.deeplabcut_config_path)
print('done analyzing ' + str(len(avi_with_key)) + ' ' + args.cam_key + ' videos')

    
    
    