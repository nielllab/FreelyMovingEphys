"""
pickle_to_h5.py

convert any DeepLabcut .pickle outputs to .h5 files

Sept. 29, 2020
"""

# package imports
import deeplabcut
import argparse

# module imports
from util.read_data import find

parser = argparse.ArgumentParser(description='deinterlace videos and adjust timestamps to match')
parser.add_argument('-c', '--dlc_config',help='DeepLabCut config .yaml path', default='/home/seuss/Desktop/MathisNetwork2/config.yaml')
parser.add_argument('-d', '--data_path', help='parent directory of pickle files', default='/home/seuss/Desktop/Phils_app/')
args = parser.parse_args()

# deeplabcut.convert_detections2tracklets(args.dlc_config, [args.data_path], videotype='avi')
pickle_list = find('*TOP*bx.pickle', args.data_path)

for vid in pickle_list:
    print('converting to pickle video at ' + vid)
    deeplabcut.convert_raw_tracks_to_h5(args.dlc_config, vid)
print('done converting ' + len(pickle_list) + ' pickles')